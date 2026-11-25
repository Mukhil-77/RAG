"""StudyRAG v2: hybrid retrieval, grounded answers, and a monitoring agent."""
import io, json, os, random, re, threading, time, uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from openai import OpenAI
from docx import Document
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
try:
    from fastembed import TextEmbedding      # best quality; needs Python <= 3.13 (onnxruntime)
except Exception:
    TextEmbedding = None
from pydantic import BaseModel
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
FAST = os.getenv("LLM_FAST_MODEL", MODEL)   # rewriting, judging, canaries
MIN_COSINE = 0.35        # below this the question is treated as "not in the documents"
AGENT_EVERY = 300        # seconds between automatic monitoring cycles
DATA = Path("data"); DATA.mkdir(exist_ok=True)
STORE, LOG, STATE = DATA / "store.json", DATA / "monitor.jsonl", DATA / "agent.json"

app = FastAPI(title="StudyRAG")
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("NVIDIA_API_KEY", ""), timeout=180)
THINK = os.getenv("NVIDIA_THINKING", "0") == "1"   # set to 1 for slower, deeper reasoning
embedder = TextEmbedding("BAAI/bge-small-en-v1.5") if TextEmbedding else None
HASH_DIM = 1024
if embedder is None:
    MIN_COSINE = 0.12   # lighter embedder -> lower similarity scale
    print("fastembed not available (Python 3.14+): using built-in lightweight embedder + BM25")
lock = threading.Lock()
now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")

# ---------- store & index ----------
store = json.loads(STORE.read_text()) if STORE.exists() else {"docs": {}, "chunks": []}
for c in store["chunks"]: c.setdefault("id", uuid.uuid4().hex[:8])
for d in store["docs"].values(): d.setdefault("added", now())
M, BM = None, None
tok = lambda t: re.findall(r"\w+", t.lower())

def rebuild():
    global M, BM
    ch = store["chunks"]
    M = np.array([c["vec"] for c in ch], dtype="float32") if ch else None
    BM = BM25Okapi([tok(c["text"]) for c in ch]) if ch else None

def save(): STORE.write_text(json.dumps(store)); rebuild()
rebuild()

import zlib
def _hash_vec(t):
    w = tok(t); v = np.zeros(HASH_DIM, dtype="float32")
    for g in w + [a + "_" + b for a, b in zip(w, w[1:])]:
        v[zlib.crc32(g.encode()) % HASH_DIM] += 1.0
    v = np.log1p(v)
    return v / (np.linalg.norm(v) or 1.0)

def embed_docs(texts):
    if embedder is None: return np.array([_hash_vec(t) for t in texts], dtype="float32")
    v = np.array(list(embedder.embed(texts)), dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def embed_query(q):
    if embedder is None: return _hash_vec(q)
    v = np.array(list(embedder.query_embed([q]))[0], dtype="float32")   # applies BGE retrieval prefix
    return v / np.linalg.norm(v)

_dim = HASH_DIM if embedder is None else 384
if store["chunks"] and len(store["chunks"][0]["vec"]) != _dim:   # embedder changed -> re-embed saved chunks
    for c, v in zip(store["chunks"], embed_docs([c["text"] for c in store["chunks"]])): c["vec"] = v.tolist()
    save()

# ---------- ingestion ----------
def extract(name, raw):
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(io.BytesIO(raw)).pages)]
    if ext == "docx":
        return [(1, "\n".join(p.text for p in Document(io.BytesIO(raw)).paragraphs))]
    if ext in ("txt", "md"):
        return [(1, raw.decode("utf-8", "ignore"))]
    raise HTTPException(400, "Supported files: PDF, DOCX, TXT, MD")

def chunk(text, size=800, overlap_sents=1):
    """Sentence-aware chunks so ideas are never cut mid-sentence."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", re.sub(r"[ \t]+", " ", text)) if s.strip()]
    out, cur = [], []
    for s in sents:
        if cur and sum(len(x) for x in cur) + len(s) > size:
            out.append(" ".join(cur)); cur = cur[-overlap_sents:]
        cur.append(s)
    if cur: out.append(" ".join(cur))
    return [c for c in out if len(c) > 60]

# ---------- retrieval ----------
def retrieve(query, k=6, doc_ids=None):
    """Hybrid search: vector similarity + BM25 keywords, merged with reciprocal rank fusion."""
    if M is None: return [], 0.0
    idx = [i for i, c in enumerate(store["chunks"]) if not doc_ids or c["doc"] in doc_ids]
    if not idx: return [], 0.0
    cos = M[idx] @ embed_query(query)
    kw = BM.get_scores(tok(query))[idx]
    score = np.zeros(len(idx))
    for arr in (cos, kw):
        for rank, j in enumerate(np.argsort(-arr)): score[j] += 1 / (60 + rank)
    top = np.argsort(-score)[:k]
    return [{**store["chunks"][idx[j]], "cos": float(cos[j])} for j in top], float(cos.max())

def ctx_block(hits):
    return "\n\n".join(f'[{i+1}] ({store["docs"][c["doc"]]["name"]}, p.{c["page"]})\n{c["text"]}' for i, c in enumerate(hits))

def ask(system, user, max_tokens=1500, model=None):
    try:
        r = client.chat.completions.create(
            model=model or MODEL, max_tokens=max_tokens, temperature=0.3, top_p=0.95,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            extra_body={"chat_template_kwargs": {"enable_thinking": THINK}})
        text = r.choices[0].message.content or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    except Exception as e:
        raise HTTPException(502, f"AI service error: {e}")

def rewrite(question, history):
    if not history: return question
    convo = "\n".join(f'{m["role"]}: {m["content"][:400]}' for m in history[-6:])
    try:
        return ask("Rewrite the user's last question as one standalone search query, resolving pronouns using the conversation. Output only the query.",
                   f"{convo}\n\nLast question: {question}", 100, FAST).strip()
    except HTTPException:
        return question

# ---------- monitoring ----------
def log(rec):
    with lock, LOG.open("a") as f: f.write(json.dumps(rec) + "\n")

def read_log(n=200):
    if not LOG.exists(): return []
    return [json.loads(l) for l in LOG.read_text().splitlines()[-n:] if l.strip()]

def parse_json(t):
    m = re.search(r"\{.*\}", t, re.S)
    try: return json.loads(m.group(0)) if m else {}
    except Exception: return {}

def judge(rec, ctx):
    """LLM-as-judge: is the answer supported by the retrieved context, and does it address the question?"""
    try:
        j = parse_json(ask("You audit a RAG system. Reply with JSON only: "
                           '{"faithfulness":0-1,"relevance":0-1,"issue":"one short sentence or empty"}. '
                           "faithfulness = every claim is supported by the context. relevance = the answer addresses the question.",
                           f"Question: {rec['q']}\n\nContext:\n{ctx}\n\nAnswer:\n{rec['answer']}", 200, FAST))
        rec.update(faithfulness=j.get("faithfulness"), relevance=j.get("relevance"), issue=j.get("issue", ""))
    except HTTPException as e:
        rec["issue"] = f"judge failed: {e.detail}"
    log(rec)

PROG = {"running": False, "pct": 0, "step": "", "error": None}

def canary(n=6, cb=None):
    """Synthetic self-test: write a question from a random chunk and check retrieval finds that chunk."""
    ch = store["chunks"]
    if not ch: return None
    hits, rows = 0, []
    for c in random.sample(ch, min(n, len(ch))):
        q = ask("Write one specific question that is answered by the passage. Output only the question.", c["text"], 80, FAST).strip()
        found = c["id"] in [h["id"] for h in retrieve(q, 6)[0]]
        hits += found; rows.append({"q": q, "found": found})
        if cb: cb(len(rows), min(n, len(ch)))
    return {"recall": hits / len(rows), "n": len(rows), "rows": rows, "ts": now()}

def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"alerts": [], "canary": None, "last_run": None, "report": ""}

def health():
    L = read_log(); ev = [r for r in L if r.get("faithfulness") is not None]
    avg = lambda k: round(sum(r[k] for r in ev) / len(ev), 2) if ev else None
    lat = sorted(r["latency_ms"] for r in L if r.get("latency_ms"))
    return {"queries": len(L), "faithfulness": avg("faithfulness"), "relevance": avg("relevance"),
            "p95_ms": lat[int(len(lat) * .95) - 1] if lat else None,
            "no_answer_rate": round(sum(r.get("no_match", False) for r in L) / len(L), 2) if L else None,
            "errors": sum(1 for r in L if r.get("error")),
            "docs": len(store["docs"]), "chunks": len(store["chunks"])}

def agent_cycle(diagnose=False, track=False):
    """Monitoring agent: checks index integrity, runs canaries, inspects live metrics, auto-repairs, raises alerts."""
    P = lambda p, t: PROG.update(pct=p, step=t) if track else None
    P(5, "Checking index integrity")
    alerts, st, h = [], state(), health()
    add = lambda sev, msg, fix: alerts.append({"severity": sev, "message": msg, "fix": fix, "ts": now()})
    dims = {len(c["vec"]) for c in store["chunks"]}
    if len(dims) > 1 or (M is None and store["chunks"]):
        reindex(); add("info", "Index was inconsistent; the agent rebuilt it automatically.", "No action needed.")
    P(10, "Checking documents for searchable text")
    empty = [d["name"] for k, d in store["docs"].items() if not any(c["doc"] == k for c in store["chunks"])]
    if empty: add("warning", f"Documents with no searchable text: {', '.join(empty)}", "Likely scanned images. Run OCR and re-upload.")
    P(15, "Running retrieval self-tests")
    try: st["canary"] = canary(cb=lambda i, n: P(15 + int(45 * i / n), f"Self-test question {i} of {n}"))
    except HTTPException as e: add("critical", f"Canary test could not run: {e.detail}", "Check NVIDIA_API_KEY and network access.")
    c = st["canary"]
    P(62, "Reviewing live answer quality")
    if c and c["recall"] < 0.6: add("critical", f"Retrieval recall is {c['recall']:.0%} on self-tests.", "Re-index, or check for noisy documents.")
    elif c and c["recall"] < 0.8: add("warning", f"Retrieval recall is {c['recall']:.0%} on self-tests.", "Review recent failed questions.")
    if h["faithfulness"] is not None and h["faithfulness"] < 0.75: add("critical", f"Answer faithfulness is {h['faithfulness']}: answers may contain unsupported claims.", "Check retrieved passages for the flagged questions.")
    if h["no_answer_rate"] and h["no_answer_rate"] > 0.4: add("warning", f"{h['no_answer_rate']:.0%} of questions found nothing relevant.", "Users may be asking about topics that are not uploaded.")
    if h["p95_ms"] and h["p95_ms"] > 15000: add("warning", f"Slow responses (p95 {h['p95_ms']/1000:.1f}s).", "Reduce top-k or check network.")
    if h["errors"]: add("warning", f"{h['errors']} recent requests failed.", "See the interaction log.")
    st.update(alerts=alerts, last_run=now())
    if diagnose:
        P(70, "Writing diagnosis")
        worst = sorted([r for r in read_log() if r.get("faithfulness") is not None], key=lambda r: r["faithfulness"] + r["relevance"])[:5]
        st["report"] = ask("You are an SRE for a RAG study app. Write a short diagnosis: what is healthy, what is failing, the likely root cause, and 3 concrete fixes. Plain text, no markdown headers.",
                           json.dumps({"health": h, "canary": c, "alerts": alerts, "worst": [{k: r.get(k) for k in ('q', 'standalone', 'top_cos', 'faithfulness', 'relevance', 'issue')} for r in worst]}), 700)
    P(95, "Saving results")
    STATE.write_text(json.dumps(st)); return st

def agent_loop():
    time.sleep(20)
    while True:
        try: agent_cycle()
        except Exception as e: print("agent error:", e)
        time.sleep(AGENT_EVERY)

@app.on_event("startup")
def start(): threading.Thread(target=agent_loop, daemon=True).start()

def reindex():
    for d in store["docs"]: pass
    vecs = embed_docs([c["text"] for c in store["chunks"]]) if store["chunks"] else []
    for c, v in zip(store["chunks"], vecs): c["vec"] = v.tolist()
    save()

# ---------- API ----------
@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    added = []
    for f in files:
        items = [(p, c) for p, t in extract(f.filename, await f.read()) for c in chunk(t)]
        if not items: raise HTTPException(400, f"No readable text in {f.filename}. Scanned PDFs need OCR.")
        did = uuid.uuid4().hex[:8]
        store["docs"][did] = {"name": f.filename, "chunks": len(items), "added": now()}
        store["chunks"] += [{"id": uuid.uuid4().hex[:8], "doc": did, "page": p, "text": c, "vec": v.tolist()}
                            for (p, c), v in zip(items, embed_docs([c for _, c in items]))]
        added.append(did)
    save(); return added

@app.get("/api/docs")
def docs(): return [{"id": k, **v} for k, v in store["docs"].items()]

@app.delete("/api/docs/{did}")
def delete(did: str):
    store["docs"].pop(did, None); store["chunks"] = [c for c in store["chunks"] if c["doc"] != did]; save(); return {"ok": True}

class Chat(BaseModel):
    question: str
    history: list[dict] = []
    doc_ids: list[str] = []

@app.post("/api/chat")
def chat(req: Chat, bg: BackgroundTasks):
    t0 = time.time()
    if not store["chunks"]: raise HTTPException(400, "Upload a document first.")
    sq = rewrite(req.question, req.history)
    hits, top = retrieve(sq, 6, req.doc_ids)
    rec = {"id": uuid.uuid4().hex[:8], "ts": now(), "q": req.question, "standalone": sq, "top_cos": round(top, 3), "n_hits": len(hits)}
    if top < MIN_COSINE:
        answer, rec["no_match"] = "I couldn't find anything about that in your documents. Try rephrasing, or upload material that covers it.", True
        hits = []
    else:
        answer = ask("You are a precise study tutor. Answer ONLY using the numbered context passages. Cite every claim with the passage number in square brackets, like [1] or [2] (never a bare number). "
                     "Write formulas in LaTeX between $ ... $ (inline) or $$ ... $$ (display). Use Markdown tables with | separators when comparing items. "
                     "If the context only partly answers, say what is missing. Never use outside knowledge. Be clear and well organised.",
                     f"Context:\n{ctx_block(hits)}\n\nQuestion: {req.question}")
    rec.update(answer=answer, latency_ms=int((time.time() - t0) * 1000))
    if hits: bg.add_task(judge, rec, ctx_block(hits))
    else: log(rec)
    return {"answer": answer, "sources": [{"n": i + 1, "doc": store["docs"][c["doc"]]["name"], "page": c["page"], "text": c["text"][:350], "score": round(c["cos"], 2)} for i, c in enumerate(hits)]}

class Study(BaseModel):
    mode: str
    topic: str = ""
    doc_ids: list[str] = []

PROMPTS = {"summary": "Write a structured study summary with short headings and bullet points of the key ideas.",
           "quiz": "Write 6 multiple-choice questions (A-D) that test understanding, then an answer key with one-line explanations.",
           "flashcards": "Create 10 flashcards. Format each as 'Q: ...' then 'A: ...' on the next line."}

@app.post("/api/study")
def study(req: Study):
    if req.mode not in PROMPTS: raise HTTPException(400, "Unknown mode")
    hits, _ = retrieve(req.topic or "main concepts, definitions and key facts", 14, req.doc_ids)
    if not hits: raise HTTPException(400, "Upload a document first.")
    return {"result": ask("You are a study assistant. Use ONLY the provided material.",
                          f"Material:\n{ctx_block(hits)}\n\nTask: {PROMPTS[req.mode]}" + (f" Focus on: {req.topic}" if req.topic else ""), 2500)}

@app.get("/api/monitor")
def monitor(): return {"health": health(), "agent": state(), "recent": read_log(25)[::-1], "interval": AGENT_EVERY}

@app.post("/api/monitor/run")
def monitor_run():
    if PROG["running"]: return PROG
    PROG.update(running=True, pct=2, step="Starting", error=None)
    def work():
        try: agent_cycle(diagnose=True, track=True); PROG.update(pct=100, step="Complete")
        except Exception as e: PROG.update(error=str(e))
        finally: PROG["running"] = False
    threading.Thread(target=work, daemon=True).start()
    return PROG

@app.get("/api/monitor/progress")
def monitor_progress(): return PROG

@app.post("/api/monitor/reindex")
def monitor_reindex(): reindex(); return {"ok": True}

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
def index(): return FileResponse("static/index.html")
