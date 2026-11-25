# StudyRAG

Upload PDFs, DOCX, TXT or MD files, ask questions with cited answers, and generate summaries, quizzes and flashcards.

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export NVIDIA_API_KEY=nvapi-...                     # Windows: set NVIDIA_API_KEY=...
uvicorn main:app --reload
```
Open http://localhost:8000 (the UI lives in `static/index.html`)

## How it works
1. **Ingest**: text is extracted per page, split into ~900-character overlapping chunks, and embedded locally (fastembed BGE model on Python <= 3.13; on 3.14+ a built-in lightweight embedder is used automatically).
2. **Retrieve**: your question is embedded and the closest chunks are found by cosine similarity.
3. **Generate**: the LLM (NVIDIA Nemotron) answers only from those chunks and cites them as [1], [2]...

Data is saved in `data/store.json`. Set `LLM_MODEL` to change the model, `NVIDIA_THINKING=1` for deeper (slower) reasoning.
Scanned PDFs (images only) need OCR first.

- **Retrieval**: sentence-aware chunks, hybrid search (embeddings + BM25 keywords), BGE query prefix, follow-up question rewriting, and a relevance threshold so the app says "not found" instead of guessing.
- **Monitor tab**: an agent runs every 5 minutes (and on demand). It runs retrieval self-tests, scores live answers for faithfulness and relevance, checks index health, auto-rebuilds a broken index, raises alerts, and writes a diagnosis.
- Re-upload old documents or click "Rebuild index" after upgrading.
