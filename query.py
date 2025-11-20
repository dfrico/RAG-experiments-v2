import argparse
import asyncio
import concurrent.futures
import os
from typing import List, Tuple

import lancedb
from dotenv import load_dotenv
from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

def make_paraphrases(llm, q: str, n: int = 3) -> List[str]:
    prompt = (
        f"Generate {n} diverse retrieval paraphrases of this question. "
        "Return ONE paraphrase per line, no numbering, no extra text.\n\n"
        f"Question: {q}"
    )
    text = llm.invoke(prompt)
    lines = [ln.strip(" -\t") for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [q]
    # keep the original too, just in case
    outs = [q] + lines
    # cap and dedupe while preserving order
    seen, uniq = set(), []
    for s in outs:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:n]

async def parallel_search(db: LanceDB, queries: List[str], per_query_k: int) -> List[Tuple[Document, float]]:
    # Run db.similarity_search_with_score in parallel threads (safe: it's I/O bound)
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futs = [
            loop.run_in_executor(pool, db.similarity_search_with_score, q, per_query_k)
            for q in queries
        ]
        results = await asyncio.gather(*futs)
    # Flatten
    flat: List[Tuple[Document, float]] = [item for batch in results for item in batch]
    return flat

def merge_and_rerank(hits: List[Tuple[Document, float]], final_k: int) -> List[Document]:
    # Deduplicate by (text, source) and keep best score for each
    best_by_key = {}
    for doc, score in hits:
        key = (doc.page_content, doc.metadata.get("source"))
        if key not in best_by_key or score < best_by_key[key][1]:
            best_by_key[key] = (doc, score)
    # Rerank by score ascending (LanceDB uses distance; lower is better)
    merged = sorted(best_by_key.values(), key=lambda x: x[1])
    return [d for d, _ in merged[:final_k]]

def build_answer_prompt(docs: List[Document], q: str) -> str:
    context = "\n\n".join(
        f"[{i+1}] {d.page_content}\n(Source: {d.metadata.get('source','unknown')})"
        for i, d in enumerate(docs)
    )
    return (
        "Answer the question using only the context. "
        "Cite sources by their bracket numbers when relevant.\n\n"
        f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
    )

def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--table-name", default="wiki_vectors")
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--llm-model", default="gemini-2.5-flash-lite", help="Gemini model name")
    ap.add_argument("--paraphrases", type=int, default=3, help="Number of paraphrases to generate")
    ap.add_argument("--k", type=int, default=5, help="Final top-k to keep after merge/rerank")
    ap.add_argument("--per-query-k", type=int, default=8, help="k per sub-query before merging")
    args = ap.parse_args()

    # Connect to LanceDB and prepare retrievers/LLM
    dburi=os.getenv("S3_BUCKET_NAME")+"/lancedb"
    conn = lancedb.connect(dburi)
    emb = HuggingFaceEmbeddings(model_name=args.embed_model)
    db = LanceDB(connection=conn, table_name=args.table_name, embedding=emb)
    llm = ChatGoogleGenerativeAI(
        model=args.llm_model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )

    print("Ready. Type 'exit' to quit.")
    while True:
        q = input("\nAsk a question: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        queries = make_paraphrases(llm, q, n=args.paraphrases)
        hits = asyncio.run(parallel_search(db, queries, per_query_k=args.per_query_k))
        docs = merge_and_rerank(hits, final_k=args.k)

        if not docs:
            print("No results.")
            continue

        prompt = build_answer_prompt(docs, q)
        print("\n" + llm.invoke(prompt))

if __name__ == "__main__":
    main()
