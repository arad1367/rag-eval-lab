# %% [markdown]
# # Playground: try every building block by hand
# Run one cell at a time in VS Code (Shift+Enter). Change things and re-run.
# Understanding comes from poking at the pieces, not from reading the final code.

# %% [markdown]
# ## Hour 3 - Embeddings and cosine similarity
# The first run downloads the embedding model (a few dozen MB) into .cache/

# %%
import numpy as np

from rag_lab.config import PAPERS_PATH, settings
from rag_lab.embeddings import FastEmbedder

embedder = FastEmbedder(settings.embed_model, settings.embed_cache_dir)
sentences = [
    "The cat sits on the mat.",
    "A kitten is resting on a rug.",
    "Stock markets fell sharply today.",
]
vectors = embedder.embed_documents(sentences)
print("shape:", vectors.shape)  # (3 sentences, 384 numbers each)
print("length of each vector:", np.linalg.norm(vectors, axis=1))  # all 1.0 (normalized)

# %% Similarity matrix: vectors @ vectors.T = cosine similarity of every pair
print(np.round(vectors @ vectors.T, 2))
# Expect: cat vs kitten is HIGH even with no shared words. cat vs stocks is LOW.

# %% [markdown]
# ## Hour 3 - Chunking

# %%
from rag_lab.data import load_jsonl, make_chunks, split_words

print(split_words("one two three four five six seven", chunk_size=3, overlap=1))

papers = load_jsonl(PAPERS_PATH)
small = make_chunks(papers, chunk_size=50)
big = make_chunks(papers, chunk_size=300)
print(f"{len(papers)} papers -> {len(small)} chunks of 50 words, {len(big)} chunks of 300 words")
print(small[0].text)

# %% [markdown]
# ## Hour 4 - Compare the three retrievers on the same question

# %%
from rag_lab.retrievers import build_retrievers

retrievers = build_retrievers(big, embedder)
question = "How can we make language models hallucinate less?"
for name, retriever in retrievers.items():
    print(f"\n=== {name} ===")
    for r in retriever.search(question, k=3):
        print(f"{r.score:.3f}  {r.chunk.title[:90]}")

# %% Try a question with a rare exact term (an acronym or a dataset name from your data).
# BM25 usually wins on exact terms; dense usually wins on paraphrases.

# %% [markdown]
# ## Hour 5 - Call the LLM, then run the full RAG pipeline (needs your key in .env)

# %%
from rag_lab.llm import LLMClient

llm = LLMClient()
reply = llm.complete("You are a helpful assistant.", "Explain RAG in one sentence.")
print(reply.text)
print("tokens in/out:", reply.prompt_tokens, reply.completion_tokens)

# %%
from rag_lab.rag import RAGPipeline

pipeline = RAGPipeline(retrievers, llm)
result = pipeline.answer(question, k=4)
print(result["answer"])
for s in result["sources"]:
    print(s["rank"], s["title"][:80], s["url"])
print(result["latency_ms"], result["usage"])

# %% Ask something your papers do NOT cover. Does it say "I don't know"?
print(pipeline.answer("What is the best pizza in Naples?")["answer"])
# %%
