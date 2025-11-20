# RAG-experiments-2

Experiments with RAG + vector DBs in pyrun.

Run `python query.py`.

Example:
```
Ask a question: How do vector databases work in RAG systems?

--- Retrieved Sources ---
[1] Vector database
[2] Large language model
[3] Retrieval-augmented generation
[4] Retrieval-augmented generation
[5] Retrieval-augmented generation
-------------------------

In Retrieval-Augmented Generation (RAG) systems, vector databases are frequently
used to implement the retrieval component [1]. Text documents relevant to a
specific domain are collected, and feature vectors, also known as "embeddings,"
are computed for each document or document section, typically using a deep
learning network [1]. These embeddings are then stored in a vector database [1, 4].
When a user provides a prompt, its feature vector is computed, and the database
is queried to find the most relevant documents [1]. These retrieved documents
are then added to the context window of a large language model (LLM) [1].
The LLM then generates a response based on both the user's query and the
information from the retrieved documents [2]. This approach enhances LLMs by
allowing them to access and use data beyond their original training set [3].
The retrieved documents are typically found by encoding the query and documents
into vectors and then identifying documents with vectors most similar to the
query vector, which are usually stored in a vector database [2]. Sometimes, to
address potential misses in vector database searches, a traditional text search
is performed, and its results are combined with text chunks linked to vectors
retrieved from the vector search before being fed to the language model [5].
```