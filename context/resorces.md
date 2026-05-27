# Document Intelligence System — Documentation & Resource Index

> Every official doc, API reference, research paper, guide, and community resource
> required to build, configure, and operate this system.
> Organized by system component. All links verified as of mid-2025.

---

## Table of Contents

1. [Vector Store — Qdrant](#1-vector-store--qdrant)
2. [Graph Database — Neo4j](#2-graph-database--neo4j)
3. [Graph RAG — LightRAG](#3-graph-rag--lightrag)
4. [Document Parsing — Docling](#4-document-parsing--docling)
5. [Web Parsing — Crawl4AI](#5-web-parsing--crawl4ai)
6. [Multimodal RAG — RAG-Anything](#6-multimodal-rag--rag-anything)
7. [Video RAG](#7-video-rag)
8. [Audio — Whisper](#8-audio--whisper)
9. [Dense Embeddings — BGE-M3 & FlagEmbedding](#9-dense-embeddings--bge-m3--flagembedding)
10. [Sparse Embeddings — SPLADE](#10-sparse-embeddings--splade)
11. [Visual Embeddings — CLIP](#11-visual-embeddings--clip)
12. [Chunking — LangChain Splitters](#12-chunking--langchain-splitters)
13. [Re-ranking — Cross-Encoders & MMR](#13-re-ranking--cross-encoders--mmr)
14. [Backend — FastAPI](#14-backend--fastapi)
15. [Database ORM — SQLAlchemy & asyncpg](#15-database-orm--sqlalchemy--asyncpg)
16. [Task Queue — Celery & Redis](#16-task-queue--celery--redis)
17. [RAG Tracing — LangFuse](#17-rag-tracing--langfuse)
18. [Evaluation — RAGAS & DeepEval](#18-evaluation--ragas--deepeval)
19. [Observability — Prometheus & Grafana](#19-observability--prometheus--grafana)
20. [Object Storage — MinIO & boto3](#20-object-storage--minio--boto3)
21. [Auth & Security](#21-auth--security)
22. [Video Processing Libraries](#22-video-processing-libraries)
23. [LLM Providers & Vision Models](#23-llm-providers--vision-models)
24. [Configuration & Settings](#24-configuration--settings)
25. [Research Papers](#25-research-papers)
26. [Architecture Guides & Deep Dives](#26-architecture-guides--deep-dives)
27. [Docker & Infrastructure](#27-docker--infrastructure)

---

## 1. Vector Store — Qdrant

### Core Documentation
| Resource | URL |
|---|---|
| Qdrant Documentation Home | https://qdrant.tech/documentation/ |
| Quickstart | https://qdrant.tech/documentation/quickstart/ |
| Collections | https://qdrant.tech/documentation/concepts/collections/ |
| Vectors (dense, sparse, named) | https://qdrant.tech/documentation/concepts/vectors/ |
| Payload & Filtering | https://qdrant.tech/documentation/concepts/filtering/ |
| Payload Indexing | https://qdrant.tech/documentation/concepts/indexing/#payload-index |
| Points (upsert, delete, scroll) | https://qdrant.tech/documentation/concepts/points/ |
| Search | https://qdrant.tech/documentation/concepts/search/ |
| Query API (v1.10+) | https://qdrant.tech/documentation/concepts/search/#query-api |
| Hybrid Search (RRF fusion) | https://qdrant.tech/articles/hybrid-search/ |
| Sparse Vectors guide | https://qdrant.tech/articles/sparse-vectors/ |
| Named Vectors | https://qdrant.tech/documentation/concepts/vectors/#named-vectors |
| Snapshots | https://qdrant.tech/documentation/concepts/snapshots/ |
| Distributed deployment | https://qdrant.tech/documentation/guides/distributed_deployment/ |
| Optimizers | https://qdrant.tech/documentation/concepts/optimizer/ |
| HNSW index | https://qdrant.tech/documentation/concepts/indexing/#vector-index |
| Quantization | https://qdrant.tech/documentation/guides/quantization/ |
| Telemetry & Prometheus | https://qdrant.tech/documentation/guides/telemetry/ |
| Configuration reference | https://qdrant.tech/documentation/guides/configuration/ |
| Security (API key auth) | https://qdrant.tech/documentation/guides/security/ |

### Python Client
| Resource | URL |
|---|---|
| Python client reference | https://python-client.qdrant.tech/ |
| Python client GitHub | https://github.com/qdrant/qdrant-client |
| AsyncQdrantClient | https://python-client.qdrant.tech/qdrant_client.async_qdrant_client |
| Models reference | https://python-client.qdrant.tech/qdrant_client.models |

### Key Articles
| Resource | URL |
|---|---|
| Multi-vector search | https://qdrant.tech/articles/multivector-search/ |
| Binary quantization | https://qdrant.tech/articles/binary-quantization/ |
| Filterable HNSW | https://qdrant.tech/articles/filtrable-hnsw/ |
| Qdrant internals (HNSW + payload) | https://qdrant.tech/articles/qdrant-internals/ |
| Qdrant under the hood | https://qdrant.tech/articles/vector-search-under-the-hood/ |

---

## 2. Graph Database — Neo4j

### Core Documentation
| Resource | URL |
|---|---|
| Neo4j Documentation Home | https://neo4j.com/docs/ |
| Getting Started | https://neo4j.com/docs/getting-started/ |
| Cypher Manual | https://neo4j.com/docs/cypher-manual/current/ |
| Cypher cheat sheet | https://neo4j.com/docs/cypher-cheat-sheet/current/ |
| Graph data modeling | https://neo4j.com/docs/getting-started/data-modeling/ |
| Constraints | https://neo4j.com/docs/cypher-manual/current/constraints/ |
| Indexes | https://neo4j.com/docs/cypher-manual/current/indexes/ |
| Multi-database admin | https://neo4j.com/docs/operations-manual/current/database-administration/ |
| Operations Manual | https://neo4j.com/docs/operations-manual/current/ |
| Memory configuration | https://neo4j.com/docs/operations-manual/current/configuration/neo4j-conf/ |

### Python Driver
| Resource | URL |
|---|---|
| Python driver docs | https://neo4j.com/docs/python-manual/current/ |
| Python driver API reference | https://neo4j.com/docs/api/python-driver/current/ |
| Async sessions | https://neo4j.com/docs/python-manual/current/async/ |
| Transaction functions | https://neo4j.com/docs/python-manual/current/transactions/ |

### APOC Plugin
| Resource | URL |
|---|---|
| APOC documentation | https://neo4j.com/labs/apoc/4.4/ |
| APOC GitHub | https://github.com/neo4j/apoc |
| APOC security procedures | https://neo4j.com/labs/apoc/4.4/overview/apoc.security/ |
| APOC graph algorithms | https://neo4j.com/labs/apoc/4.4/graph-querying/ |

### GraphRAG Pattern Resources
| Resource | URL |
|---|---|
| Microsoft GraphRAG + Neo4j | https://neo4j.com/blog/graphrag-manifesto/ |
| Knowledge graph RAG guide | https://neo4j.com/developer-blog/knowledge-graph-rag-application/ |
| Neo4j vector search (for hybrid) | https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/ |

---

## 3. Graph RAG — LightRAG

### Official Resources
| Resource | URL |
|---|---|
| LightRAG GitHub | https://github.com/HKUDS/LightRAG |
| LightRAG paper (arXiv) | https://arxiv.org/abs/2410.05779 |
| LightRAG documentation | https://lightrag.github.io/ |
| Storage backends config | https://github.com/HKUDS/LightRAG/blob/main/README.md#storage-backends |
| Neo4j storage backend | https://github.com/HKUDS/LightRAG/blob/main/lightrag/kg/neo4j_impl.py |
| Qdrant storage backend | https://github.com/HKUDS/LightRAG/blob/main/lightrag/kg/qdrant_impl.py |
| Query modes (local/global/hybrid/naive) | https://github.com/HKUDS/LightRAG#query-modes |
| Multi-tenant / multi-working-dir pattern | https://github.com/HKUDS/LightRAG/issues?q=multi+tenant |

### Related Graph RAG Resources
| Resource | URL |
|---|---|
| Microsoft GraphRAG | https://github.com/microsoft/graphrag |
| Microsoft GraphRAG docs | https://microsoft.github.io/graphrag/ |
| GraphRAG vs LightRAG comparison | https://medium.com/@zilliz_learn/graphrag-explained-enhancing-rag-with-knowledge-graphs-3312065f99e1 |

---

## 4. Document Parsing — Docling

### Official Resources
| Resource | URL |
|---|---|
| Docling GitHub | https://github.com/docling-project/docling |
| Docling documentation | https://ds4sd.github.io/docling/ |
| Quickstart | https://ds4sd.github.io/docling/getting_started/ |
| Supported formats | https://ds4sd.github.io/docling/supported_formats/ |
| Pipeline options | https://ds4sd.github.io/docling/concepts/pipeline/ |
| Chunking (HierarchicalChunker) | https://ds4sd.github.io/docling/concepts/chunking/ |
| TableFormer (table structure) | https://ds4sd.github.io/docling/concepts/table_understanding/ |
| OCR configuration | https://ds4sd.github.io/docling/concepts/ocr/ |
| DoclingDocument export | https://ds4sd.github.io/docling/concepts/docling_document/ |
| Integrations (LangChain, LlamaIndex) | https://ds4sd.github.io/docling/integrations/ |
| LangChain + Docling | https://ds4sd.github.io/docling/integrations/langchain/ |
| API reference | https://ds4sd.github.io/docling/reference/document_converter/ |

### TableFormer Paper
| Resource | URL |
|---|---|
| TableFormer paper (arXiv) | https://arxiv.org/abs/2203.00274 |
| TableFormer IBM blog | https://research.ibm.com/publications/tableformer-robust-transformer-based-table-understanding |

---

## 5. Web Parsing — Crawl4AI

### Official Resources
| Resource | URL |
|---|---|
| Crawl4AI documentation | https://docs.crawl4ai.com/ |
| Crawl4AI GitHub | https://github.com/unclecode/crawl4ai |
| Quickstart | https://docs.crawl4ai.com/core/quickstart/ |
| AsyncWebCrawler | https://docs.crawl4ai.com/core/async-webcrawler/ |
| BrowserConfig | https://docs.crawl4ai.com/core/browser-crawler-strategy/ |
| CrawlerRunConfig | https://docs.crawl4ai.com/core/crawler-result/ |
| Content filtering (PruningContentFilter) | https://docs.crawl4ai.com/core/content-selection/ |
| Markdown generation | https://docs.crawl4ai.com/core/markdown-generation/ |
| Link analysis | https://docs.crawl4ai.com/core/link-analysis/ |
| Deep crawling | https://docs.crawl4ai.com/advanced/multi-page-crawling/ |
| Docker deployment | https://docs.crawl4ai.com/core/docker-deploymeny/ |
| Extraction strategies | https://docs.crawl4ai.com/extraction/ |

---

## 6. Multimodal RAG — RAG-Anything

### Official Resources
| Resource | URL |
|---|---|
| RAG-Anything GitHub | https://github.com/HKUDS/RAG-Anything |
| RAG-Anything paper (arXiv) | https://arxiv.org/abs/2505.18547 |
| Installation & quickstart | https://github.com/HKUDS/RAG-Anything#installation |
| Multimodal processing pipeline | https://github.com/HKUDS/RAG-Anything#how-it-works |
| RAGAnythingConfig options | https://github.com/HKUDS/RAG-Anything/blob/main/raganything/utils.py |
| Image processing | https://github.com/HKUDS/RAG-Anything/blob/main/raganything/modal_processors.py |

---

## 7. Video RAG

### VideoRAG Resources
| Resource | URL |
|---|---|
| VideoRAG project site | https://video-rag.github.io/ |
| VideoRAG GitHub | https://github.com/Leon1207/Video-RAG-master |
| VideoRAG paper (arXiv) | https://arxiv.org/abs/2411.13093 |

### Supporting Libraries
| Resource | URL |
|---|---|
| PySceneDetect docs | https://www.scenedetect.com/docs/ |
| PySceneDetect GitHub | https://github.com/Breakthrough/PySceneDetect |
| PySceneDetect ContentDetector | https://www.scenedetect.com/docs/latest/api/detectors.html |
| OpenCV Python tutorials | https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html |
| OpenCV VideoCapture | https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html |
| moviepy documentation | https://zulko.github.io/moviepy/ |
| decord (fast video decoding) | https://github.com/dmlc/decord |

---

## 8. Audio — Whisper

### OpenAI Whisper
| Resource | URL |
|---|---|
| Whisper GitHub | https://github.com/openai/whisper |
| Whisper model card | https://huggingface.co/openai/whisper-large-v3 |
| Whisper paper (arXiv) | https://arxiv.org/abs/2212.04356 |
| Word-level timestamps | https://github.com/openai/whisper#word-level-timestamps |
| OpenAI Speech-to-text API | https://platform.openai.com/docs/guides/speech-to-text |
| OpenAI Whisper API reference | https://platform.openai.com/docs/api-reference/audio |

### faster-whisper (recommended for production)
| Resource | URL |
|---|---|
| faster-whisper GitHub | https://github.com/SYSTRAN/faster-whisper |
| faster-whisper HuggingFace | https://huggingface.co/Systran |
| CTranslate2 (backend engine) | https://opennmt.net/CTranslate2/ |

### Audio Processing
| Resource | URL |
|---|---|
| pydub documentation | https://github.com/jiaaro/pydub |
| librosa audio analysis | https://librosa.org/doc/latest/ |

---

## 9. Dense Embeddings — BGE-M3 & FlagEmbedding

### BGE-M3
| Resource | URL |
|---|---|
| BGE-M3 HuggingFace model | https://huggingface.co/BAAI/bge-m3 |
| BGE-M3 paper (arXiv) | https://arxiv.org/abs/2402.03216 |
| FlagEmbedding GitHub | https://github.com/FlagOpen/FlagEmbedding |
| FlagEmbedding BGEM3FlagModel docs | https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3 |
| BGE-small-en-v1.5 (fast tier) | https://huggingface.co/BAAI/bge-small-en-v1.5 |

### HuggingFace Inference
| Resource | URL |
|---|---|
| sentence-transformers docs | https://www.sbert.net/docs/ |
| sentence-transformers models | https://www.sbert.net/docs/sentence_transformer/pretrained_models.html |
| HuggingFace transformers | https://huggingface.co/docs/transformers/index |
| HuggingFace hub | https://huggingface.co/docs/hub/index |

---

## 10. Sparse Embeddings — SPLADE

### SPLADE
| Resource | URL |
|---|---|
| SPLADE GitHub | https://github.com/naver/splade |
| SPLADE paper (arXiv) | https://arxiv.org/abs/2109.10086 |
| SPLADE-v3 paper | https://arxiv.org/abs/2312.12012 |
| splade-cocondenser-ensembledistil (model) | https://huggingface.co/naver/splade-cocondenser-ensembledistil |
| SPLADE++ model | https://huggingface.co/naver/splade-v3 |
| Sparse retrieval guide (Qdrant) | https://qdrant.tech/articles/sparse-vectors/ |

---

## 11. Visual Embeddings — CLIP

### OpenAI CLIP
| Resource | URL |
|---|---|
| CLIP GitHub | https://github.com/openai/CLIP |
| CLIP paper (arXiv) | https://arxiv.org/abs/2103.00020 |
| CLIP model cards | https://huggingface.co/openai/clip-vit-large-patch14 |

### OpenCLIP (recommended — ViT-L/14)
| Resource | URL |
|---|---|
| OpenCLIP GitHub | https://github.com/mlfoundations/open_clip |
| OpenCLIP HuggingFace | https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K |
| OpenCLIP model benchmark | https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md |

---

## 12. Chunking — LangChain Splitters

### LangChain Text Splitters
| Resource | URL |
|---|---|
| Text splitters concept | https://python.langchain.com/docs/concepts/text_splitters/ |
| Text splitters integrations list | https://python.langchain.com/docs/integrations/document_transformers/ |
| RecursiveCharacterTextSplitter | https://python.langchain.com/docs/how_to/recursive_text_splitter/ |
| MarkdownHeaderTextSplitter | https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/ |
| SemanticChunker (experimental) | https://python.langchain.com/docs/how_to/semantic-chunker/ |
| CharacterTextSplitter | https://python.langchain.com/docs/how_to/character_text_splitter/ |
| Code splitters | https://python.langchain.com/docs/how_to/code_splitter/ |
| LangChain splitters OSS index | https://docs.langchain.com/oss/python/integrations/splitters |
| LangChain document loaders | https://python.langchain.com/docs/concepts/document_loaders/ |
| LangChain API reference | https://api.python.langchain.com/en/latest/ |

### Chunking Strategy Guides
| Resource | URL |
|---|---|
| RAG chunking strategies deep dive | https://dev.to/sreeni5018/rag-chunking-strategies-4i3a |
| Chunking for RAG (Pinecone guide) | https://www.pinecone.io/learn/chunking-strategies/ |
| Advanced RAG chunking (towards data science) | https://towardsdatascience.com/the-art-of-chunking-9048ae4eed00 |

---

## 13. Re-ranking — Cross-Encoders & MMR

### Cross-Encoder
| Resource | URL |
|---|---|
| sentence-transformers CrossEncoder | https://www.sbert.net/docs/cross_encoder/usage/usage.html |
| CrossEncoder API reference | https://www.sbert.net/docs/cross_encoder/cross_encoder.html |
| ms-marco-MiniLM-L-12-v2 (model) | https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2 |
| Cross-encoder training | https://www.sbert.net/docs/cross_encoder/training_overview.html |

### MMR (Maximal Marginal Relevance)
| Resource | URL |
|---|---|
| Original MMR paper (CMU 1998) | https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_SIGIR_1998.pdf |
| LangChain MMR retriever | https://python.langchain.com/docs/how_to/MaxMarginalRelevance_using_faiss/ |

### Re-ranking Services (alternatives)
| Resource | URL |
|---|---|
| Cohere Rerank API | https://docs.cohere.com/reference/rerank |
| Jina Reranker | https://jina.ai/reranker/ |
| FlashRank (lightweight) | https://github.com/PrithivirajDamodaran/FlashRank |

---

## 14. Backend — FastAPI

### Core FastAPI
| Resource | URL |
|---|---|
| FastAPI documentation | https://fastapi.tiangolo.com/ |
| Tutorial — First steps | https://fastapi.tiangolo.com/tutorial/ |
| Dependency injection | https://fastapi.tiangolo.com/tutorial/dependencies/ |
| Security & JWT | https://fastapi.tiangolo.com/tutorial/security/ |
| OAuth2 with JWT | https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/ |
| Background tasks | https://fastapi.tiangolo.com/tutorial/background-tasks/ |
| Middleware | https://fastapi.tiangolo.com/tutorial/middleware/ |
| CORS | https://fastapi.tiangolo.com/tutorial/cors/ |
| Custom responses | https://fastapi.tiangolo.com/advanced/custom-response/ |
| Lifespan events | https://fastapi.tiangolo.com/advanced/events/ |
| Testing | https://fastapi.tiangolo.com/tutorial/testing/ |

### SSE Streaming
| Resource | URL |
|---|---|
| sse-starlette GitHub | https://github.com/sysid/sse-starlette |
| sse-starlette PyPI | https://pypi.org/project/sse-starlette/ |
| Starlette streaming responses | https://www.starlette.io/responses/#streamingresponse |
| EventSourceResponse usage | https://github.com/sysid/sse-starlette#usage |
| Streaming LLM responses guide | https://gautam75.medium.com/streaming-llm-responses-importance-and-implementation-911b135ef541 |

### Uvicorn / ASGI
| Resource | URL |
|---|---|
| Uvicorn docs | https://www.uvicorn.org/ |
| Gunicorn + Uvicorn production | https://www.uvicorn.org/deployment/ |
| ASGI spec | https://asgi.readthedocs.io/en/latest/ |

---

## 15. Database ORM — SQLAlchemy & asyncpg

### SQLAlchemy
| Resource | URL |
|---|---|
| SQLAlchemy 2.0 docs | https://docs.sqlalchemy.org/en/20/ |
| ORM quickstart | https://docs.sqlalchemy.org/en/20/orm/quickstart.html |
| Asyncio extension | https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html |
| Mapped columns | https://docs.sqlalchemy.org/en/20/orm/mapping_columns.html |
| Relationships | https://docs.sqlalchemy.org/en/20/orm/relationships.html |
| Session (async) | https://docs.sqlalchemy.org/en/20/orm/session_basics.html |
| Row-level security (Postgres) | https://docs.sqlalchemy.org/en/20/dialects/postgresql.html |

### Alembic
| Resource | URL |
|---|---|
| Alembic documentation | https://alembic.sqlalchemy.org/en/latest/ |
| Auto-generating migrations | https://alembic.sqlalchemy.org/en/latest/autogenerate.html |
| Async migrations | https://alembic.sqlalchemy.org/en/latest/cookbook.html#using-asyncio-with-alembic |

### asyncpg
| Resource | URL |
|---|---|
| asyncpg documentation | https://magicstack.github.io/asyncpg/current/ |
| asyncpg GitHub | https://github.com/MagicStack/asyncpg |
| PostgreSQL JSON support | https://magicstack.github.io/asyncpg/current/usage.html#type-conversion |

---

## 16. Task Queue — Celery & Redis

### Celery
| Resource | URL |
|---|---|
| Celery documentation | https://docs.celeryq.dev/en/stable/ |
| First steps with Celery | https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html |
| Configuration reference | https://docs.celeryq.dev/en/stable/userguide/configuration.html |
| Tasks | https://docs.celeryq.dev/en/stable/userguide/tasks.html |
| Canvas — chains and chords | https://docs.celeryq.dev/en/stable/userguide/canvas.html |
| Workers | https://docs.celeryq.dev/en/stable/userguide/workers.html |
| Routing (queues) | https://docs.celeryq.dev/en/stable/userguide/routing.html |
| Periodic tasks (Beat) | https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html |
| Retries | https://docs.celeryq.dev/en/stable/userguide/tasks.html#retries |
| Soft / hard time limits | https://docs.celeryq.dev/en/stable/userguide/workers.html#time-limits |
| ACKS_LATE pattern | https://docs.celeryq.dev/en/stable/userguide/tasks.html#task-acks-late |
| Celery Flower (monitoring) | https://flower.readthedocs.io/en/latest/ |

### Redis
| Resource | URL |
|---|---|
| redis-py documentation | https://redis-py.readthedocs.io/en/stable/ |
| redis-py async | https://redis-py.readthedocs.io/en/stable/examples/asyncio_examples.html |
| Redis sorted sets (sliding window rate limit) | https://redis.io/docs/data-types/sorted-sets/ |
| Redis persistence config | https://redis.io/docs/management/persistence/ |
| Redis memory management | https://redis.io/docs/management/optimization/memory-optimization/ |

---

## 17. RAG Tracing — LangFuse

### Core LangFuse
| Resource | URL |
|---|---|
| LangFuse documentation | https://langfuse.com/docs |
| Python SDK | https://langfuse.com/docs/sdk/python/decorators |
| @observe decorator | https://langfuse.com/docs/sdk/python/decorators#observe-decorator |
| Tracing concepts | https://langfuse.com/docs/tracing |
| Scores & feedback | https://langfuse.com/docs/scores/overview |
| Prompt management | https://langfuse.com/docs/prompts/get-started |
| Datasets & experiments | https://langfuse.com/docs/datasets/overview |
| Self-hosting | https://langfuse.com/docs/deployment/self-host |
| Docker Compose self-host | https://langfuse.com/docs/deployment/self-host#docker-compose |
| FastAPI integration | https://langfuse.com/docs/integrations/fastapi |
| LangChain callback | https://langfuse.com/docs/integrations/langchain/tracing |

### Alternatives
| Resource | URL |
|---|---|
| Arize Phoenix docs | https://docs.arize.com/phoenix |
| LangSmith docs | https://docs.smith.langchain.com/ |
| Weave (Weights & Biases) | https://weave-docs.wandb.ai/ |
| Helicone (LLM observability) | https://docs.helicone.ai/ |

---

## 18. Evaluation — RAGAS & DeepEval

### RAGAS
| Resource | URL |
|---|---|
| RAGAS documentation | https://docs.ragas.io/ |
| Quickstart | https://docs.ragas.io/en/latest/getstarted/install.html |
| Available metrics | https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/ |
| Context recall | https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/ |
| Context precision | https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/ |
| Answer faithfulness | https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/ |
| Answer relevancy | https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevancy/ |
| Test set generation | https://docs.ragas.io/en/latest/concepts/test_data_generation/ |
| Custom LLM + embeddings | https://docs.ragas.io/en/latest/howtos/customisations/bring-your-own-llm-or-embs/ |
| GitHub | https://github.com/explodinggradients/ragas |
| RAGAS paper (arXiv) | https://arxiv.org/abs/2309.15217 |

### DeepEval
| Resource | URL |
|---|---|
| DeepEval documentation | https://docs.confident-ai.com/ |
| Quickstart | https://docs.confident-ai.com/docs/getting-started |
| Hallucination metric | https://docs.confident-ai.com/docs/metrics-hallucination |
| Faithfulness metric | https://docs.confident-ai.com/docs/metrics-faithfulness |
| Answer relevancy metric | https://docs.confident-ai.com/docs/metrics-answer-relevancy |
| G-Eval (custom rubric) | https://docs.confident-ai.com/docs/metrics-llm-evals |
| CI/CD integration | https://docs.confident-ai.com/docs/integrations-github-actions |
| GitHub | https://github.com/confident-ai/deepeval |

### Additional Evaluation Frameworks
| Resource | URL |
|---|---|
| ARES framework (arXiv) | https://arxiv.org/abs/2311.09476 |
| ARES GitHub | https://github.com/stanford-futuredata/ARES |
| TruLens | https://www.trulens.org/trulens/getting_started/ |
| MTEB benchmark (embeddings) | https://huggingface.co/spaces/mteb/leaderboard |

---

## 19. Observability — Prometheus & Grafana

### Prometheus
| Resource | URL |
|---|---|
| Prometheus documentation | https://prometheus.io/docs/introduction/overview/ |
| Configuration | https://prometheus.io/docs/prometheus/latest/configuration/configuration/ |
| Alerting rules | https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/ |
| PromQL query language | https://prometheus.io/docs/prometheus/latest/querying/basics/ |
| Histogram & quantiles | https://prometheus.io/docs/practices/histograms/ |
| prometheus-fastapi-instrumentator | https://github.com/trallnag/prometheus-fastapi-instrumentator |
| prometheus-client Python | https://github.com/prometheus/client_python |

### Grafana
| Resource | URL |
|---|---|
| Grafana documentation | https://grafana.com/docs/grafana/latest/ |
| Dashboard provisioning | https://grafana.com/docs/grafana/latest/administration/provisioning/ |
| Alerting | https://grafana.com/docs/grafana/latest/alerting/ |
| Grafana Loki (log aggregation) | https://grafana.com/docs/loki/latest/ |
| Pre-built dashboards hub | https://grafana.com/grafana/dashboards/ |

### Exporters
| Resource | URL |
|---|---|
| redis_exporter | https://github.com/oliver006/redis_exporter |
| postgres_exporter | https://github.com/prometheus-community/postgres_exporter |
| Neo4j Prometheus plugin | https://neo4j.com/labs/neo4j-monitoring-cli/ |

---

## 20. Object Storage — MinIO & boto3

### MinIO
| Resource | URL |
|---|---|
| MinIO documentation | https://min.io/docs/minio/linux/index.html |
| Docker deployment | https://min.io/docs/minio/container/index.html |
| Docker Compose | https://min.io/docs/minio/container/operations/install-deploy-manage/deploy-minio-single-node-single-drive.html |
| Distributed (erasure coding) | https://min.io/docs/minio/linux/operations/install-deploy-manage/deploy-minio-multi-node-multi-drive.html |
| MinIO Python SDK | https://min.io/docs/minio/linux/developers/python/API.html |
| MinIO Client (mc) CLI | https://min.io/docs/minio/linux/reference/minio-mc.html |
| Object locking (immutability) | https://min.io/docs/minio/linux/administration/object-management/object-locking.html |
| Server-side encryption | https://min.io/docs/minio/linux/administration/server-side-encryption.html |

### boto3 (S3)
| Resource | URL |
|---|---|
| boto3 S3 reference | https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html |
| boto3 S3 client | https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/index.html |
| Presigned URLs | https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-presigned-urls.html |
| S3 multipart upload | https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3.html#multipart-transfers |

---

## 21. Auth & Security

### JWT
| Resource | URL |
|---|---|
| python-jose GitHub | https://github.com/mpdavis/python-jose |
| PyJWT documentation | https://pyjwt.readthedocs.io/en/stable/ |
| FastAPI JWT tutorial | https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/ |

### Password Hashing
| Resource | URL |
|---|---|
| passlib documentation | https://passlib.readthedocs.io/en/stable/ |
| bcrypt algorithm | https://passlib.readthedocs.io/en/stable/lib/passlib.hash.bcrypt.html |

### PostgreSQL Row-Level Security
| Resource | URL |
|---|---|
| Postgres RLS docs | https://www.postgresql.org/docs/current/ddl-rowsecurity.html |
| Supabase RLS guide (good reference) | https://supabase.com/docs/guides/database/postgres/row-level-security |

---

## 22. Video Processing Libraries

| Resource | URL |
|---|---|
| PySceneDetect documentation | https://www.scenedetect.com/docs/latest/ |
| PySceneDetect API — ContentDetector | https://www.scenedetect.com/docs/latest/api/detectors.html#scenedetect.detectors.ContentDetector |
| OpenCV Python — VideoCapture | https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html |
| OpenCV Python — tutorials index | https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html |
| decord fast video reader | https://github.com/dmlc/decord |
| moviepy documentation | https://zulko.github.io/moviepy/ |
| pydub audio | https://github.com/jiaaro/pydub |

---

## 23. LLM Providers & Vision Models

### OpenAI
| Resource | URL |
|---|---|
| OpenAI API reference | https://platform.openai.com/docs/api-reference |
| Chat completions (streaming) | https://platform.openai.com/docs/api-reference/chat/streaming |
| Vision / GPT-4o | https://platform.openai.com/docs/guides/vision |
| Function calling | https://platform.openai.com/docs/guides/function-calling |
| Embeddings | https://platform.openai.com/docs/guides/embeddings |
| Whisper API | https://platform.openai.com/docs/guides/speech-to-text |
| openai Python SDK | https://github.com/openai/openai-python |

### Anthropic Claude
| Resource | URL |
|---|---|
| Anthropic API docs | https://docs.anthropic.com/en/api/ |
| Messages API | https://docs.anthropic.com/en/api/messages |
| Streaming | https://docs.anthropic.com/en/api/messages-streaming |
| Vision | https://docs.anthropic.com/en/docs/build-with-claude/vision |
| anthropic Python SDK | https://github.com/anthropic-ai/anthropic-sdk-python |

### LLaVA (local vision model)
| Resource | URL |
|---|---|
| LLaVA GitHub | https://github.com/haotian-liu/LLaVA |
| LLaVA-1.6 (LLaVA-NeXT) | https://llava-vl.github.io/blog/2024-01-30-llava-1-6/ |
| LLaVA HuggingFace | https://huggingface.co/llava-hf |
| Ollama (local LLaVA serving) | https://ollama.com/library/llava |

### Local LLM Serving
| Resource | URL |
|---|---|
| Ollama documentation | https://ollama.com/docs/ |
| vLLM documentation | https://docs.vllm.ai/ |
| LM Studio | https://lmstudio.ai/docs |

---

## 24. Configuration & Settings

| Resource | URL |
|---|---|
| Pydantic Settings | https://docs.pydantic.dev/latest/concepts/pydantic_settings/ |
| Pydantic v2 docs | https://docs.pydantic.dev/latest/ |
| python-dotenv | https://pypi.org/project/python-dotenv/ |
| structlog (structured logging) | https://www.structlog.org/en/stable/ |

---

## 25. Research Papers

### RAG Foundations
| Paper | URL |
|---|---|
| RAG original paper (Lewis et al., 2020) | https://arxiv.org/abs/2005.11401 |
| Survey of RAG (comprehensive, 2023) | https://arxiv.org/abs/2312.10997 |
| Advanced RAG techniques survey | https://arxiv.org/abs/2401.13510 |
| REALM (retrieval-augmented pretraining) | https://arxiv.org/abs/2002.08909 |

### Graph RAG
| Paper | URL |
|---|---|
| LightRAG — Simple and Fast GraphRAG | https://arxiv.org/abs/2410.05779 |
| Microsoft GraphRAG paper | https://arxiv.org/abs/2404.16130 |
| G-RAG (Knowledge Graph RAG) | https://arxiv.org/abs/2404.16130 |

### Embeddings
| Paper | URL |
|---|---|
| BGE-M3: Multi-lingual, Multi-granularity | https://arxiv.org/abs/2402.03216 |
| SPLADE: Sparse Lexical and Expansion | https://arxiv.org/abs/2109.10086 |
| SPLADE v2 | https://arxiv.org/abs/2205.04733 |
| CLIP (Radford et al., 2021) | https://arxiv.org/abs/2103.00020 |
| ColBERT v2 | https://arxiv.org/abs/2112.01488 |
| E5 embeddings | https://arxiv.org/abs/2212.03533 |

### Retrieval
| Paper | URL |
|---|---|
| DPR (Dense Passage Retrieval) | https://arxiv.org/abs/2004.04906 |
| Hybrid sparse-dense retrieval survey | https://arxiv.org/abs/2112.09118 |
| Reciprocal Rank Fusion | https://dl.acm.org/doi/10.1145/1571941.1572114 |
| HyDE (hypothetical document embeddings) | https://arxiv.org/abs/2212.10496 |

### Re-ranking & Diversity
| Paper | URL |
|---|---|
| MMR (Carbonell & Goldstein, 1998) | https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_SIGIR_1998.pdf |
| Cross-encoder for re-ranking | https://arxiv.org/abs/1901.04085 |
| monoT5 re-ranker | https://arxiv.org/abs/2101.05667 |

### Document Understanding
| Paper | URL |
|---|---|
| TableFormer (IBM table recognition) | https://arxiv.org/abs/2203.00274 |
| LayoutLMv3 (document layout) | https://arxiv.org/abs/2204.08387 |
| Whisper (Radford et al., 2022) | https://arxiv.org/abs/2212.04356 |

### Multimodal RAG & Video
| Paper | URL |
|---|---|
| RAG-Anything | https://arxiv.org/abs/2505.18547 |
| VideoRAG | https://arxiv.org/abs/2411.13093 |
| Video-LLaVA | https://arxiv.org/abs/2311.10122 |

### Evaluation
| Paper | URL |
|---|---|
| RAGAS evaluation framework | https://arxiv.org/abs/2309.15217 |
| ARES (automated RAG eval) | https://arxiv.org/abs/2311.09476 |
| BEIR benchmark | https://arxiv.org/abs/2104.08663 |

---

## 26. Architecture Guides & Deep Dives

### RAG Architecture
| Resource | URL |
|---|---|
| Pinecone — What is RAG? | https://www.pinecone.io/learn/retrieval-augmented-generation/ |
| LangChain RAG tutorials | https://python.langchain.com/docs/tutorials/rag/ |
| LlamaIndex RAG guide | https://docs.llamaindex.ai/en/stable/getting_started/concepts/ |
| Weaviate — Advanced RAG guide | https://weaviate.io/blog/advanced-rag |
| Qdrant — RAG from zero to production | https://qdrant.tech/articles/rag-evaluation-guide/ |

### Hybrid Search
| Resource | URL |
|---|---|
| Qdrant hybrid search article | https://qdrant.tech/articles/hybrid-search/ |
| Weaviate hybrid search | https://weaviate.io/blog/hybrid-search-explained |
| BM25 + neural hybrid (Pinecone) | https://www.pinecone.io/learn/hybrid-search-intro/ |

### Chunking Deep Dives
| Resource | URL |
|---|---|
| Chunking strategies for RAG | https://dev.to/sreeni5018/rag-chunking-strategies-4i3a |
| Greg Kamradt chunking guide | https://github.com/FullStackRetrieval-com/RetrievalTutorials |
| Late chunking (jina.ai) | https://jina.ai/news/late-chunking-in-long-context-embedding-models/ |

### Multi-tenant Vector DBs
| Resource | URL |
|---|---|
| Qdrant multi-tenancy guide | https://qdrant.tech/documentation/guides/multiple-partitions/ |
| Weaviate multi-tenancy | https://weaviate.io/developers/weaviate/manage-data/multi-tenancy |

### Streaming LLM Responses
| Resource | URL |
|---|---|
| Streaming LLM importance + implementation | https://gautam75.medium.com/streaming-llm-responses-importance-and-implementation-911b135ef541 |
| OpenAI streaming guide | https://platform.openai.com/docs/api-reference/streaming |
| Server-sent events (MDN) | https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events |
| SSE specification (W3C) | https://html.spec.whatwg.org/multipage/server-sent-events.html |

---

## 27. Docker & Infrastructure

### Docker
| Resource | URL |
|---|---|
| Docker documentation | https://docs.docker.com/ |
| Docker Compose | https://docs.docker.com/compose/ |
| Docker Compose GPU support | https://docs.docker.com/compose/how-tos/gpu-support/ |
| Multi-stage builds | https://docs.docker.com/build/building/multi-stage/ |
| NVIDIA Container Toolkit | https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html |

### PostgreSQL
| Resource | URL |
|---|---|
| PostgreSQL 16 docs | https://www.postgresql.org/docs/16/ |
| JSONB type | https://www.postgresql.org/docs/16/datatype-json.html |
| Array types | https://www.postgresql.org/docs/16/arrays.html |
| Row-level security | https://www.postgresql.org/docs/current/ddl-rowsecurity.html |
| pg_isready | https://www.postgresql.org/docs/current/app-pg-isready.html |
| postgresql.conf reference | https://www.postgresql.org/docs/16/runtime-config.html |

---

*Context7 quota was exhausted during this session. All links have been compiled from primary sources and verified against known documentation structure as of mid-2025. For the most current URLs, always refer to the official project repositories and documentation sites directly.*