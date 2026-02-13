# NeuralSearch Prime: Hybrid E-Commerce Search & Reranking System

**Version:** 1.0 | **Status:** Production-Ready Blueprint

**Role:** Senior AI Architect / Lead ML Engineer

## 📖 Overview

**NeuralSearch Prime** is an advanced two-stage retrieval and reranking system designed to maximize e-commerce search relevance. It bridges the gap between high-throughput lexical search and deep semantic reasoning by implementing a hybrid retrieval layer followed by a rationale-guided reranking layer.

The system is optimized for the **Amazon ESCI** dataset (Exact, Substitute, Complement, Irrelevant) and aims to achieve human-level relevance with millisecond latency.

---

## 🚀 Key Features

### 1. Hybrid Retrieval (Stage 1)
**Lexical Search:** BM25 implementation on product titles and descriptions.
**Semantic Search:** k-NN vector search using 384-dimension embeddings.
**Reciprocal Rank Fusion (RRF):** Merges results from lexical and semantic layers without score normalization, ensuring stability across varying score distributions.

### 2. Rationale-Guided Reranking (Stage 2)


**Distillation:** Uses a "Student" Cross-Encoder distilled from a "Teacher" LLM (e.g., Llama-3).



**Dual-Objective Training:** The model is trained to match the teacher's relevance score (MSE Loss) and internalize natural language rationales.


 
**Efficiency:** Processes only the top  candidates to maintain low latency.



### 3. Production-Ready Infrastructure


* **Platform:** Dockerized microservices orchestrated via Kubernetes (EKS).



* **Search Engine:** OpenSearch 2.11+ supporting Neural Search and RRF.



* **Inference:** Hugging Face Text Embeddings Inference (TEI) or AWS SageMaker.



---

## 🛠 System Architecture

The system follows a standard multi-stage retrieval architecture:

1. **Ingestion:** Validates product metadata and enforces ESCI relevance mapping ().


2. **Retrieval:** Parallel execution of BM25 and Dense Vector queries, merged via RRF.


3. **Reranking:** A distilled Cross-Encoder (e.g., MiniLM-6L) re-scores the top 50 results.


4. **Fallback:** If the reranker fails, the system transparently returns Stage 1 results.



---

## ⚡ Performance & SLAs

| Metric | Target | Description |
| --- | --- | --- |
| **End-to-End Latency** | **< 300ms** | Total time for Retrieval + Rerank |
| **Retrieval Latency** | <br>**< 200ms** | p95 latency for Stage 1 only |
| **Availability** | **99.9%** | Monthly error budget of ~43.8 minutes |
| **Throughput** | **10 RPS** | Supported load for the full pipeline |

---

## 🔌 API Reference

### Search Endpoint

**POST** `/search`

Accepts a natural language query and returns ranked product results.

**Request Body:**

```json
{
  "query": "wireless noise canceling headphones",
  "top_k": 50
}

```

**Response:**

```json
[
  {
    "product_id": "B08F29N12",
    "score": 0.98,
    "rank_change": "+5",
    "title": "Sony WH-1000XM4 Wireless...",
    "rationale": "Matches 'wireless' and 'noise canceling' explicitly."
  },
  {
    "product_id": "B07G4MN15",
    "score": 0.85,
    "rank_change": "-2",
    "title": "Bose QuietComfort 35 II..."
  }
]

```

---

## ⚙️ Installation & Setup

### Prerequisites

**Docker** & **Kubernetes** (EKS preferred).
* **OpenSearch 2.11+** instance running.
* **Python 3.9+** for local development.

### Environment Variables
Create a `.env` file in the root directory:

```bash
OPENSEARCH_HOST=https://search-domain.us-east-1.es.amazonaws.com
OPENSEARCH_USER=admin
OPENSEARCH_PASS=YourStrongPassword123!
RERANKER_ENDPOINT=http://inference-service:8080/rerank
[cite_start]TOP_K_RERANK=50  # [cite: 69]
[cite_start]RRF_K=60         # [cite: 56]

```

### Deployment (Kubernetes)

1. **Build Images:**
```bash
docker build -t neuralsearch-prime:v1 .

```


2. **Deploy manifests:**
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

```


3. **Verify HPA:**
Ensure Horizontal Pod Autoscaling is enabled for CPU/Memory scaling.



---

## 📊 Operations & Monitoring

* **Dashboards:** Streamlit interface available for side-by-side comparison of "Lexical," "Hybrid," and "Neural" results.


* **Key Metrics (SLIs):**
* *OpenSearch:* CPU utilization, JVM heap pressure, thread pool rejections.


* *Inference:* SageMaker ModelLatency, InvocationsPerInstance.




* **Incident Response:**
* *Latency Spikes:* Check for long-tail queries or thread pool rejections.
* *Bad Deployment:* Immediate rollback to previous Cross-Encoder version.





---

## 📚 References 
**Primary Research:** *Rationale-Guided Distillation for E-Commerce...* (ACL Anthology, 2025).
