# k8s-rag-assistant

A production-grade RAG (Retrieval-Augmented Generation) Q&A assistant over Kubernetes documentation. Ask natural-language questions about Kubernetes and receive context-grounded answers powered by Anthropic Claude, with full streaming support and automatic hallucination guardrails.

## Architecture

```
  User Query
      │
      ▼
  FastAPI /v1/query  (or /v1/query/stream for SSE)
      │
      ├─► Embedder (all-MiniLM-L6-v2, 384-dim, CPU)
      │       │ query vector (L2-normalized)
      │       ▼
      │   FAISS IndexFlatIP ──► top-k Chunks + cosine scores
      │
      ├─► Guardrail 1 (pre-LLM): max_score < 0.3 → "Insufficient context" (no LLM call)
      │
      ├─► PromptBuilder — hybrid mode
      │       ├── DOCUMENTATION MODE: answer from retrieved context + cite sources
      │       ├── GENERAL KNOWLEDGE MODE: Claude answers from training ("Based on general
      │       │   Kubernetes knowledge:") when docs don't cover the topic
      │       └── cache_control on system + context blocks (0.1× token cost on cache hit)
      │
      ├─► Anthropic Claude Haiku 4.5 (prompt caching enabled)
      │       │ answer tokens (JSON-encoded over SSE)
      │       ▼
      ├─► Guardrail 2: "Based on general Kubernetes knowledge:" → passes as general_knowledge
      ├─► Guardrail 3: out-of-scope phrase detection (non-K8s questions only)
      ├─► Guardrail 4: word coverage check (answer words found in context / answer words)
      │
      └─► QueryResponse { answer, sources, validation, latency_ms }
```

**Ingestion pipeline** (runs once, or on `--force`):

```
~50 K8s doc URLs → httpx fetch → BeautifulSoup clean → word-based chunker (500/50 overlap)
    → SentenceTransformer embed → FAISS IndexFlatIP → persist data/faiss.index + data/chunks.pkl
```

## Prerequisites

| Tool | Min version | Install |
|---|---|---|
| Python | 3.12 | [python.org/downloads](https://www.python.org/downloads/) |
| Docker Desktop | 4.x | [docs.docker.com/get-docker](https://docs.docker.com/get-docker/) — includes Compose v2 |
| Docker Compose | v2 (plugin) | Bundled with Docker Desktop; standalone: [docs.docker.com/compose/install](https://docs.docker.com/compose/install/) |
| Helm | 3.x | [helm.sh/docs/intro/install](https://helm.sh/docs/intro/install/) |
| kubectl | any | [kubernetes.io/docs/tasks/tools](https://kubernetes.io/docs/tasks/tools/) — only needed for K8s deploy |

You also need an [Anthropic API key](https://console.anthropic.com/).

> **Note**: Docker Compose v2 is invoked as `docker compose` (no hyphen). The `Makefile` and instructions below use the `docker-compose` alias which is equivalent when the Compose CLI plugin is installed.

## Local Development

### 1. Clone and set up the Python environment

```bash
git clone https://github.com/vkrishnabhatk/k8s-rag-assistant.git
cd k8s-rag-assistant

python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -e ".[dev]"
```

### 2. Configure secrets

```bash
cp .env.example .env
# Open .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Ingest Kubernetes documentation

This fetches ~50 K8s doc pages covering workloads, networking, storage, scheduling, security, and cluster administration, chunks them into 500-word windows, embeds them with `all-MiniLM-L6-v2`, and writes `data/faiss.index` + `data/chunks.pkl`. It is idempotent — re-running without `--force` skips ingestion if the index already exists.

```bash
make ingest

# Force a full re-index (e.g. after adding more URLs to K8S_DOCS_URLS):
python scripts/ingest.py --force
```

Expected output:

```
INFO  index_built  chunks=900+  index_path=data/faiss.index  ...
```

### 4. Start the API server

```bash
make serve
# Equivalent: uvicorn rag_assistant.api.app:create_app --reload --factory --port 8000
```

### 5. (Optional) Start the web chat UI

A Streamlit chat interface is included for interactive testing. It requires the API server to be running.

```bash
# Install UI dependencies (one time)
make install-ui

# Launch at http://localhost:8501
make ui
```

Features: streaming responses, conversation history, source cards with confidence scores, latency display, and a sidebar with example questions and API health check.

### 6. Smoke test the running server

```bash
# Liveness
curl http://localhost:8000/health
# → {"status":"ok"}

# Readiness (503 before ingestion, 200 after)
curl http://localhost:8000/ready
# → {"status":"ready"}

# Standard JSON query
curl -s -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is a Kubernetes Deployment?"}' | jq .

# Streaming query (SSE)
curl -N -X POST http://localhost:8000/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I scale a Deployment?"}'

# Out-of-domain query — guardrail fires, no LLM call
curl -s -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is quantum entanglement?"}' | jq .answer
# → "Insufficient context: the Kubernetes documentation does not contain..."
```

### 7. Run tests and linting

```bash
make test          # pytest, ≥80% coverage required
make lint          # ruff check + format check
make type-check    # mypy
```

## API Reference

### `POST /v1/query`

Returns a complete JSON response.

**Request**

```json
{ "query": "How does a Kubernetes Service route traffic?" }
```

**Response**

```json
{
  "answer": "A Kubernetes Service routes traffic by...",
  "sources": [
    {
      "url": "https://kubernetes.io/docs/concepts/services-networking/service/",
      "title": "Service",
      "score": 0.87
    }
  ],
  "validation": {
    "passed": true,
    "guardrail_triggered": null,           // or "general_knowledge" | "out_of_context" | "low_overlap" | "low_confidence"
    "confidence_score": 0.87,
    "overlap_score": 0.31
  },
  "latency_ms": 1240.5
}
```

### `POST /v1/query/stream`

Streams answer tokens as Server-Sent Events, then emits a final `done` event with sources and validation.

```bash
curl -N -X POST http://localhost:8000/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I scale a Deployment?"}'
```

```
data: "A Kubernetes Deployment can be scaled"
data: " by updating the replicas field..."
...
event: done
data: {"sources": [...], "validation": {...}}
```

Each token is JSON-encoded in the `data` field to preserve newlines and special characters faithfully across the SSE transport layer. The final `done` event carries the full metadata object (sources, validation, latency).

### `GET /health`

Liveness check. Returns `{"status": "ok"}` (200).

### `GET /ready`

Readiness check. Returns `{"status": "ready"}` (200) when the FAISS index is loaded, `{"status": "not ready"}` (503) otherwise.

**Rate limit**: 10 requests/minute per IP (configurable via `RATE_LIMIT_PER_MINUTE`).

## Configuration

All settings are read from environment variables (or `.env`):

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Claude model to use |
| `ANTHROPIC_MAX_TOKENS` | `1024` | Max tokens in completion |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `FAISS_INDEX_PATH` | `data/faiss.index` | Path to persisted FAISS index |
| `FAISS_METADATA_PATH` | `data/chunks.pkl` | Path to persisted chunk metadata |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve per query |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.3` | Min cosine similarity before refusing to answer |
| `CHUNK_SIZE_WORDS` | `500` | Words per chunk |
| `CHUNK_OVERLAP_WORDS` | `50` | Overlap between adjacent chunks |
| `GUARDRAIL_OVERLAP_THRESHOLD` | `0.15` | Min word coverage (answer words found in context ÷ answer words) |
| `RATE_LIMIT_PER_MINUTE` | `10` | Requests per minute per IP |
| `LOG_LEVEL` | `INFO` | Structured log level |

## Docker Compose

```bash
# Build the image locally
make docker-build

# Step 1 — run ingestion (writes FAISS index to the rag_data named volume)
docker-compose --profile ingest up --exit-code-from ingest

# Step 2 — start the API (reads from the same volume)
docker-compose up api

# Verify
curl http://localhost:8000/health    # → {"status":"ok"}
curl http://localhost:8000/ready     # → {"status":"ready"}

# Force a full re-index
docker-compose run --rm \
  -e FORCE_REINDEX=true \
  --profile ingest ingest python scripts/ingest.py --force
```

The `rag_data` named volume is shared between the `ingest` and `api` services so the FAISS index persists across container restarts.

## Kubernetes (Helm) Deployment

> **Local testing tip**: Use [kind](https://kind.sigs.k8s.io/) to test the Helm chart locally without a cloud cluster. See the step-by-step guide below.

### Local testing with kind

The easiest way to validate the Helm chart locally — no cloud account required:

```bash
# 1. Install kind
brew install kind

# 2. Create a cluster and load the local image
kind create cluster --name rag-assistant
make docker-build
kind load docker-image rag-assistant:local --name rag-assistant

# 3. Deploy (imagePullPolicy: Never uses the pre-loaded image)
kubectl create secret generic rag-assistant-secrets \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-...
kubectl annotate secret rag-assistant-secrets \
  meta.helm.sh/release-name=rag-assistant \
  meta.helm.sh/release-namespace=default
kubectl label secret rag-assistant-secrets \
  app.kubernetes.io/managed-by=Helm

helm upgrade --install rag-assistant deploy/helm/rag-assistant/ \
  --set image.repository=rag-assistant \
  --set image.tag=local \
  --set image.pullPolicy=Never \
  --set replicaCount=1 \
  --set autoscaling.enabled=false \
  --set ingress.enabled=false \
  --set anthropic.apiKeySecretName=rag-assistant-secrets

# 4. Wait for ingestion, then restart the API pod to load the index
kubectl logs -f job/rag-assistant-rag-assistant-ingest
kubectl rollout restart deployment/rag-assistant-rag-assistant
kubectl rollout status deployment/rag-assistant-rag-assistant

# 5. Test
kubectl port-forward svc/rag-assistant-rag-assistant 8080:8000
curl http://localhost:8080/ready   # → {"status":"ready"}
```

### Cloud / CI cluster pre-requisites

- A running Kubernetes cluster (`kubectl cluster-info` should succeed)
- The Docker image pushed to a registry accessible from the cluster (CI/CD does this automatically via `cd.yml` on merge to `main`)
- `helm` 3.x installed

### Step 1 — create the API key secret

```bash
kubectl create secret generic my-anthropic-secret \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-...
```

### Step 2 — validate the chart

```bash
make helm-lint
# or: helm lint deploy/helm/rag-assistant/

# Optional: dry-run render to inspect all manifests
helm template rag-assistant deploy/helm/rag-assistant/ \
  --set anthropic.apiKeySecretName=my-anthropic-secret | less
```

### Step 3 — install or upgrade

```bash
helm upgrade --install rag-assistant deploy/helm/rag-assistant/ \
  --set image.tag=<git-sha>                        \
  --set anthropic.apiKeySecretName=my-anthropic-secret \
  --set ingress.host=rag.example.com               \
  --wait
```

`--wait` blocks until all pods are ready. On a fresh install this includes waiting for the **ingestion Job** to run and write the FAISS index to the PVC before the Deployment passes its readiness probe.

### Step 4 — verify the deployment

```bash
# Check all resources
kubectl get pod,svc,ingress,pvc,hpa -l app.kubernetes.io/name=rag-assistant

# Watch the ingestion Job logs
kubectl logs -f job/rag-assistant-ingest

# Check API pod readiness
kubectl rollout status deployment/rag-assistant

# Quick health check via port-forward (no ingress needed)
kubectl port-forward svc/rag-assistant 8080:8000 &
curl http://localhost:8080/health    # → {"status":"ok"}
curl http://localhost:8080/ready     # → {"status":"ready"}

# Test a query
curl -s -X POST http://localhost:8080/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is a Kubernetes Deployment?"}' | jq .
```

### Key `values.yaml` fields

| Field | Default | Notes |
|---|---|---|
| `image.repository` | `ghcr.io/vkrishnabhatk/k8s-rag-assistant` | Override for private registry |
| `replicaCount` | `2` | |
| `autoscaling.enabled` | `true` | HPA on CPU (70%, 2–10 replicas) |
| `persistence.size` | `5Gi` | FAISS index storage |
| `persistence.storageClass` | `""` (cluster default) | Use `ReadWriteMany` class for multi-replica |
| `ingestionJob.enabled` | `true` | Runs `scripts/ingest.py` as a K8s Job on `helm upgrade` |
| `ingress.host` | `rag-assistant.local` | Set to your domain |
| `ingress.tls` | `[]` | Add cert-manager annotation for TLS |

### TLS with cert-manager

```bash
helm upgrade --install rag-assistant deploy/helm/rag-assistant/ \
  --set ingress.host=rag.example.com \
  --set "ingress.tls[0].hosts[0]=rag.example.com" \
  --set "ingress.tls[0].secretName=rag-tls" \
  --set "ingress.annotations.cert-manager\\.io/cluster-issuer=letsencrypt-prod"
```

### Uninstall

```bash
helm uninstall rag-assistant
kubectl delete secret my-anthropic-secret
# PVC is retained by default — delete manually if you want to wipe the index:
kubectl delete pvc rag-assistant-data
```

## Evaluation (RAGAS)

```bash
pip install -e ".[eval]"
make evaluate
# Scores written to data/eval_report.json
```

Metrics reported: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.

The golden dataset (`tests/data/golden_dataset.json`) covers 20 Kubernetes concepts: Deployment, Service, ConfigMap, Secret, Ingress, PersistentVolume, StatefulSet, DaemonSet, Job, CronJob, HPA, NetworkPolicy, Taint/Toleration, Pod affinity, liveness/readiness probes, resource limits, RBAC, Namespace, ServiceAccount, pod lifecycle.

## Development

```bash
make lint        # ruff check + format check
make type-check  # mypy
make test        # pytest with >=80% coverage requirement
```

Pre-commit hooks (ruff autofix, mypy, detect-secrets):

```bash
pip install pre-commit
pre-commit install
```

## Releases

This project uses **tag-driven SemVer**. The version lives in three files that are always kept in sync: `pyproject.toml`, `src/rag_assistant/__init__.py`, and `deploy/helm/rag-assistant/Chart.yaml`. The `scripts/release.py` helper updates all three atomically.

### How to cut a release

```bash
# Bump versions, commit, and create an annotated git tag
make release VERSION=1.0.0

# Push the commit and the tag to trigger the CD pipeline
git push && git push origin v1.0.0
```

The `release` target validates that:
- The version is valid SemVer (`MAJOR.MINOR.PATCH`)
- You are on the `main` branch
- The working tree is clean
- The tag does not already exist

### What the CD pipeline does

| Trigger | Image tags produced |
|---|---|
| Push to `main` | `sha-<short>` (dev/edge build, no `latest`) |
| Push `v1.2.3` tag | `1.2.3`, `1.2`, `1`, `latest` |

### Image tag reference

| Tag | Meaning |
|---|---|
| `latest` | Most recent release |
| `1` | Latest `1.x.x` release |
| `1.2` | Latest `1.2.x` patch |
| `1.2.3` | Exact release — use this in production Helm installs |
| `sha-<short>` | Dev build from a specific commit |

Always pin production Helm installs to an exact version tag:

```bash
helm upgrade --install rag-assistant deploy/helm/rag-assistant/ \
  --set image.tag=1.2.3 \
  --set anthropic.apiKeySecretName=my-anthropic-secret
```

## Limitations

- **Static corpus**: The FAISS index is built from ~50 hardcoded Kubernetes doc URLs. Re-run `make ingest-force` after adding new URLs to `_DEFAULT_K8S_URLS` in `config.py`.
- **Hybrid knowledge**: Topics not in the ingested docs (e.g. node pools, CNI plugins, cloud-provider specifics) are answered from Claude's training knowledge, clearly marked as "Based on general Kubernetes knowledge:". Accuracy for these answers is not doc-verified.
- **CPU-only embeddings**: `all-MiniLM-L6-v2` runs on CPU (~10 ms/query). GPU acceleration is not configured.
- **Single-node PVC**: The Helm chart defaults to `ReadWriteOnce`. Multi-replica deployments require a `ReadWriteMany` storage class (e.g., NFS, EFS).
- **No conversation history**: Each `/v1/query` call is stateless. Follow-up questions do not have access to prior answers. The Streamlit UI maintains in-memory history within a single browser session only.
- **Rate limit is in-process**: `slowapi` stores state in memory; it resets on pod restart and does not coordinate across replicas.
