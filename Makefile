.PHONY: ingest serve ui test evaluate lint type-check format \
        docker-build docker-up docker-ingest helm-lint install install-eval install-ui release

PYTHON  := python
UVICORN := uvicorn
IMAGE   := rag-assistant:local

# ── Local development ──────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

install-eval:
	pip install -e ".[dev,eval]"

install-ui:
	pip install -e ".[ui]"

ingest:
	$(PYTHON) scripts/ingest.py

ingest-force:
	$(PYTHON) scripts/ingest.py --force

serve:
	$(UVICORN) rag_assistant.api.app:create_app \
		--factory \
		--host 0.0.0.0 \
		--port 8000 \
		--reload

ui:
	streamlit run streamlit_app.py

# ── Testing ────────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/unit/ tests/integration/ \
		--cov=rag_assistant \
		--cov-report=term-missing \
		--cov-fail-under=80 \
		-v

test-unit:
	$(PYTHON) -m pytest tests/unit/ -v

test-integration:
	$(PYTHON) -m pytest tests/integration/ -v

# ── Evaluation ─────────────────────────────────────────────────────────────────

evaluate:
	$(PYTHON) scripts/evaluate.py

# ── Code quality ───────────────────────────────────────────────────────────────

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

type-check:
	mypy src/

# ── Docker ─────────────────────────────────────────────────────────────────────

docker-build:
	docker build -t $(IMAGE) .

docker-ingest:
	docker compose --profile ingest up --build ingest

docker-up:
	docker compose up --build api

docker-down:
	docker compose down

# ── Helm ───────────────────────────────────────────────────────────────────────

helm-lint:
	helm lint deploy/helm/rag-assistant/

helm-template:
	helm template rag-assistant deploy/helm/rag-assistant/

# ── Release ────────────────────────────────────────────────────────────────────

release:
	@test -n "$(VERSION)" || (echo "usage: make release VERSION=1.0.0" && exit 1)
	$(PYTHON) scripts/release.py $(VERSION)
