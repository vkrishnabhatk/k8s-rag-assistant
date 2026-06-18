from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag_assistant.api.middleware import AccessLogMiddleware, RequestIDMiddleware
from rag_assistant.api.routers import health, query
from rag_assistant.config import get_settings


def _configure_logging(log_level: str) -> None:
    import logging

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
    )


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    limiter = Limiter(
        key_func=get_remote_address, default_limits=[f"{settings.rate_limit_per_minute}/minute"]
    )

    app = FastAPI(
        title="K8s RAG Assistant",
        description="Production-grade RAG Q&A assistant over Kubernetes documentation",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware (order matters: outermost = first to run on request)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Routers
    app.include_router(health.router)
    app.include_router(query.router)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log = structlog.get_logger(__name__)
        request_id = getattr(request.state, "request_id", "unknown")
        log.error("unhandled_exception", error=str(exc), request_id=request_id)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )

    return app
