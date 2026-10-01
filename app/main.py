import json
import sqlite3
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status

from app import repo
from app.config import Settings, get_settings
from app.db import connect, init_db
from app.schemas import EmailCreated, EmailIn, EmailOut, EmailStatus, ExtractionOut


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    conn = connect(request.app.state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


def _extraction_out(row: sqlite3.Row) -> ExtractionOut:
    return ExtractionOut(
        fields=json.loads(row["data_json"]),
        confidence=json.loads(row["field_confidence_json"]),
        decision=row["decision"],
        reasons=json.loads(row["decision_reasons_json"]),
        model=row["model"],
        prompt_version=row["prompt_version"],
        created_at=row["created_at"],
        review_action=row["review_action"],
        corrected=json.loads(row["corrected_data_json"]) if row["corrected_data_json"] else None,
        reviewed_by=row["reviewed_by"],
        reviewed_at=row["reviewed_at"],
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        conn = connect(settings.db_path)
        init_db(conn)
        conn.close()
        yield

    app = FastAPI(title="shipment-intake-ai", lifespan=lifespan)
    app.state.settings = settings

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/emails", status_code=status.HTTP_202_ACCEPTED)
    def ingest_email(
        email: EmailIn, response: Response, conn: sqlite3.Connection = Depends(get_conn)
    ) -> EmailCreated:
        """202 for a new email (queued). 200 with the existing record for a duplicate."""
        email_id, created = repo.insert_email(conn, email)
        if not created:
            response.status_code = status.HTTP_200_OK
        row = repo.get_email(conn, email_id)
        return EmailCreated(id=email_id, status=EmailStatus(row["status"]), duplicate=not created)

    @app.get("/emails/{email_id}")
    def get_email(email_id: int, conn: sqlite3.Connection = Depends(get_conn)) -> EmailOut:
        row = repo.get_email(conn, email_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "email not found")
        extraction = repo.get_latest_extraction(conn, email_id)
        return EmailOut(
            id=row["id"],
            message_id=row["message_id"],
            sender=row["sender"],
            subject=row["subject"],
            received_at=row["received_at"],
            status=row["status"],
            attempts=row["attempts"],
            last_error=row["last_error"],
            extraction=_extraction_out(extraction) if extraction else None,
        )

    return app


app = create_app()
