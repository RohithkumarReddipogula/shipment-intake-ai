import sqlite3

import pytest

from app.db import init_db, utcnow
from app.schemas import ExtractionOutput, ShipmentFields, WeightUnit


def _insert_email(conn, dedup_key):
    now = utcnow()
    conn.execute(
        "INSERT INTO emails (dedup_key, sender, subject, body, received_at, next_attempt_at, created_at, updated_at)"
        " VALUES (?, 'a@b.de', 's', 'b', ?, ?, ?, ?)",
        (dedup_key, now, now, now, now),
    )


def test_init_db_is_idempotent(conn):
    init_db(conn)


def test_dedup_key_is_unique(conn):
    _insert_email(conn, "k1")
    with pytest.raises(sqlite3.IntegrityError):
        _insert_email(conn, "k1")


@pytest.mark.parametrize(
    "value,unit,expected",
    [(1000, WeightUnit.KG, 1000.0), (2200, WeightUnit.LBS, 997.9), (1.5, WeightUnit.T, 1500.0)],
)
def test_weight_kg_conversion(value, unit, expected):
    assert ShipmentFields(weight_value=value, weight_unit=unit).weight_kg == expected


def test_weight_kg_none_when_missing():
    assert ShipmentFields(weight_value=500).weight_kg is None


def test_extraction_output_parses_llm_json():
    out = ExtractionOutput.model_validate_json(
        '{"fields": {"origin": "Hamburg", "pickup_date": "2026-10-06", "pallet_count": 4},'
        ' "confidence": {"origin": 0.95}}'
    )
    assert out.fields.origin == "Hamburg"
    assert out.fields.pickup_date.isoformat() == "2026-10-06"
    assert out.confidence.origin == 0.95
    assert out.confidence.weight == 0.0
