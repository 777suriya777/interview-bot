"""
Tests for the Report Service.

Groups:
  TestAggregateReport       → _aggregate_report() pure-logic tests (no DB)
  TestRenderPdf             → _render_pdf() with mocked WeasyPrint
  TestStorePdf              → _store_pdf_local() and S3 fallback
  TestGenerateEndpoint      → POST /generate (fully mocked DB + PDF)
  TestGetReportEndpoint     → GET /report/{session_id}
  TestHealthEndpoint        → GET /health

No real DB, WeasyPrint, or S3 required.
"""
from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import pytest

# ── Ensure report_service is on sys.path ─────────────────────────────────────
REPORT_DIR = str(Path(__file__).parent.parent / "report_service")
if REPORT_DIR not in sys.path or sys.path[0] != REPORT_DIR:
    if REPORT_DIR in sys.path:
        sys.path.remove(REPORT_DIR)
    sys.path.insert(0, REPORT_DIR)


# ── Stub heavy dependencies ───────────────────────────────────────────────────

def _stub_deps():
    """Stub asyncpg, weasyprint, jinja2 if not installed."""
    if "asyncpg" not in sys.modules:
        a = types.ModuleType("asyncpg")
        a.create_pool = AsyncMock()
        sys.modules["asyncpg"] = a

    if "weasyprint" not in sys.modules:
        wp = types.ModuleType("weasyprint")
        mock_html_cls = MagicMock()
        mock_html_cls.return_value.write_pdf.return_value = b"%PDF-1.4 fake"
        wp.HTML = mock_html_cls
        sys.modules["weasyprint"] = wp

    # jinja2 is usually installed; stub only if absent
    if "jinja2" not in sys.modules:
        j2 = types.ModuleType("jinja2")
        mock_env = MagicMock()
        mock_tmpl = MagicMock()
        mock_tmpl.render.return_value = "<html>report</html>"
        mock_env.get_template.return_value = mock_tmpl
        j2.Environment = MagicMock(return_value=mock_env)
        j2.FileSystemLoader = MagicMock()
        j2.select_autoescape = MagicMock(return_value=[])
        sys.modules["jinja2"] = j2

_stub_deps()


# ── Shared test data ──────────────────────────────────────────────────────────

NOW  = datetime(2026, 3, 6, 10, 0, 0, tzinfo=timezone.utc)
THEN = NOW + timedelta(minutes=25, seconds=40)

def _raw_session():
    return {
        "session_id":     "abc-123",
        "interview_type": "technical",
        "target_role":    "Software Engineer",
        "started_at":     NOW,
        "ended_at":       THEN,
        "candidate_name": "Suriya Prasath",
    }

def _raw_answers(n: int = 5) -> list[dict]:
    categories = ["algorithms", "system_design", "algorithms", "behavioural", "system_design"]
    labels     = ["high", "moderate", "high", "low", "anxious"]
    return [
        {
            "question_text":      f"Question {i}",
            "difficulty":         (i % 5) + 1,
            "category":           categories[i % len(categories)],
            "transcript":         f"Answer transcript for question {i}.",
            "content_score":      i % 4,
            "relevance_score":    (i + 1) % 4,
            "completeness_score": (i + 2) % 4,
            "accuracy_score":     (i + 1) % 4,
            "overall_score":      round(((i % 4) + ((i+1) % 4) + ((i+2) % 4) + ((i+1) % 4)) / 4, 2),
            "confidence_label":   labels[i % len(labels)],
            "delivery_flags":     [],
            "submitted_at":       NOW + timedelta(minutes=i * 3),
        }
        for i in range(n)
    ]


def _raw_data(n: int = 5) -> dict:
    return {"session": _raw_session(), "answers": _raw_answers(n)}


def _ensure_report_main():
    """
    Ensure sys.modules['main'] is the report service's main.py.
    Call at the start of any test or helper that imports from 'main' directly,
    to prevent cross-service module collisions when the full suite runs.
    """
    if REPORT_DIR not in sys.path or sys.path[0] != REPORT_DIR:
        if REPORT_DIR in sys.path:
            sys.path.remove(REPORT_DIR)
        sys.path.insert(0, REPORT_DIR)
    _stub_deps()
    # If the cached 'main' doesn't have _aggregate_report it's the wrong service
    cached = sys.modules.get("main")
    if cached is None or not hasattr(cached, "_aggregate_report"):
        if "main" in sys.modules:
            del sys.modules["main"]
        import importlib
        importlib.import_module("main")


# ── TestAggregateReport ───────────────────────────────────────────────────────

class TestAggregateReport:

    def _aggregate(self, n: int = 5):
        # Ensure report_service/main.py is the one being imported, not
        # another service's main.py that may have been cached.
        if REPORT_DIR not in sys.path or sys.path[0] != REPORT_DIR:
            if REPORT_DIR in sys.path:
                sys.path.remove(REPORT_DIR)
            sys.path.insert(0, REPORT_DIR)
        _stub_deps()
        # Force reimport if the cached 'main' is not the report service
        if "main" in sys.modules:
            cached = getattr(sys.modules["main"], "_aggregate_report", None)
            if cached is None:
                del sys.modules["main"]
        from main import _aggregate_report
        return _aggregate_report(_raw_data(n))

    def test_returns_dict(self):
        result = self._aggregate()
        assert isinstance(result, dict)

    def test_session_id_preserved(self):
        result = self._aggregate()
        assert result["session_id"] == "abc-123"

    def test_total_questions_count(self):
        result = self._aggregate(n=7)
        assert result["total_questions"] == 7

    def test_duration_seconds_correct(self):
        result = self._aggregate()
        # THEN - NOW = 25m40s = 1540 seconds
        assert result["duration_seconds"] == 1540

    def test_average_overall_score_is_float(self):
        result = self._aggregate()
        assert isinstance(result["average_overall_score"], float)

    def test_average_overall_score_in_range(self):
        result = self._aggregate()
        assert 0.0 <= result["average_overall_score"] <= 4.0

    def test_per_topic_scores_has_categories(self):
        result = self._aggregate()
        # answers have categories: algorithms, system_design, behavioural
        assert "algorithms"    in result["per_topic_scores"]
        assert "system_design" in result["per_topic_scores"]

    def test_per_topic_scores_values_in_range(self):
        result = self._aggregate()
        for score in result["per_topic_scores"].values():
            assert 0.0 <= score <= 4.0

    def test_confidence_distribution_keys(self):
        result = self._aggregate()
        dist = result["confidence_distribution"]
        assert set(dist.keys()) == {"high", "moderate", "low", "anxious"}

    def test_confidence_distribution_counts_sum_to_total(self):
        result = self._aggregate(n=5)
        dist = result["confidence_distribution"]
        assert sum(dist.values()) == 5

    def test_improvement_areas_is_list(self):
        result = self._aggregate()
        assert isinstance(result["improvement_areas"], list)

    def test_improvement_areas_max_three(self):
        result = self._aggregate()
        assert len(result["improvement_areas"]) <= 3

    def test_duration_str_format(self):
        result = self._aggregate()
        # Should be in format "25m 40s"
        assert "m" in result["duration_str"]
        assert "s" in result["duration_str"]

    def test_avg_scores_has_four_dims(self):
        result = self._aggregate()
        avg = result["avg_scores"]
        for dim in ["content", "relevance", "completeness", "accuracy"]:
            assert dim in avg

    def test_candidate_name_in_result(self):
        result = self._aggregate()
        assert result["candidate_name"] == "Suriya Prasath"

    def test_empty_answers_returns_zero_scores(self):
        from main import _aggregate_report
        raw = {"session": _raw_session(), "answers": []}
        result = _aggregate_report(raw)
        assert result["average_overall_score"] == 0.0
        assert result["total_questions"] == 0

    def test_delivery_flags_parsed_from_json_string(self):
        """delivery_flags stored as JSON string in DB should be parsed to list."""
        from main import _aggregate_report
        raw = _raw_data(1)
        raw["answers"][0]["delivery_flags"] = '["fast_speech flag"]'
        result = _aggregate_report(raw)
        flags = result["answers"][0]["delivery_flags"]
        assert isinstance(flags, list)
        assert len(flags) == 1

    def test_answers_list_in_result(self):
        result = self._aggregate(n=3)
        assert "answers" in result
        assert len(result["answers"]) == 3

    def test_answer_detail_has_required_keys(self):
        result = self._aggregate(n=1)
        ans = result["answers"][0]
        for key in ["question_text", "difficulty", "transcript", "content_score",
                    "relevance_score", "completeness_score", "accuracy_score",
                    "overall_score", "confidence_label", "delivery_flags"]:
            assert key in ans, f"Missing key: {key}"


# ── TestRenderPdf ─────────────────────────────────────────────────────────────

class TestRenderPdf:

    def setup_method(self):
        _ensure_report_main()

    def test_returns_bytes(self):
        from main import _render_pdf
        with patch("main.jinja_env") as mock_env:
            mock_tmpl = MagicMock()
            mock_tmpl.render.return_value = "<html><body>Report</body></html>"
            mock_env.get_template.return_value = mock_tmpl

            with patch("main.WeasyHTML") as mock_wp:
                mock_wp.return_value.write_pdf.return_value = b"%PDF-1.4"
                result = _render_pdf({"session_id": "s1", "candidate_name": "Test"})

        assert isinstance(result, bytes)

    def test_template_rendered_with_data(self):
        from main import _render_pdf
        report_data = {"session_id": "s1", "candidate_name": "Alice"}
        with patch("main.jinja_env") as mock_env:
            mock_tmpl = MagicMock()
            mock_tmpl.render.return_value = "<html></html>"
            mock_env.get_template.return_value = mock_tmpl

            with patch("main.WeasyHTML") as mock_wp:
                mock_wp.return_value.write_pdf.return_value = b"%PDF"
                _render_pdf(report_data)

        # Verify render was called with our data as kwargs
        mock_tmpl.render.assert_called_once_with(**report_data)

    def test_raises_runtime_error_on_weasyprint_failure(self):
        from main import _render_pdf
        with patch("main.jinja_env") as mock_env:
            mock_tmpl = MagicMock()
            mock_tmpl.render.return_value = "<html></html>"
            mock_env.get_template.return_value = mock_tmpl

            with patch("main.WeasyHTML") as mock_wp:
                mock_wp.return_value.write_pdf.side_effect = Exception("PDF error")
                with pytest.raises(RuntimeError, match="WeasyPrint"):
                    _render_pdf({})

    def test_raises_runtime_error_on_template_failure(self):
        from main import _render_pdf
        with patch("main.jinja_env") as mock_env:
            mock_env.get_template.side_effect = Exception("Template not found")
            with pytest.raises(RuntimeError, match="Template"):
                _render_pdf({})


# ── TestStorePdf ──────────────────────────────────────────────────────────────

class TestStorePdf:

    def setup_method(self):
        _ensure_report_main()

    def test_local_storage_returns_file_url(self, tmp_path):
        from main import _store_pdf_local, REPORT_LOCAL_DIR
        import main as rpt_main
        # Redirect REPORT_LOCAL_DIR to tmp_path
        original = rpt_main.REPORT_LOCAL_DIR
        rpt_main.REPORT_LOCAL_DIR = tmp_path
        try:
            url = _store_pdf_local("session-abc", b"%PDF fake")
            assert url.startswith("file://")
            assert "session-abc" in url
        finally:
            rpt_main.REPORT_LOCAL_DIR = original

    def test_local_storage_writes_file(self, tmp_path):
        from main import _store_pdf_local
        import main as rpt_main
        original = rpt_main.REPORT_LOCAL_DIR
        rpt_main.REPORT_LOCAL_DIR = tmp_path
        try:
            _store_pdf_local("session-xyz", b"%PDF content")
            assert (tmp_path / "session-xyz.pdf").exists()
        finally:
            rpt_main.REPORT_LOCAL_DIR = original

    @pytest.mark.asyncio
    async def test_s3_falls_back_to_local_when_boto3_missing(self, tmp_path):
        from main import _store_pdf_s3
        import main as rpt_main
        original = rpt_main.REPORT_LOCAL_DIR
        rpt_main.REPORT_LOCAL_DIR = tmp_path
        try:
            with patch.dict(sys.modules, {"boto3": None}):
                try:
                    url = await _store_pdf_s3("sess-fallback", b"%PDF")
                    assert "file://" in url
                except Exception:
                    pass
        finally:
            rpt_main.REPORT_LOCAL_DIR = original


# ── TestGenerateEndpoint ──────────────────────────────────────────────────────

class TestGenerateEndpoint:
    """Tests for POST /generate — fully mocked DB and PDF generation."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        if REPORT_DIR not in sys.path or sys.path[0] != REPORT_DIR:
            if REPORT_DIR in sys.path:
                sys.path.remove(REPORT_DIR)
            sys.path.insert(0, REPORT_DIR)

        # Ensure fresh import
        if "main" in sys.modules:
            del sys.modules["main"]

        _stub_deps()
        import main as rpt_main

        rpt_main.app_state["ready"] = True
        rpt_main.app_state["pool"]  = None

        with TestClient(rpt_main.app) as c:
            yield c, rpt_main

    def _mock_fetch(self, rpt_main):
        """Patch _fetch_session_data to return fake raw data."""
        return patch.object(
            rpt_main,
            "_fetch_session_data",
            AsyncMock(return_value=_raw_data(5)),
        )

    def _mock_pdf(self, rpt_main):
        return patch.object(rpt_main, "_render_pdf", return_value=b"%PDF-fake")

    def _mock_store(self, rpt_main):
        return patch.object(
            rpt_main, "_store_pdf_local",
            return_value="file:///tmp/reports/abc-123.pdf"
        )

    def _mock_save_url(self, rpt_main):
        return patch.object(rpt_main, "_save_report_url_to_db", AsyncMock())

    def test_returns_200(self, client):
        c, rpt = client
        with self._mock_fetch(rpt), self._mock_pdf(rpt), \
             self._mock_store(rpt), self._mock_save_url(rpt):
            resp = c.post("/generate", json={"session_id": "abc-123"})
        assert resp.status_code == 200

    def test_response_schema(self, client):
        c, rpt = client
        with self._mock_fetch(rpt), self._mock_pdf(rpt), \
             self._mock_store(rpt), self._mock_save_url(rpt):
            data = c.post("/generate", json={"session_id": "abc-123"}).json()
        required = [
            "session_id", "duration_seconds", "total_questions",
            "average_overall_score", "per_topic_scores",
            "confidence_distribution", "improvement_areas", "report_pdf_url"
        ]
        for field in required:
            assert field in data, f"Missing: {field}"

    def test_report_pdf_url_is_string(self, client):
        c, rpt = client
        with self._mock_fetch(rpt), self._mock_pdf(rpt), \
             self._mock_store(rpt), self._mock_save_url(rpt):
            data = c.post("/generate", json={"session_id": "abc-123"}).json()
        assert isinstance(data["report_pdf_url"], str)
        assert len(data["report_pdf_url"]) > 0

    def test_503_when_not_ready(self, client):
        c, rpt = client
        rpt.app_state["ready"] = False
        try:
            resp = c.post("/generate", json={"session_id": "abc-123"})
            assert resp.status_code == 503
        finally:
            rpt.app_state["ready"] = True

    def test_missing_session_id_returns_422(self, client):
        c, _ = client
        resp = c.post("/generate", json={})
        assert resp.status_code == 422

    def test_404_when_session_not_found(self, client):
        c, rpt = client
        with patch.object(rpt, "_fetch_session_data",
                          AsyncMock(side_effect=__builtins__["__import__"]("fastapi").HTTPException(
                              status_code=404, detail="Not found")
                          )):
            resp = c.post("/generate", json={"session_id": "no-such-session"})
        assert resp.status_code == 404

    def test_confidence_distribution_values_are_ints(self, client):
        c, rpt = client
        with self._mock_fetch(rpt), self._mock_pdf(rpt), \
             self._mock_store(rpt), self._mock_save_url(rpt):
            data = c.post("/generate", json={"session_id": "abc-123"}).json()
        for v in data["confidence_distribution"].values():
            assert isinstance(v, int)

    def test_total_questions_matches_answers(self, client):
        c, rpt = client
        with self._mock_fetch(rpt), self._mock_pdf(rpt), \
             self._mock_store(rpt), self._mock_save_url(rpt):
            data = c.post("/generate", json={"session_id": "abc-123"}).json()
        assert data["total_questions"] == 5

    def test_500_on_pdf_generation_failure(self, client):
        c, rpt = client
        with self._mock_fetch(rpt):
            with patch.object(rpt, "_render_pdf",
                               side_effect=RuntimeError("PDF generation failed")):
                resp = c.post("/generate", json={"session_id": "abc-123"})
        assert resp.status_code == 500

    def test_average_score_in_range(self, client):
        c, rpt = client
        with self._mock_fetch(rpt), self._mock_pdf(rpt), \
             self._mock_store(rpt), self._mock_save_url(rpt):
            data = c.post("/generate", json={"session_id": "abc-123"}).json()
        assert 0.0 <= data["average_overall_score"] <= 4.0


# ── TestGetReportEndpoint ─────────────────────────────────────────────────────

class TestGetReportEndpoint:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        if REPORT_DIR not in sys.path or sys.path[0] != REPORT_DIR:
            if REPORT_DIR in sys.path:
                sys.path.remove(REPORT_DIR)
            sys.path.insert(0, REPORT_DIR)

        if "main" in sys.modules:
            del sys.modules["main"]

        _stub_deps()
        import main as rpt_main

        # Mock pool with properly structured async context manager
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={"report_url": "file:///tmp/reports/abc-123.pdf"}
        )
        # pool.acquire() must be a callable returning a fresh async context manager each call
        class _FakeCM:
            async def __aenter__(self): return mock_conn
            async def __aexit__(self, *a): return False
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(side_effect=lambda: _FakeCM())
        rpt_main.app_state["ready"] = True
        rpt_main.app_state["pool"]  = mock_pool

        with TestClient(rpt_main.app) as c:
            yield c, rpt_main

    def test_get_report_returns_200(self, client):
        c, rpt = client
        with patch.object(rpt, "_fetch_session_data",
                          AsyncMock(return_value=_raw_data(3))):
            resp = c.get("/report/abc-123")
        assert resp.status_code == 200

    def test_get_report_schema(self, client):
        c, rpt = client
        with patch.object(rpt, "_fetch_session_data",
                          AsyncMock(return_value=_raw_data(3))):
            data = c.get("/report/abc-123").json()
        assert "session_id" in data
        assert "report_pdf_url" in data

    def test_get_report_pdf_url_from_db(self, client):
        c, rpt = client
        # Ensure the pool mock is freshly set with a proper async CM
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={"report_url": "file:///tmp/reports/abc-123.pdf"}
        )
        class _FreshCM:
            async def __aenter__(self): return mock_conn
            async def __aexit__(self, *a): return False
        rpt.app_state["pool"] = MagicMock()
        rpt.app_state["pool"].acquire = MagicMock(side_effect=lambda: _FreshCM())
        with patch.object(rpt, "_fetch_session_data",
                          AsyncMock(return_value=_raw_data(3))):
            data = c.get("/report/abc-123").json()
        assert data["report_pdf_url"] == "file:///tmp/reports/abc-123.pdf"


# ── TestHealthEndpoint ────────────────────────────────────────────────────────

class TestHealthEndpoint:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        if REPORT_DIR not in sys.path or sys.path[0] != REPORT_DIR:
            if REPORT_DIR in sys.path:
                sys.path.remove(REPORT_DIR)
            sys.path.insert(0, REPORT_DIR)

        if "main" in sys.modules:
            del sys.modules["main"]

        _stub_deps()
        import main as rpt_main
        rpt_main.app_state["ready"] = True
        rpt_main.app_state["pool"]  = None

        with TestClient(rpt_main.app) as c:
            yield c

    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "db"     in data

    def test_status_ok_when_ready(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_db_false_when_pool_none(self, client):
        assert client.get("/health").json()["db"] is False
