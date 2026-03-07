"""
Unit tests for the database layer.

These tests use pytest-asyncio and an in-memory SQLite engine for speed
(no PostgreSQL required). The small number of PostgreSQL-specific types
(UUID, JSONB, TIMESTAMPTZ) are mocked via SQLAlchemy's type coercions.

Run:
    pytest tests/test_database.py -v
"""
import asyncio
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Helpers ────────────────────────────────────────────────────────────

def make_uuid() -> str:
    return str(uuid.uuid4())


# ── Test: seed.py question data ────────────────────────────────────────

class TestSeedData:
    """Validate the QUESTIONS list in seed.py without touching the DB."""

    def _load_questions(self):
        """Import QUESTIONS dynamically to avoid needing a live DB."""
        import sys
        from pathlib import Path
        db_dir = str(Path(__file__).parent.parent / "database")
        if db_dir not in sys.path:
            sys.path.insert(0, db_dir)
        # Guard against __main__ block by setting argv[0] to empty
        import importlib
        # Temporarily mask __name__ so the if __name__ == '__main__' block skips
        original_argv = sys.argv[:]
        sys.argv = [""]
        try:
            import seed as seed_mod
            importlib.reload(seed_mod)
            return seed_mod.QUESTIONS
        finally:
            sys.argv = original_argv

    def test_question_count(self):
        """Expect exactly 40 questions as documented in seed.py."""
        questions = self._load_questions()
        assert len(questions) == 40, f"Expected 40, got {len(questions)}"

    def test_type_distribution(self):
        """15 technical, 15 behavioural, 10 HR."""
        questions = self._load_questions()
        types = [q["type"] for q in questions]
        assert types.count("technical") == 15
        assert types.count("behavioural") == 15
        assert types.count("hr") == 10

    def test_all_types_valid(self):
        valid = {"technical", "behavioural", "hr"}
        for q in self._load_questions():
            assert q["type"] in valid, f"Invalid type: {q['type']}"

    def test_difficulty_range(self):
        """All difficulties must be between 1 and 5 inclusive."""
        for q in self._load_questions():
            assert 1 <= q["difficulty"] <= 5, (
                f"Difficulty {q['difficulty']} out of range for: {q['text'][:60]}"
            )

    def test_difficulty_spread(self):
        """Every difficulty level 1-5 must appear at least once."""
        difficulties = {q["difficulty"] for q in self._load_questions()}
        assert difficulties == {1, 2, 3, 4, 5}

    def test_required_fields_present(self):
        required = {"text", "type", "difficulty", "category", "key_concepts", "created_by"}
        for q in self._load_questions():
            missing = required - q.keys()
            assert not missing, f"Missing fields {missing} in: {q.get('text','?')[:60]}"

    def test_key_concepts_is_list(self):
        for q in self._load_questions():
            assert isinstance(q["key_concepts"], list), (
                f"key_concepts must be list for: {q['text'][:60]}"
            )

    def test_no_empty_text(self):
        for q in self._load_questions():
            assert q["text"].strip(), "Empty question text found"

    def test_technical_categories(self):
        """Technical questions should cover core CS categories."""
        questions = self._load_questions()
        tech_cats = {q["category"] for q in questions if q["type"] == "technical"}
        expected = {"data_structures", "algorithms", "system_design", "databases", "os_concepts"}
        assert expected.issubset(tech_cats), f"Missing categories: {expected - tech_cats}"

    def test_json_serializable(self):
        """All question data must be JSON-serialisable (for asyncpg JSONB insert)."""
        for q in self._load_questions():
            try:
                json.dumps(q["key_concepts"])
            except (TypeError, ValueError) as e:
                pytest.fail(f"key_concepts not JSON-serialisable: {e}")


# ── Test: models.py ORM definitions ────────────────────────────────────

class TestOrmModels:
    """Test that ORM model definitions are correct without a live DB."""

    def test_user_model_imports(self):
        from database.models import User
        assert User.__tablename__ == "users"

    def test_question_model_imports(self):
        from database.models import Question
        assert Question.__tablename__ == "questions"

    def test_session_model_imports(self):
        from database.models import Session
        assert Session.__tablename__ == "sessions"

    def test_answer_model_imports(self):
        from database.models import Answer
        assert Answer.__tablename__ == "answers"

    def test_user_performance_model_imports(self):
        from database.models import UserPerformance
        assert UserPerformance.__tablename__ == "user_performance"

    def test_user_has_sessions_relationship(self):
        from database.models import User
        assert hasattr(User, "sessions")

    def test_user_has_performance_relationship(self):
        from database.models import User
        assert hasattr(User, "performance")

    def test_session_has_answers_relationship(self):
        from database.models import Session
        assert hasattr(Session, "answers")

    def test_answer_has_session_relationship(self):
        from database.models import Answer
        assert hasattr(Answer, "session")

    def test_answer_has_question_relationship(self):
        from database.models import Answer
        assert hasattr(Answer, "question")

    def test_question_columns(self):
        """Spot-check that critical columns exist in Question."""
        from database.models import Question
        col_names = {c.key for c in Question.__table__.columns}
        expected = {
            "id", "text", "type", "difficulty", "category",
            "key_concepts", "is_active", "flag_count", "created_at"
        }
        assert expected.issubset(col_names), f"Missing: {expected - col_names}"

    def test_answer_columns(self):
        """Spot-check all NLP score columns exist."""
        from database.models import Answer
        col_names = {c.key for c in Answer.__table__.columns}
        expected = {
            "content_score", "relevance_score", "completeness_score",
            "accuracy_score", "overall_score", "confidence_label",
            "hedging_flags", "delivery_flags", "speaking_rate_wpm", "pitch_std"
        }
        assert expected.issubset(col_names), f"Missing: {expected - col_names}"

    def test_repr_methods(self):
        """__repr__ should not raise for any model."""
        from database.models import User, Question, Session, Answer, UserPerformance

        uid = uuid.uuid4()
        u = User(); u.id = uid; u.email = "t@t.com"
        assert str(uid) in repr(u)

        q = Question(); q.id = uid; q.type = "technical"; q.difficulty = 3
        assert "technical" in repr(q)

        s = Session(); s.id = uid; s.user_id = uid; s.status = "active"
        assert "active" in repr(s)

        a = Answer(); a.id = uid; a.session_id = uid; a.overall_score = 3.5
        assert "3.5" in repr(a)

        p = UserPerformance(); p.user_id = uid; p.topic = "algorithms"; p.score = 75.0
        assert "algorithms" in repr(p)


# ── Test: connection.py URL conversion ─────────────────────────────────

class TestDatabaseConnection:
    """Test URL parsing logic in connection.py (no live DB needed)."""

    def test_url_conversion_plain(self):
        """postgresql:// should be converted to postgresql+asyncpg://"""
        with patch.dict("os.environ", {"DATABASE_URL": "postgresql://user:pw@host/db"}):
            # Re-import to trigger _build_async_url with patched env
            import importlib
            import database.connection as conn_mod
            importlib.reload(conn_mod)
            # The engine URL should contain asyncpg
            assert "asyncpg" in str(conn_mod.engine.url)

    def test_url_conversion_already_asyncpg(self):
        """postgresql+asyncpg:// should be kept as-is."""
        with patch.dict("os.environ", {
            "DATABASE_URL": "postgresql+asyncpg://user:pw@host/db"
        }):
            import importlib
            import database.connection as conn_mod
            importlib.reload(conn_mod)
            assert "asyncpg" in str(conn_mod.engine.url)

    def test_missing_url_raises(self):
        """RuntimeError when DATABASE_URL is not set."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove DATABASE_URL if present
            import os
            os.environ.pop("DATABASE_URL", None)
            import database.connection as conn_mod

            # _build_async_url reads from env at call time
            with pytest.raises(RuntimeError, match="DATABASE_URL"):
                conn_mod._build_async_url()


# ── Test: get_db_session dependency behaviour ──────────────────────────

class TestGetDbSession:
    """Test that get_db_session rolls back on exception."""

    @pytest.fixture(autouse=True)
    def set_db_url(self, monkeypatch):
        """Ensure DATABASE_URL is set for connection.py module import."""
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pw@host/db")

    @pytest.mark.asyncio
    async def test_rollback_on_exception(self):
        """Session must rollback when body raises."""
        from database.connection import AsyncSessionFactory
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        with patch("database.connection.AsyncSessionFactory", return_value=mock_session):
            from database.connection import get_db_session
            gen = get_db_session()
            await gen.__anext__()
            with pytest.raises(ValueError):
                await gen.athrow(ValueError("test error"))

            mock_session.rollback.assert_awaited_once()
            mock_session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_commit_on_success(self):
        """Session must commit on clean exit."""
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        import database.connection as conn_mod
        original = conn_mod.AsyncSessionFactory
        conn_mod.AsyncSessionFactory = MagicMock(return_value=mock_session)
        try:
            from database.connection import get_db_session
            gen = get_db_session()
            # Step into generator up to yield
            await gen.__anext__()
            # Advance past yield — this hits commit() in the try block
            try:
                await gen.asend(None)
            except StopAsyncIteration:
                pass  # generator is exhausted — commit was called
            mock_session.commit.assert_awaited_once()
            mock_session.rollback.assert_not_awaited()
        finally:
            conn_mod.AsyncSessionFactory = original
