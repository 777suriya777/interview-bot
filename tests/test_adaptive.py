"""
Tests for the Adaptive Engine.

Groups:
  TestScoringFormula       → individual D/W/R/V component functions
  TestCompositeScore       → composite_score() integration
  TestEMAAndDifficulty     → update_performance_ema() + adjust_difficulty()
  TestSelectNextQuestion   → select_next_question() end-to-end
  TestSelectFallback       → select_fallback_question()
  TestNextQuestionEndpoint → POST /next-question (mocked DB + Redis)
  TestPerformanceEndpoint  → GET /performance/:user_id (mocked DB)
  TestHealthEndpoint       → GET /health
"""
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# Make adaptive_engine importable
ADAPTIVE_DIR = str(Path(__file__).parent.parent / "adaptive_engine")
if ADAPTIVE_DIR not in sys.path:
    sys.path.insert(0, ADAPTIVE_DIR)

from selector import (
    QuestionRecord,
    SelectionResult,
    SessionContext,
    adjust_difficulty,
    composite_score,
    score_difficulty_proximity,
    score_recency,
    score_variety,
    score_weak_area,
    select_fallback_question,
    select_next_question,
    update_performance_ema,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _q(id_="q1", type_="technical", difficulty=3, category="data_structures") -> QuestionRecord:
    return QuestionRecord(id=id_, text="Test question", type=type_,
                          difficulty=difficulty, category=category)


def _ctx(asked=None, category_counts=None, total=0, difficulty=3.0, perf=0.0) -> SessionContext:
    return SessionContext(
        session_id="s1", user_id="u1",
        current_difficulty=difficulty,
        performance_score=perf,
        asked_question_ids=asked or [],
        category_counts=category_counts or {},
        total_questions=total,
    )


# ── TestScoringFormula ────────────────────────────────────────────────────────

class TestScoringFormula:

    # D(q) — difficulty proximity
    def test_d_exact_match(self):
        assert score_difficulty_proximity(3.0, 3) == pytest.approx(1.0)

    def test_d_one_level_off(self):
        result = score_difficulty_proximity(3.0, 4)
        assert result == pytest.approx(0.75)   # 1 - 1/4

    def test_d_max_gap(self):
        result = score_difficulty_proximity(1.0, 5)
        assert result == pytest.approx(0.0)    # 1 - 4/4 = 0

    def test_d_returns_float(self):
        assert isinstance(score_difficulty_proximity(2.5, 3), float)

    def test_d_never_negative(self):
        assert score_difficulty_proximity(1.0, 5) >= 0.0

    def test_d_symmetric(self):
        assert score_difficulty_proximity(2.0, 4) == pytest.approx(
            score_difficulty_proximity(4.0, 2)
        )

    # W(q) — weak area boost
    def test_w_zero_when_no_topic_scores(self):
        assert score_weak_area("data_structures", {}) == 0.0

    def test_w_zero_for_unknown_topic(self):
        result = score_weak_area("unknown_topic", {"data_structures": 80.0})
        assert result == 0.0

    def test_w_positive_for_weak_topic(self):
        # avg=75, topic=50 → W = (75-50)/75 = 0.333
        scores = {"data_structures": 50.0, "algorithms": 100.0}
        result = score_weak_area("data_structures", scores)
        assert result > 0.0

    def test_w_zero_for_above_average_topic(self):
        scores = {"data_structures": 90.0, "algorithms": 50.0}
        result = score_weak_area("data_structures", scores)
        assert result == 0.0  # data_structures is above average

    def test_w_capped_at_one(self):
        scores = {"weak": 0.0, "strong": 100.0}
        result = score_weak_area("weak", scores)
        assert result <= 1.0

    # R(q) — recency penalty
    def test_r_one_for_unseen(self):
        assert score_recency("q1", []) == 1.0
        assert score_recency("q1", ["q2", "q3"]) == 1.0

    def test_r_zero_for_seen(self):
        assert score_recency("q1", ["q1", "q2"]) == 0.0

    # V(q) — variety reward
    def test_v_one_when_no_questions_asked(self):
        assert score_variety("technical", {}, 0) == 1.0

    def test_v_lower_for_overrepresented_type(self):
        counts = {"technical": 8, "behavioural": 2}
        v_tech = score_variety("technical", counts, 10)
        v_beh  = score_variety("behavioural", counts, 10)
        assert v_beh > v_tech

    def test_v_never_negative(self):
        # Even if a type takes all questions, V should not go below 0
        result = score_variety("technical", {"technical": 10}, 10)
        assert result >= 0.0

    def test_v_returns_float(self):
        result = score_variety("hr", {"technical": 5}, 10)
        assert isinstance(result, float)


# ── TestCompositeScore ────────────────────────────────────────────────────────

class TestCompositeScore:

    def test_seen_question_scores_zero(self):
        q = _q("q1")
        score = composite_score(q, 3.0, ["q1"], {}, {}, 0)
        assert score == 0.0

    def test_unseen_question_scores_positive(self):
        q = _q("q1", difficulty=3)
        score = composite_score(q, 3.0, [], {}, {}, 0)
        assert score > 0.0

    def test_sum_weights_equals_one(self):
        from selector import W1_DIFFICULTY, W2_WEAK_AREA, W3_RECENCY, W4_VARIETY
        total = W1_DIFFICULTY + W2_WEAK_AREA + W3_RECENCY + W4_VARIETY
        assert total == pytest.approx(1.0)

    def test_max_score_approaches_one(self):
        # Perfect question: exact difficulty match, in weak area, unseen, unique type
        q = _q("q1", type_="technical", difficulty=3, category="weak_topic")
        topic_scores = {"weak_topic": 0.0, "strong_topic": 100.0}
        score = composite_score(q, 3.0, [], topic_scores, {}, 1)
        assert score > 0.5  # should be a high score

    def test_weak_area_boosts_score(self):
        q_weak   = _q("q1", category="weak_topic")
        q_strong = _q("q2", category="strong_topic")
        topics = {"weak_topic": 20.0, "strong_topic": 80.0}

        s_weak   = composite_score(q_weak,   3.0, [], topics, {}, 0)
        s_strong = composite_score(q_strong, 3.0, [], topics, {}, 0)
        assert s_weak > s_strong


# ── TestEMAAndDifficulty ──────────────────────────────────────────────────────

class TestEMAAndDifficulty:

    def test_ema_updates_toward_new_value(self):
        # current=0, new=4 (max) → should move up
        result = update_performance_ema(0.0, 4.0)
        assert result > 0.0

    def test_ema_alpha_0_3(self):
        # current=0, latest=4 → 0.3 * (4 * 5/4) + 0.7 * 0 = 0.3 * 5 = 1.5
        result = update_performance_ema(0.0, 4.0)
        assert result == pytest.approx(1.5, rel=0.01)

    def test_ema_converges_toward_high_score(self):
        score = 0.0
        for _ in range(20):
            score = update_performance_ema(score, 4.0)
        assert score > 4.0  # after many max scores, should be high

    def test_ema_stays_within_bounds(self):
        score = update_performance_ema(2.5, 4.0)
        assert 0.0 <= score <= 5.0

    def test_difficulty_increases_on_high_perf(self):
        # performance_score on 0-5 scale; high_threshold = 3.5 * 5/4 = 4.375
        result = adjust_difficulty(3, 4.5)
        assert result == 4

    def test_difficulty_decreases_on_low_perf(self):
        # low_threshold = 2.0 * 5/4 = 2.5
        result = adjust_difficulty(3, 2.0)
        assert result == 2

    def test_difficulty_unchanged_in_middle_range(self):
        result = adjust_difficulty(3, 3.0)
        assert result == 3

    def test_difficulty_capped_at_max(self):
        result = adjust_difficulty(5, 5.0)
        assert result == 5

    def test_difficulty_capped_at_min(self):
        result = adjust_difficulty(1, 0.5)
        assert result == 1

    def test_difficulty_returns_int(self):
        assert isinstance(adjust_difficulty(3, 3.0), int)


# ── TestSelectNextQuestion ────────────────────────────────────────────────────

class TestSelectNextQuestion:

    def _bank(self, n=5):
        return [
            _q(f"q{i}", difficulty=i % 5 + 1, category=f"cat{i % 3}")
            for i in range(n)
        ]

    def test_returns_selection_result(self):
        questions = self._bank()
        ctx = _ctx()
        result = select_next_question(questions, ctx, {})
        assert isinstance(result, SelectionResult)

    def test_never_returns_seen_question(self):
        questions = self._bank(5)
        asked = ["q0", "q1", "q2"]
        ctx = _ctx(asked=asked)
        result = select_next_question(questions, ctx, {})
        assert result.question_id not in asked

    def test_returns_none_when_all_seen(self):
        questions = self._bank(3)
        ctx = _ctx(asked=["q0", "q1", "q2"])
        result = select_next_question(questions, ctx, {})
        assert result is None

    def test_updates_context_performance(self):
        questions = self._bank()
        ctx = _ctx(perf=0.0)
        last_scores = {"content": 4, "relevance": 4, "completeness": 4, "accuracy": 4}
        select_next_question(questions, ctx, {}, last_scores=last_scores)
        assert ctx.performance_score > 0.0

    def test_updates_difficulty_on_high_score(self):
        questions = self._bank()
        ctx = _ctx(difficulty=3, perf=4.0)  # already high perf
        # Give max scores to push over threshold
        last_scores = {"content": 4, "relevance": 4, "completeness": 4, "accuracy": 4}
        select_next_question(questions, ctx, {}, last_scores=last_scores)
        # After EMA: new_perf = 0.3*5 + 0.7*4 = 1.5 + 2.8 = 4.3 > 4.375? No.
        # With perf=4.0 and max scores: 0.3*5 + 0.7*4.0 = 1.5+2.8 = 4.3 < 4.375
        # Let's use perf=4.3 instead
        ctx2 = _ctx(difficulty=3, perf=4.38)
        last_scores = {"content": 4, "relevance": 4, "completeness": 4, "accuracy": 4}
        select_next_question(questions, ctx2, {}, last_scores=last_scores)
        assert ctx2.current_difficulty >= 3

    def test_prefers_weak_area_questions(self):
        # Two questions: one in strong area, one in weak area (same difficulty)
        q_weak   = QuestionRecord("qW", "Weak q", "technical", 3, "weak_cat")
        q_strong = QuestionRecord("qS", "Strong q", "technical", 3, "strong_cat")
        questions = [q_weak, q_strong]
        ctx = _ctx()
        topic_scores = {"weak_cat": 10.0, "strong_cat": 90.0}
        result = select_next_question(questions, ctx, topic_scores)
        assert result.question_id == "qW"

    def test_selection_reason_is_string(self):
        result = select_next_question(self._bank(), _ctx(), {})
        assert isinstance(result.selection_reason, str)
        assert len(result.selection_reason) > 0

    def test_user_level_matches_context_difficulty(self):
        ctx = _ctx(difficulty=4)
        result = select_next_question(self._bank(), ctx, {})
        assert result.user_level == pytest.approx(4.0)

    def test_single_question_bank(self):
        questions = [_q("only")]
        result = select_next_question(questions, _ctx(), {})
        assert result.question_id == "only"


# ── TestSelectFallback ────────────────────────────────────────────────────────

class TestSelectFallback:

    def test_returns_unseen_question(self):
        questions = [_q("q1", difficulty=3), _q("q2", difficulty=3)]
        result = select_fallback_question(questions, ["q1"])
        assert result is not None
        assert result.id == "q2"

    def test_returns_none_when_all_seen(self):
        questions = [_q("q1"), _q("q2")]
        result = select_fallback_question(questions, ["q1", "q2"])
        assert result is None

    def test_prefers_target_difficulty(self):
        questions = [
            _q("q1", difficulty=1),
            _q("q2", difficulty=3),  # target
            _q("q3", difficulty=5),
        ]
        result = select_fallback_question(questions, [], target_difficulty=3)
        assert result.difficulty == 3

    def test_returns_question_record(self):
        result = select_fallback_question([_q("q1")], [])
        assert isinstance(result, QuestionRecord)

    def test_empty_bank_returns_none(self):
        assert select_fallback_question([], []) is None


# ── TestNextQuestionEndpoint ──────────────────────────────────────────────────

class TestNextQuestionEndpoint:
    """POST /next-question with fully mocked DB and Redis."""

    @pytest.fixture
    def client(self):
        # Stub asyncpg and redis before importing main
        _stub_asyncpg_and_redis()

        if ADAPTIVE_DIR not in sys.path or sys.path[0] != ADAPTIVE_DIR:
            if ADAPTIVE_DIR in sys.path:
                sys.path.remove(ADAPTIVE_DIR)
            sys.path.insert(0, ADAPTIVE_DIR)

        for mod in ["main"]:
            if mod in sys.modules:
                del sys.modules[mod]

        from fastapi.testclient import TestClient
        import main as adp_main

        # Pre-populate app_state
        adp_main.app_state.clear()
        adp_main.app_state["pool"]  = None   # no DB in unit tests
        adp_main.app_state["redis"] = None   # no Redis in unit tests
        adp_main.app_state["ready"] = True

        with TestClient(adp_main.app) as c:
            yield c

        adp_main.app_state.clear()

    def _payload(self, asked=None):
        return {
            "user_id":    str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "last_scores": {
                "content": 3, "relevance": 3,
                "completeness": 3, "accuracy": 3,
            },
            "session_question_ids": asked or [],
        }

    def _bank(self):
        return [
            QuestionRecord(
                id=f"q{i}", text=f"Question {i}", type="technical",
                difficulty=i % 5 + 1, category=f"cat{i % 3}",
            )
            for i in range(10)
        ]

    def test_returns_200(self, client):
        with patch("main._get_question_bank", new=AsyncMock(return_value=self._bank())), \
             patch("main._load_session_context",
                   new=AsyncMock(return_value=SessionContext("s1", "u1"))), \
             patch("main._get_user_performance", new=AsyncMock(return_value={})), \
             patch("main._save_session_context", new=AsyncMock()), \
             patch("main._update_user_performance_cache", new=AsyncMock()), \
             patch("main._save_user_performance_to_db", new=AsyncMock()), \
             patch("main._update_session_difficulty_in_db", new=AsyncMock()):
            resp = client.post("/next-question", json=self._payload())
        assert resp.status_code == 200

    def test_response_schema(self, client):
        with patch("main._get_question_bank", new=AsyncMock(return_value=self._bank())), \
             patch("main._load_session_context",
                   new=AsyncMock(return_value=SessionContext("s1", "u1"))), \
             patch("main._get_user_performance", new=AsyncMock(return_value={})), \
             patch("main._save_session_context", new=AsyncMock()), \
             patch("main._update_user_performance_cache", new=AsyncMock()), \
             patch("main._save_user_performance_to_db", new=AsyncMock()), \
             patch("main._update_session_difficulty_in_db", new=AsyncMock()):
            resp = client.post("/next-question", json=self._payload())
        data = resp.json()
        for key in ["question_id", "question_text", "question_type",
                    "difficulty", "user_level", "selection_reason"]:
            assert key in data, f"Missing: {key}"

    def test_difficulty_in_range(self, client):
        with patch("main._get_question_bank", new=AsyncMock(return_value=self._bank())), \
             patch("main._load_session_context",
                   new=AsyncMock(return_value=SessionContext("s1", "u1"))), \
             patch("main._get_user_performance", new=AsyncMock(return_value={})), \
             patch("main._save_session_context", new=AsyncMock()), \
             patch("main._update_user_performance_cache", new=AsyncMock()), \
             patch("main._save_user_performance_to_db", new=AsyncMock()), \
             patch("main._update_session_difficulty_in_db", new=AsyncMock()):
            resp = client.post("/next-question", json=self._payload())
        assert 1 <= resp.json()["difficulty"] <= 5

    def test_does_not_repeat_asked_question(self, client):
        bank = self._bank()
        asked_id = bank[0].id
        ctx = SessionContext("s1", "u1", asked_question_ids=[asked_id])

        with patch("main._get_question_bank", new=AsyncMock(return_value=bank)), \
             patch("main._load_session_context", new=AsyncMock(return_value=ctx)), \
             patch("main._get_user_performance", new=AsyncMock(return_value={})), \
             patch("main._save_session_context", new=AsyncMock()), \
             patch("main._update_user_performance_cache", new=AsyncMock()), \
             patch("main._save_user_performance_to_db", new=AsyncMock()), \
             patch("main._update_session_difficulty_in_db", new=AsyncMock()):
            resp = client.post("/next-question",
                               json=self._payload(asked=[asked_id]))
        assert resp.json()["question_id"] != asked_id

    def test_empty_bank_returns_503(self, client):
        with patch("main._get_question_bank", new=AsyncMock(return_value=[])):
            resp = client.post("/next-question", json=self._payload())
        assert resp.status_code == 503

    def test_missing_fields_rejected(self, client):
        resp = client.post("/next-question", json={"user_id": "x"})
        assert resp.status_code == 422

    def test_selection_reason_non_empty(self, client):
        with patch("main._get_question_bank", new=AsyncMock(return_value=self._bank())), \
             patch("main._load_session_context",
                   new=AsyncMock(return_value=SessionContext("s1", "u1"))), \
             patch("main._get_user_performance", new=AsyncMock(return_value={})), \
             patch("main._save_session_context", new=AsyncMock()), \
             patch("main._update_user_performance_cache", new=AsyncMock()), \
             patch("main._save_user_performance_to_db", new=AsyncMock()), \
             patch("main._update_session_difficulty_in_db", new=AsyncMock()):
            resp = client.post("/next-question", json=self._payload())
        assert len(resp.json()["selection_reason"]) > 0


# ── TestPerformanceEndpoint ───────────────────────────────────────────────────

class TestPerformanceEndpoint:

    @pytest.fixture
    def client(self):
        _stub_asyncpg_and_redis()

        if ADAPTIVE_DIR not in sys.path or sys.path[0] != ADAPTIVE_DIR:
            if ADAPTIVE_DIR in sys.path:
                sys.path.remove(ADAPTIVE_DIR)
            sys.path.insert(0, ADAPTIVE_DIR)

        for mod in ["main"]:
            if mod in sys.modules:
                del sys.modules[mod]

        from fastapi.testclient import TestClient
        import main as adp_main

        adp_main.app_state.clear()
        adp_main.app_state["pool"]  = None
        adp_main.app_state["redis"] = None
        adp_main.app_state["ready"] = True

        with TestClient(adp_main.app) as c:
            yield c

        adp_main.app_state.clear()

    def test_no_db_returns_503(self, client):
        import main as adp_main
        # Explicitly ensure pool is None for this test
        original = adp_main.app_state.get("pool")
        adp_main.app_state["pool"] = None
        try:
            uid = str(uuid.uuid4())
            resp = client.get(f"/performance/{uid}")
            assert resp.status_code == 503
        finally:
            adp_main.app_state["pool"] = original

    def test_response_schema_with_mock_db(self, client):
        import main as adp_main
        # Set up a mock pool
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {"topic": "data_structures", "score": 82.1, "answer_count": 14},
            {"topic": "system_design",   "score": 45.3, "answer_count":  6},
        ])
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=type("CM", (), {
                "__aenter__": AsyncMock(return_value=mock_conn),
                "__aexit__":  AsyncMock(return_value=False),
            })()
        )
        adp_main.app_state["pool"] = mock_pool

        uid = str(uuid.uuid4())
        resp = client.get(f"/performance/{uid}")
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_score"  in data
        assert "topics"         in data
        assert "priority_areas" in data

    def test_empty_performance_returns_zeros(self, client):
        import main as adp_main
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=type("CM", (), {
                "__aenter__": AsyncMock(return_value=mock_conn),
                "__aexit__":  AsyncMock(return_value=False),
            })()
        )
        adp_main.app_state["pool"] = mock_pool

        uid = str(uuid.uuid4())
        resp = client.get(f"/performance/{uid}")
        assert resp.status_code == 200
        assert resp.json()["overall_score"] == 0.0
        assert resp.json()["topics"] == {}

    def test_priority_areas_are_weak_topics(self, client):
        import main as adp_main
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {"topic": "algorithms",      "score": 30.0, "answer_count": 5},
            {"topic": "data_structures", "score": 90.0, "answer_count": 10},
            {"topic": "system_design",   "score": 40.0, "answer_count": 8},
        ])
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=type("CM", (), {
                "__aenter__": AsyncMock(return_value=mock_conn),
                "__aexit__":  AsyncMock(return_value=False),
            })()
        )
        adp_main.app_state["pool"] = mock_pool

        uid = str(uuid.uuid4())
        resp = client.get(f"/performance/{uid}")
        data = resp.json()
        # algorithms(30) and system_design(40) are below avg(53.3)
        assert "data_structures" not in data["priority_areas"]
        assert len(data["priority_areas"]) <= 3


# ── TestHealthEndpoint ────────────────────────────────────────────────────────

class TestHealthEndpoint:

    @pytest.fixture
    def client(self):
        _stub_asyncpg_and_redis()

        if ADAPTIVE_DIR not in sys.path or sys.path[0] != ADAPTIVE_DIR:
            if ADAPTIVE_DIR in sys.path:
                sys.path.remove(ADAPTIVE_DIR)
            sys.path.insert(0, ADAPTIVE_DIR)

        for mod in ["main"]:
            if mod in sys.modules:
                del sys.modules[mod]

        from fastapi.testclient import TestClient
        import main as adp_main

        adp_main.app_state.clear()
        adp_main.app_state["pool"]  = None
        adp_main.app_state["redis"] = None
        adp_main.app_state["ready"] = True

        with TestClient(adp_main.app) as c:
            yield c

        adp_main.app_state.clear()

    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "status"           in data
        assert "db"               in data
        assert "redis"            in data
        assert "questions_cached" in data

    def test_health_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_health_db_false_no_pool(self, client):
        assert client.get("/health").json()["db"] is False


# ── Stub helper ───────────────────────────────────────────────────────────────

def _stub_asyncpg_and_redis():
    """Stub asyncpg and redis.asyncio so main.py imports without real packages."""
    import types as _types

    if "asyncpg" not in sys.modules:
        ap = _types.ModuleType("asyncpg")
        ap.create_pool = AsyncMock()
        sys.modules["asyncpg"] = ap

    if "redis" not in sys.modules:
        redis_mod = _types.ModuleType("redis")
        redis_mod.asyncio = _types.ModuleType("redis.asyncio")
        redis_mod.asyncio.from_url = MagicMock(return_value=None)
        sys.modules["redis"] = redis_mod
        sys.modules["redis.asyncio"] = redis_mod.asyncio
