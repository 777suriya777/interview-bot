-- ============================================================
-- Interview Bot — PostgreSQL 16 Schema
-- All timestamps UTC. UUIDs for all PKs.
-- Auto-executed by postgres Docker entrypoint on first boot.
-- ============================================================

-- Enable UUID generation (built-in since PG 13)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── users ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    name            VARCHAR(255) NOT NULL,
    -- NULL when user registered via OAuth only
    hashed_password VARCHAR(255),
    -- 'google' | 'github' | NULL for email/password users
    oauth_provider  VARCHAR(50),
    -- Provider's own user identifier (sub claim from OIDC)
    oauth_sub       VARCHAR(255),
    -- JSONB blob: { "target_role": "SWE", "difficulty_pref": 3 }
    preferences     JSONB       NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email
    ON users(email);

-- Partial index: only OAuth users have oauth_sub; speeds up OAuth login lookup
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_oauth
    ON users(oauth_provider, oauth_sub)
    WHERE oauth_provider IS NOT NULL;

-- ── questions ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS questions (
    id           UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    text         TEXT         NOT NULL,
    -- Enforced domain: only these three types exist
    type         VARCHAR(20)  NOT NULL CHECK (type IN ('technical','behavioural','hr')),
    -- 1 = easiest, 5 = hardest
    difficulty   SMALLINT     NOT NULL CHECK (difficulty BETWEEN 1 AND 5),
    -- e.g. 'data_structures', 'system_design', 'leadership', 'salary'
    category     VARCHAR(100),
    -- Array of concept tags: ["recursion","time_complexity"]
    key_concepts JSONB        NOT NULL DEFAULT '[]',
    -- Who created this question
    created_by   VARCHAR(100),
    -- Soft-delete: inactive questions are excluded from selection
    is_active    BOOLEAN      NOT NULL DEFAULT TRUE,
    -- Users can flag low-quality questions; reviewed when flag_count > threshold
    flag_count   INT          NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Composite index on type+difficulty WHERE active — the adaptive engine
-- queries exactly this combination on every question selection
CREATE INDEX IF NOT EXISTS idx_questions_type_diff
    ON questions(type, difficulty)
    WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_questions_category
    ON questions(category)
    WHERE is_active = TRUE;

-- ── sessions ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- 'mixed' allows the adaptive engine to pull from all question types
    interview_type    VARCHAR(20) NOT NULL
        CHECK (interview_type IN ('technical','behavioural','hr','mixed')),
    status            VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active','completed','abandoned')),
    target_role       VARCHAR(100),
    -- Current adaptive difficulty (1-5); updated by EMA after each answer
    difficulty_level  SMALLINT    NOT NULL DEFAULT 3,
    -- EMA of answer overall_scores (0-4 scale); updated in real-time
    performance_score NUMERIC(5,2) NOT NULL DEFAULT 0,
    started_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at          TIMESTAMPTZ,
    -- S3 / blob URL of generated PDF report; populated when session completes
    report_url        VARCHAR(500),
    resume_text       TEXT
);

-- Most common query: "get user's recent sessions for dashboard"
CREATE INDEX IF NOT EXISTS idx_sessions_user
    ON sessions(user_id, started_at DESC);

-- ── answers ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS answers (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    question_id         UUID        NOT NULL REFERENCES questions(id),
    -- Input mode determines whether ASR pipeline ran
    input_mode          VARCHAR(10) NOT NULL CHECK (input_mode IN ('text','voice')),
    -- Raw answer text (either typed or ASR-transcribed)
    transcript          TEXT,

    -- ── NLP Scores (BERT output, 0-4 each) ──────────────────────────
    content_score       SMALLINT    CHECK (content_score BETWEEN 0 AND 4),
    relevance_score     SMALLINT    CHECK (relevance_score BETWEEN 0 AND 4),
    completeness_score  SMALLINT    CHECK (completeness_score BETWEEN 0 AND 4),
    accuracy_score      SMALLINT    CHECK (accuracy_score BETWEEN 0 AND 4),
    -- Weighted average of the 4 dimension scores
    overall_score       NUMERIC(4,2),

    -- ── Confidence (BiLSTM output, independent of content) ──────────
    confidence_label    VARCHAR(20)
        CHECK (confidence_label IN ('high','moderate','low','anxious')),

    -- ── Sentiment Details ────────────────────────────────────────────
    -- Array of {word, position} objects for hedging words detected
    hedging_flags       JSONB       NOT NULL DEFAULT '[]',

    -- ── ASR / Delivery (NULL for text input) ────────────────────────
    -- Array of flag strings: ['fast_speech','nervous_pitch']
    delivery_flags      JSONB       NOT NULL DEFAULT '[]',
    speaking_rate_wpm   NUMERIC(6,2),
    pitch_std           NUMERIC(8,4),

    submitted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- "Get all answers for a session, in order" — used by report service
CREATE INDEX IF NOT EXISTS idx_answers_session
    ON answers(session_id, submitted_at);

-- ── user_performance ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_performance (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- Topic matches question.category (e.g. 'data_structures')
    topic        VARCHAR(100) NOT NULL,
    -- EMA score 0-100 (scaled from 0-4 answer scores × 25)
    score        NUMERIC(5,2) NOT NULL DEFAULT 0,
    answer_count INT         NOT NULL DEFAULT 0,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- One row per (user, topic) — upserted on each answer
    UNIQUE (user_id, topic)
);

CREATE INDEX IF NOT EXISTS idx_perf_user
    ON user_performance(user_id);
