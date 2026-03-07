"""
SQLAlchemy 2.0 async ORM models.

Import this module in services that need full ORM access
(adaptive_engine, report_service). Services that only need
raw SQL can use asyncpg directly via connection.py.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, CheckConstraint, ForeignKey, Index, Integer,
    Numeric, SmallInteger, String, Text, UniqueConstraint,
    func,
)
from sqlalchemy import DateTime
from sqlalchemy.dialects.postgresql import JSONB, UUID

# PostgreSQL TIMESTAMPTZ = DateTime with timezone=True in SQLAlchemy
# The schema.sql uses TIMESTAMPTZ directly; ORM uses this equivalent
TIMESTAMPTZ = DateTime(timezone=True)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base — all models inherit from this."""
    pass


# ── users ─────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        # Partial unique index: only enforced when oauth_provider is set.
        # DDL-level index is in schema.sql; this keeps ORM metadata in sync.
        Index(
            "idx_users_oauth",
            "oauth_provider",
            "oauth_sub",
            unique=True,
            postgresql_where="oauth_provider IS NOT NULL",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Primary key — UUIDv4",
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, comment="NULL for OAuth-only accounts"
    )
    oauth_provider: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment="'google' | 'github' | NULL"
    )
    oauth_sub: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, comment="Provider user id (OIDC sub claim)"
    )
    preferences: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}", comment="{target_role, difficulty_pref}"
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    last_active_at: Mapped[datetime] = mapped_column(
        TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    # Relationships
    sessions: Mapped[list["Session"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    performance: Mapped[list["UserPerformance"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email}>"


# ── questions ─────────────────────────────────────────────────────────

class Question(Base):
    __tablename__ = "questions"
    __table_args__ = (
        CheckConstraint("type IN ('technical','behavioural','hr')", name="ck_question_type"),
        CheckConstraint("difficulty BETWEEN 1 AND 5", name="ck_question_difficulty"),
        # Partial composite index — used by adaptive engine on every question selection
        Index(
            "idx_questions_type_diff",
            "type",
            "difficulty",
            postgresql_where="is_active = TRUE",
        ),
        Index(
            "idx_questions_category",
            "category",
            postgresql_where="is_active = TRUE",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="'technical' | 'behavioural' | 'hr'"
    )
    difficulty: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, comment="1 (easiest) – 5 (hardest)"
    )
    category: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True,
        comment="e.g. 'data_structures', 'system_design', 'leadership'"
    )
    key_concepts: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default="[]",
        comment='["recursion", "time_complexity"]'
    )
    created_by: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, comment="'admin' | 'hr_professional' | 'system'"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="TRUE"
    )
    flag_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0",
        comment="User flags for quality review; reviewed when > threshold"
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    # Relationships
    answers: Mapped[list["Answer"]] = relationship(back_populates="question")

    def __repr__(self) -> str:
        return f"<Question id={self.id} type={self.type} difficulty={self.difficulty}>"


# ── sessions ──────────────────────────────────────────────────────────

class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        CheckConstraint(
            "interview_type IN ('technical','behavioural','hr','mixed')",
            name="ck_session_interview_type",
        ),
        CheckConstraint(
            "status IN ('active','completed','abandoned')",
            name="ck_session_status",
        ),
        # Most common query pattern: user's recent sessions (dashboard)
        Index("idx_sessions_user", "user_id", "started_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    interview_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="active"
    )
    target_role: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # Adaptive difficulty 1-5; updated by EMA after each answer
    difficulty_level: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default="3"
    )
    # EMA of overall_scores (0-4 scale); used by adaptive engine
    performance_score: Mapped[float] = mapped_column(
        Numeric(5, 2), nullable=False, server_default="0"
    )
    started_at: Mapped[datetime] = mapped_column(
        TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMPTZ, nullable=True)
    report_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="sessions")
    answers: Mapped[list["Answer"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Answer.submitted_at"
    )

    def __repr__(self) -> str:
        return f"<Session id={self.id} user_id={self.user_id} status={self.status}>"


# ── answers ───────────────────────────────────────────────────────────

class Answer(Base):
    __tablename__ = "answers"
    __table_args__ = (
        CheckConstraint("input_mode IN ('text','voice')", name="ck_answer_input_mode"),
        CheckConstraint("content_score BETWEEN 0 AND 4", name="ck_answer_content"),
        CheckConstraint("relevance_score BETWEEN 0 AND 4", name="ck_answer_relevance"),
        CheckConstraint("completeness_score BETWEEN 0 AND 4", name="ck_answer_completeness"),
        CheckConstraint("accuracy_score BETWEEN 0 AND 4", name="ck_answer_accuracy"),
        CheckConstraint(
            "confidence_label IN ('high','moderate','low','anxious')",
            name="ck_answer_confidence",
        ),
        # "Get all answers for this session in order" — report service query
        Index("idx_answers_session", "session_id", "submitted_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("questions.id"),  # No cascade: questions outlive answers
        nullable=False,
    )
    input_mode: Mapped[str] = mapped_column(
        String(10), nullable=False, comment="'text' | 'voice'"
    )
    transcript: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="Raw typed text OR Whisper transcript"
    )

    # ── NLP Scores (BERT, 0-4) ─────────────────────────────────────
    content_score: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    relevance_score: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    completeness_score: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    accuracy_score: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    overall_score: Mapped[Optional[float]] = mapped_column(Numeric(4, 2), nullable=True)

    # ── Confidence (BiLSTM, independent of content) ───────────────
    confidence_label: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # ── Sentiment Details ─────────────────────────────────────────
    # [{word: "I think", position: 12}, ...]
    hedging_flags: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )

    # ── ASR / Delivery (NULL columns when input_mode == 'text') ──
    # ["fast_speech", "nervous_pitch"]
    delivery_flags: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    speaking_rate_wpm: Mapped[Optional[float]] = mapped_column(Numeric(6, 2), nullable=True)
    pitch_std: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)

    submitted_at: Mapped[datetime] = mapped_column(
        TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    # Relationships
    session: Mapped["Session"] = relationship(back_populates="answers")
    question: Mapped["Question"] = relationship(back_populates="answers")

    def __repr__(self) -> str:
        return f"<Answer id={self.id} session_id={self.session_id} overall={self.overall_score}>"


# ── user_performance ──────────────────────────────────────────────────

class UserPerformance(Base):
    __tablename__ = "user_performance"
    __table_args__ = (
        UniqueConstraint("user_id", "topic", name="uq_user_topic"),
        Index("idx_perf_user", "user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    topic: Mapped[str] = mapped_column(
        String(100), nullable=False,
        comment="Matches question.category (e.g. 'data_structures')"
    )
    # EMA score 0-100 (answer overall_score × 25, then EMA'd)
    score: Mapped[float] = mapped_column(
        Numeric(5, 2), nullable=False, server_default="0"
    )
    answer_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="performance")

    def __repr__(self) -> str:
        return f"<UserPerformance user={self.user_id} topic={self.topic} score={self.score}>"
