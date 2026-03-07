"""
Seed script — inserts sample questions into the questions table.

Usage:
    # Via docker-compose (recommended)
    docker-compose run --rm adaptive-engine python /app/seed.py

    # Locally (with DATABASE_URL set)
    python database/seed.py

Design:
  - 15 technical questions (data_structures, algorithms, system_design, databases, os_concepts)
  - 15 behavioural questions (leadership, teamwork, conflict, growth, communication)
  - 10 HR questions (salary, culture, career_goals, general)
  - Spread across difficulty 1-5
  - Uses INSERT ... ON CONFLICT DO NOTHING — safe to re-run
"""
import asyncio
import os
import sys
from pathlib import Path

import asyncpg

# ── Question Data ──────────────────────────────────────────────────────────

QUESTIONS = [
    # ── Technical: Data Structures ──────────────────────────────────────
    {
        "text": "What is the difference between a stack and a queue? Give a real-world example of each.",
        "type": "technical",
        "difficulty": 1,
        "category": "data_structures",
        "key_concepts": ["stack", "queue", "LIFO", "FIFO"],
        "created_by": "system",
    },
    {
        "text": "Explain how a hash table works, including how collisions are handled.",
        "type": "technical",
        "difficulty": 2,
        "category": "data_structures",
        "key_concepts": ["hash_function", "collision", "chaining", "open_addressing"],
        "created_by": "system",
    },
    {
        "text": "What are the differences between a binary tree and a binary search tree? What is the time complexity of search in each?",
        "type": "technical",
        "difficulty": 3,
        "category": "data_structures",
        "key_concepts": ["binary_tree", "BST", "time_complexity", "O(log n)"],
        "created_by": "system",
    },

    # ── Technical: Algorithms ────────────────────────────────────────────
    {
        "text": "Explain the concept of Big-O notation. What does O(n log n) mean?",
        "type": "technical",
        "difficulty": 1,
        "category": "algorithms",
        "key_concepts": ["Big-O", "time_complexity", "space_complexity"],
        "created_by": "system",
    },
    {
        "text": "Walk me through how merge sort works. What is its time and space complexity?",
        "type": "technical",
        "difficulty": 2,
        "category": "algorithms",
        "key_concepts": ["merge_sort", "divide_and_conquer", "O(n log n)", "O(n) space"],
        "created_by": "system",
    },
    {
        "text": "Describe Dijkstra's algorithm. In what situations would you prefer Bellman-Ford instead?",
        "type": "technical",
        "difficulty": 4,
        "category": "algorithms",
        "key_concepts": ["Dijkstra", "Bellman-Ford", "shortest_path", "negative_weights"],
        "created_by": "system",
    },
    {
        "text": "Explain dynamic programming. Give an example of a problem that benefits from memoisation.",
        "type": "technical",
        "difficulty": 4,
        "category": "algorithms",
        "key_concepts": ["dynamic_programming", "memoisation", "overlapping_subproblems"],
        "created_by": "system",
    },

    # ── Technical: System Design ─────────────────────────────────────────
    {
        "text": "How would you design a URL shortener service like bit.ly? Describe the key components.",
        "type": "technical",
        "difficulty": 3,
        "category": "system_design",
        "key_concepts": ["scalability", "hashing", "database_choice", "caching"],
        "created_by": "system",
    },
    {
        "text": "Explain the CAP theorem. Can you give an example of a system that chooses CP and one that chooses AP?",
        "type": "technical",
        "difficulty": 4,
        "category": "system_design",
        "key_concepts": ["CAP_theorem", "consistency", "availability", "partition_tolerance"],
        "created_by": "system",
    },
    {
        "text": "Design a rate limiter for an API that allows 100 requests per minute per user. What algorithms would you consider?",
        "type": "technical",
        "difficulty": 5,
        "category": "system_design",
        "key_concepts": ["rate_limiting", "token_bucket", "sliding_window", "Redis"],
        "created_by": "system",
    },

    # ── Technical: Databases ─────────────────────────────────────────────
    {
        "text": "What is the difference between SQL and NoSQL databases? When would you choose one over the other?",
        "type": "technical",
        "difficulty": 2,
        "category": "databases",
        "key_concepts": ["SQL", "NoSQL", "ACID", "schema_flexibility"],
        "created_by": "system",
    },
    {
        "text": "Explain database indexing. What are the trade-offs of adding many indexes to a table?",
        "type": "technical",
        "difficulty": 3,
        "category": "databases",
        "key_concepts": ["indexing", "B-tree", "write_overhead", "query_performance"],
        "created_by": "system",
    },

    # ── Technical: OS / Concurrency ──────────────────────────────────────
    {
        "text": "What is the difference between a process and a thread? What is a race condition?",
        "type": "technical",
        "difficulty": 2,
        "category": "os_concepts",
        "key_concepts": ["process", "thread", "race_condition", "shared_memory"],
        "created_by": "system",
    },
    {
        "text": "Explain deadlock. What are the four Coffman conditions and how can deadlock be prevented?",
        "type": "technical",
        "difficulty": 4,
        "category": "os_concepts",
        "key_concepts": ["deadlock", "Coffman_conditions", "prevention", "avoidance"],
        "created_by": "system",
    },
    {
        "text": "What is virtual memory? How does paging work and what is a page fault?",
        "type": "technical",
        "difficulty": 3,
        "category": "os_concepts",
        "key_concepts": ["virtual_memory", "paging", "page_fault", "TLB"],
        "created_by": "system",
    },

    # ── Behavioural: Leadership ──────────────────────────────────────────
    {
        "text": "Tell me about a time you led a project or took initiative without being asked. What was the outcome?",
        "type": "behavioural",
        "difficulty": 2,
        "category": "leadership",
        "key_concepts": ["initiative", "ownership", "STAR_method"],
        "created_by": "system",
    },
    {
        "text": "Describe a situation where you had to make a difficult decision with incomplete information. How did you approach it?",
        "type": "behavioural",
        "difficulty": 3,
        "category": "leadership",
        "key_concepts": ["decision_making", "ambiguity", "risk_assessment"],
        "created_by": "system",
    },
    {
        "text": "Give an example of a time you mentored or coached a colleague. What techniques did you use?",
        "type": "behavioural",
        "difficulty": 3,
        "category": "leadership",
        "key_concepts": ["mentoring", "coaching", "feedback", "growth_mindset"],
        "created_by": "system",
    },

    # ── Behavioural: Teamwork ────────────────────────────────────────────
    {
        "text": "Tell me about a successful team project you were part of. What was your specific contribution?",
        "type": "behavioural",
        "difficulty": 1,
        "category": "teamwork",
        "key_concepts": ["collaboration", "roles", "contribution", "STAR_method"],
        "created_by": "system",
    },
    {
        "text": "Describe a time when your team disagreed on a technical approach. How did you reach a consensus?",
        "type": "behavioural",
        "difficulty": 3,
        "category": "teamwork",
        "key_concepts": ["disagreement", "consensus", "technical_discussion", "compromise"],
        "created_by": "system",
    },

    # ── Behavioural: Conflict ─────────────────────────────────────────────
    {
        "text": "Tell me about a time you had a conflict with a coworker. How did you resolve it?",
        "type": "behavioural",
        "difficulty": 2,
        "category": "conflict",
        "key_concepts": ["conflict_resolution", "communication", "empathy"],
        "created_by": "system",
    },
    {
        "text": "Describe a time when you disagreed with your manager's decision. What did you do?",
        "type": "behavioural",
        "difficulty": 4,
        "category": "conflict",
        "key_concepts": ["upward_feedback", "professionalism", "advocacy"],
        "created_by": "system",
    },

    # ── Behavioural: Growth / Failure ────────────────────────────────────
    {
        "text": "Tell me about your biggest professional failure. What did you learn from it?",
        "type": "behavioural",
        "difficulty": 3,
        "category": "growth",
        "key_concepts": ["self_awareness", "accountability", "learning"],
        "created_by": "system",
    },
    {
        "text": "Give an example of a time you received constructive criticism. How did you respond?",
        "type": "behavioural",
        "difficulty": 2,
        "category": "growth",
        "key_concepts": ["feedback_reception", "adaptability", "growth_mindset"],
        "created_by": "system",
    },
    {
        "text": "Describe the most challenging technical problem you have solved. Walk me through your debugging process.",
        "type": "behavioural",
        "difficulty": 4,
        "category": "growth",
        "key_concepts": ["problem_solving", "debugging", "persistence"],
        "created_by": "system",
    },

    # ── Behavioural: Communication ───────────────────────────────────────
    {
        "text": "Tell me about a time you had to explain a complex technical concept to a non-technical stakeholder.",
        "type": "behavioural",
        "difficulty": 2,
        "category": "communication",
        "key_concepts": ["technical_communication", "simplification", "audience_awareness"],
        "created_by": "system",
    },
    {
        "text": "Describe a time when you had to present your work to senior leadership. How did you prepare?",
        "type": "behavioural",
        "difficulty": 3,
        "category": "communication",
        "key_concepts": ["presentation", "executive_communication", "preparation"],
        "created_by": "system",
    },
    {
        "text": "Tell me about a time when written communication (email, docs, specs) was critical to a project's success.",
        "type": "behavioural",
        "difficulty": 2,
        "category": "communication",
        "key_concepts": ["written_communication", "documentation", "clarity"],
        "created_by": "system",
    },
    {
        "text": "Give an example of a time you had to influence someone without direct authority.",
        "type": "behavioural",
        "difficulty": 4,
        "category": "communication",
        "key_concepts": ["influence", "persuasion", "stakeholder_management"],
        "created_by": "system",
    },
    {
        "text": "Describe a time you had to manage multiple competing priorities simultaneously. How did you decide what to work on first?",
        "type": "behavioural",
        "difficulty": 3,
        "category": "leadership",
        "key_concepts": ["prioritisation", "time_management", "trade_offs"],
        "created_by": "system",
    },

    # ── HR: General / Motivation ─────────────────────────────────────────
    {
        "text": "Tell me about yourself and why you are applying for this role.",
        "type": "hr",
        "difficulty": 1,
        "category": "general",
        "key_concepts": ["self_introduction", "motivation", "fit"],
        "created_by": "system",
    },
    {
        "text": "What are your greatest strengths? Give a specific example for each one you mention.",
        "type": "hr",
        "difficulty": 1,
        "category": "general",
        "key_concepts": ["strengths", "self_awareness", "examples"],
        "created_by": "system",
    },
    {
        "text": "What is your greatest weakness? How are you actively working to improve it?",
        "type": "hr",
        "difficulty": 2,
        "category": "general",
        "key_concepts": ["self_awareness", "growth_mindset", "honesty"],
        "created_by": "system",
    },
    {
        "text": "Why do you want to leave your current position?",
        "type": "hr",
        "difficulty": 2,
        "category": "general",
        "key_concepts": ["professionalism", "motivation", "tact"],
        "created_by": "system",
    },

    # ── HR: Career Goals ─────────────────────────────────────────────────
    {
        "text": "Where do you see yourself in five years? How does this role fit into that vision?",
        "type": "hr",
        "difficulty": 2,
        "category": "career_goals",
        "key_concepts": ["career_planning", "ambition", "alignment"],
        "created_by": "system",
    },
    {
        "text": "What does your ideal work environment look like? How do you prefer to be managed?",
        "type": "hr",
        "difficulty": 2,
        "category": "culture",
        "key_concepts": ["work_style", "management_style", "culture_fit"],
        "created_by": "system",
    },

    # ── HR: Salary / Logistics ───────────────────────────────────────────
    {
        "text": "What are your salary expectations for this role?",
        "type": "hr",
        "difficulty": 3,
        "category": "salary",
        "key_concepts": ["salary_negotiation", "market_research", "anchoring"],
        "created_by": "system",
    },
    {
        "text": "Do you have any other offers currently? What is your decision timeline?",
        "type": "hr",
        "difficulty": 3,
        "category": "salary",
        "key_concepts": ["negotiation", "leverage", "transparency"],
        "created_by": "system",
    },

    # ── HR: Culture Fit ──────────────────────────────────────────────────
    {
        "text": "What do you know about our company, and why do you want to work here specifically?",
        "type": "hr",
        "difficulty": 2,
        "category": "culture",
        "key_concepts": ["research", "motivation", "company_values"],
        "created_by": "system",
    },
    {
        "text": "How do you stay current with developments in your field? Give a recent example of something you learned.",
        "type": "hr",
        "difficulty": 2,
        "category": "career_goals",
        "key_concepts": ["continuous_learning", "curiosity", "self_development"],
        "created_by": "system",
    },
]


# ── Seed Function ──────────────────────────────────────────────────────────

async def seed(database_url: str) -> None:
    """Connect to the DB and insert all questions safely."""
    # asyncpg uses plain postgresql:// (not postgresql+asyncpg://)
    url = database_url.replace("postgresql+asyncpg://", "postgresql://")

    conn = await asyncpg.connect(url)
    try:
        inserted = 0
        skipped = 0

        for q in QUESTIONS:
            import json
            result = await conn.execute(
                """
                INSERT INTO questions
                    (text, type, difficulty, category, key_concepts, created_by)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                ON CONFLICT DO NOTHING
                """,
                q["text"],
                q["type"],
                q["difficulty"],
                q["category"],
                json.dumps(q["key_concepts"]),
                q["created_by"],
            )
            # asyncpg returns "INSERT 0 N" — parse N
            count = int(result.split()[-1])
            if count > 0:
                inserted += 1
            else:
                skipped += 1

        print(f"✓ Seed complete: {inserted} inserted, {skipped} already existed.")

        # Print summary by type
        rows = await conn.fetch(
            "SELECT type, COUNT(*) as n FROM questions WHERE is_active = TRUE GROUP BY type ORDER BY type"
        )
        print("\nQuestion bank summary:")
        for row in rows:
            print(f"  {row['type']:15s} → {row['n']} questions")

    finally:
        await conn.close()


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable is not set.")
        print("Set it in .env or export it before running this script.")
        sys.exit(1)

    asyncio.run(seed(db_url))
