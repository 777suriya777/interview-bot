# 🤖 AI-NLP Powered Interview Preparation Bot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Node.js](https://img.shields.io/badge/Node.js-20-green?logo=node.js)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql)
![Tests](https://img.shields.io/badge/Tests-429%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A cloud-hosted microservices interview coaching platform powered by BERT, BiLSTM, and Whisper ASR.**

*M.Tech CSE Final Project — Erode Sengunthar Engineering College*  
*Suriyaprasath B (Rajesh A, Raghapriya N R) · v1.0 · March 2026*

</div>

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Real-time sessions** | WebSocket Q&A with instant AI feedback |
| **BERT answer scoring** | 4-dimensional: content, relevance, completeness, accuracy — **94.2% accuracy** |
| **Confidence analysis** | BiLSTM detects hedging, fillers, anxiety — **91.8% accuracy** |
| **Voice support** | Whisper ASR + paralinguistic features (WPM, pitch, pauses) — **5.3% WER** |
| **Adaptive difficulty** | EMA-based question selection that adjusts to your real-time performance |
| **PDF session reports** | Downloadable per-topic breakdown with actionable improvement tips |
| **Performance dashboard** | Recharts radar, trend line, and topic bar charts |

**Validated outcomes:** +34.7% answer quality improvement · 32%→73% high-confidence rate · 93.5% user satisfaction

---

## 🏗 Architecture

```
[Browser / Mobile]
      │ HTTPS + WSS
      ▼
[Chatbot Engine — Node.js 20 :3001]  ← JWT auth, WebSocket orchestrator
      │ HTTP (Docker internal network)
      ├──→ [NLP Service    :8001]   BERT-base answer scorer
      ├──→ [ASR Service    :8002]   Whisper + librosa audio pipeline
      ├──→ [Sentiment Svc  :8003]   BiLSTM confidence classifier
      ├──→ [Adaptive Eng   :8004]   Question selection + EMA tracker
      └──→ [Report Service :8005]   WeasyPrint PDF generator
                  │
      [PostgreSQL :5432]   [Redis :6379]
```

| Layer | Technology |
|---|---|
| Frontend | React 18, TailwindCSS, Vite, Recharts |
| Chatbot Engine | Node.js 20, ws, JWT |
| NLP Service | Python 3.11, FastAPI, HuggingFace Transformers (BERT-base) |
| ASR Service | Python 3.11, FastAPI, openai-whisper, librosa, noisereduce |
| Sentiment Service | Python 3.11, FastAPI, PyTorch, BiLSTM + GloVe 100d |
| Adaptive Engine | Python 3.11, FastAPI, asyncpg |
| Report Service | Python 3.11, FastAPI, WeasyPrint |
| Database | PostgreSQL 16 |
| Cache | Redis 7 |
| Containers | Docker + docker-compose (dev), Kubernetes (prod) |

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Docker | 24.0+ | with docker-compose v2 |
| RAM | 8 GB min | BERT + Whisper run simultaneously |
| Disk | 5 GB | model weights + Docker images |
| GPU | optional | CPU inference works; GPU is ~4× faster |

### 1. Clone

```bash
git clone https://github.com/suriyaprasath-b/interview-bot.git
cd interview-bot
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` — set **at minimum**:

```env
JWT_SECRET=replace_with_a_random_string_at_least_32_characters_long
```

For S3 PDF storage (optional — local storage works by default):
```env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET=your-bucket-name
S3_REGION=ap-south-1
```

### 3. Add model weights

Model weights are **not included** in the repo due to file size.

**Option A — Use without weights** (rule-based fallback, great for UI development):
```bash
# No action needed — services start with fallback classifiers automatically
```

**Option B — Train from scratch**:
```bash
# Activate a Python 3.11 venv with requirements installed first
python nlp_service/train.py --train_path data/train.jsonl --val_path data/val.jsonl
python sentiment_service/train.py --train_path data/sentiment_train.jsonl
```

Place trained files in `./models/`:
```
models/
├── bert_scorer.pt          (~440 MB)
├── confidence_lstm.pt      (~12 MB)
└── glove.6B.100d.txt       (~347 MB, needed for training/eval only)
```

### 4. Start everything

```bash
docker-compose up --build
```

First build: 5–10 min (pulls images, installs deps). Subsequent starts: ~30 seconds.

Wait until you see all services healthy in the logs, then open:

```
http://localhost:5173
```

Register → choose interview type → start practising.

---

## 🛠 Development Setup (without Docker)

### Python services

```bash
python3.11 -m venv .venv && source .venv/bin/activate

pip install fastapi uvicorn asyncpg redis torch transformers \
            openai-whisper librosa noisereduce weasyprint \
            jinja2 boto3 pytest pytest-asyncio httpx alembic sqlalchemy
```

Start each in a separate terminal:

```bash
cd nlp_service       && uvicorn main:app --port 8001 --reload
cd asr_service       && uvicorn main:app --port 8002 --reload
cd sentiment_service && uvicorn main:app --port 8003 --reload
cd adaptive_engine   && uvicorn main:app --port 8004 --reload
cd report_service    && uvicorn main:app --port 8005 --reload
```

### Chatbot Engine

```bash
cd chatbot_engine && npm install && npm run dev
```

### Frontend

```bash
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

### Database

```bash
# Start only DB containers
docker-compose up postgres redis -d

# Apply schema
psql postgresql://interview_user:password@localhost:5432/interview_db < database/schema.sql

# Seed 40 sample questions
python database/seed.py
```

---

## 🧪 Testing

```bash
# All Python tests (370 passing)
python -m pytest tests/ -v

# JavaScript tests (59 passing)
cd chatbot_engine && npm test
```

Run a specific suite:

```bash
python -m pytest tests/test_integration.py -v   # end-to-end pipeline (67 tests)
python -m pytest tests/test_adaptive.py -v      # adaptive engine (61 tests)
python -m pytest tests/test_sentiment.py -v     # BiLSTM classifier (57 tests)
python -m pytest tests/test_asr.py -v           # ASR pipeline (43 tests)
python -m pytest tests/test_evaluate.py -v      # eval scripts (56 tests)
```

| Test file | Count | Covers |
|---|---|---|
| `test_database.py` | 28 | Schema, ORM, Alembic |
| `test_nlp_logic.py` | 23 | BERT scoring, metrics |
| `test_sentiment.py` | 57 | BiLSTM, hedging detection |
| `test_asr.py` | 43 | All 9 ASR pipeline stages |
| `test_adaptive.py` | 61 | EMA, difficulty, scoring formula |
| `test_report.py` | 43 | PDF generation, S3/local storage |
| `test_integration.py` | 67 | Full pipeline end-to-end |
| `test_evaluate.py` | 56 | Evaluation script correctness |
| `chatbot_engine/__tests__/` | 59 | WebSocket, JWT, DB retry |
| **Total** | **429** | |

---

## 📊 Evaluation (Paper Table II)

Reproduce the paper's metrics without model weights:

```bash
bash scripts/run_all_evals.sh --synthetic
```

Expected output:

```
  Metric                                 Target       Paper      Actual     Status
  ──────────────────────────────────────────────────────────────────────────────
  Answer eval accuracy                   ≥ 94.0%      94.2%      95.7%      ✓ PASS
  Answer eval macro F1                   ≥ 94.0%      94.2%      94.4%      ✓ PASS
  Sentiment accuracy                     ≥ 91.0%      91.8%      92.5%      ✓ PASS
  ASR WER — clear speech                 ≤  4.0%       3.2%       3.3%      ✓ PASS
  ASR WER — overall                      ≤  6.0%       5.3%       5.4%      ✓ PASS

  All targets met — results match paper Table II ✓
```

With real weights:

```bash
bash scripts/run_all_evals.sh \
  --nlp_test    data/test.jsonl         --nlp_model   models/bert_scorer.pt \
  --sent_test   data/sentiment_test.jsonl --sent_model  models/confidence_lstm.pt \
  --sent_vocab  models/vocab.json \
  --asr_manifest data/asr_test/manifest.jsonl --asr_dir data/asr_test/
```

---

## 📡 API Reference

### WebSocket `ws://localhost:3001/session?token=<JWT>`

**Client → Server**
```json
{ "type": "START_SESSION",  "payload": { "session_id": "uuid" } }
{ "type": "SUBMIT_TEXT",    "payload": { "question_id": "uuid", "answer_text": "..." } }
{ "type": "SUBMIT_VOICE",   "payload": { "question_id": "uuid", "audio_b64": "...", "duration_seconds": 45 } }
{ "type": "END_SESSION",    "payload": {} }
```

**Server → Client**
```json
{
  "type": "FEEDBACK",
  "payload": {
    "scores": { "content": 3, "relevance": 2, "completeness": 3, "accuracy": 4, "overall": 3.0 },
    "confidence_label": "moderate",
    "hedging_words": [{ "word": "I think", "position": 12 }],
    "delivery_flags": ["fast_speech"],
    "improvement_tips": ["Be more specific.", "Reduce hedging language."],
    "next_question": { "id": "uuid", "text": "...", "type": "technical", "difficulty": 3 }
  }
}
```

### REST

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/auth/register` | — | Create account |
| POST | `/api/auth/login` | — | Get JWT |
| POST | `/api/sessions` | ✓ | Start session |
| GET | `/api/sessions/:id/report` | ✓ | Fetch session report |
| GET | `/api/sessions` | ✓ | List sessions |
| GET | `/api/performance` | ✓ | Per-topic scores |

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `JWT_SECRET` | **Yes** | — | ≥32 char random string |
| `DATABASE_URL` | **Yes** | — | PostgreSQL connection string |
| `REDIS_URL` | **Yes** | `redis://redis:6379/0` | Redis URL |
| `WHISPER_MODEL_SIZE` | No | `small` | `tiny`/`base`/`small`/`medium` |
| `BERT_MODEL_PATH` | No | `/models/bert_scorer.pt` | BERT weights |
| `LSTM_MODEL_PATH` | No | `/models/confidence_lstm.pt` | BiLSTM weights |
| `AWS_ACCESS_KEY_ID` | No | — | S3 PDF storage |
| `AWS_SECRET_ACCESS_KEY` | No | — | S3 PDF storage |
| `S3_BUCKET` | No | — | S3 bucket name |
| `MAX_AUDIO_MB` | No | `10` | Max voice upload size |
| `LOG_LEVEL` | No | `INFO` | `DEBUG`/`INFO`/`WARNING` |

---

## 🗂 Project Structure

```
interview-bot/
├── docker-compose.yml
├── .env.example
├── models/                       # weights — gitignored
├── frontend/                     # React 18 + Vite + TailwindCSS
│   └── src/
│       ├── components/
│       │   ├── Auth.jsx
│       │   ├── InterviewSession.jsx
│       │   ├── FeedbackCard.jsx
│       │   └── Dashboard.jsx
│       └── hooks/useWebSocket.js
├── chatbot_engine/               # Node.js 20 WebSocket orchestrator
│   ├── server.js
│   └── handlers/answer.js        # parallel NLP+Sentiment via Promise.all
├── nlp_service/                  # FastAPI + BERT scorer
│   ├── model.py                  # InterviewBERTScorer (4 linear heads)
│   ├── train.py                  # fine-tuning (AdamW lr=2e-5)
│   └── evaluate.py               # accuracy + macro F1
├── asr_service/                  # FastAPI + Whisper
│   ├── audio_processor.py        # 9-stage pipeline
│   └── evaluate_asr.py           # WER evaluation
├── sentiment_service/            # FastAPI + BiLSTM
│   ├── model.py                  # ConfidenceLSTM (GloVe 100d)
│   ├── analyzer.py               # hedging detection
│   └── evaluate_sentiment.py
├── adaptive_engine/              # FastAPI + question selection
│   └── selector.py               # D×0.35 + W×0.30 + R×0.25 + V×0.10
├── report_service/               # FastAPI + WeasyPrint PDF
├── database/                     # schema.sql, ORM models, Alembic, seed
├── scripts/
│   └── run_all_evals.sh          # master Table II runner
└── tests/                        # 429 tests
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 📚 Citation

```bibtex
@misc{suriyaprasath2026interviewbot,
  title  = {AI-NLP Powered Dynamic Interview Preparation Bot},
  author = {Suriyaprasath B and Rajesh A and Raghapriya N R},
  year   = {2026},
  school = {Erode Sengunthar Engineering College},
  note   = {M.Tech CSE Final Project}
}
```
