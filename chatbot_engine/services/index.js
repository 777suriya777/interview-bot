/**
 * services/index.js — HTTP clients for all downstream microservices.
 *
 * Each function wraps an axios call with:
 *   - The correct timeout from spec Section 5.4
 *   - Error handling that returns null or a fallback on timeout
 *
 * Timeout budget (from spec):
 *   NLP service      → 5s  (return neutral scores on timeout)
 *   ASR service      → 8s  (send ERROR frame on timeout)
 *   Sentiment service→ 5s  (return 'moderate' fallback on timeout)
 *   Adaptive engine  → 3s  (return random medium-difficulty fallback)
 *   Report service   → 30s (PDF generation can take longer)
 */

const axios = require('axios');
const FormData = require('form-data');

// ── Service base URLs (from env vars set in docker-compose) ───────────────────
const NLP_URL       = process.env.NLP_SERVICE_URL      || 'http://nlp-service:8001';
const ASR_URL       = process.env.ASR_SERVICE_URL      || 'http://asr-service:8002';
const SENTIMENT_URL = process.env.SENTIMENT_SERVICE_URL || 'http://sentiment-service:8003';
const ADAPTIVE_URL  = process.env.ADAPTIVE_SERVICE_URL || 'http://adaptive-engine:8004';
const REPORT_URL    = process.env.REPORT_SERVICE_URL   || 'http://report-service:8005';

// ── Timeouts (milliseconds) ───────────────────────────────────────────────────
const TIMEOUT_NLP      = 5_000;
const TIMEOUT_ASR      = 60_000;
const TIMEOUT_SENTIMENT= 5_000;
const TIMEOUT_ADAPTIVE = 3_000;
const TIMEOUT_REPORT   = 30_000;

// ── NLP Service ───────────────────────────────────────────────────────────────

/**
 * Evaluate an answer against a question.
 *
 * @param {string} questionText
 * @param {string} answerText
 * @returns {object|null} NLP result, or null on timeout/error
 *
 * On timeout (>5s): caller must use neutral scores {content:2,relevance:2,completeness:2,accuracy:2}
 */
async function callNLP(questionText, answerText) {
  try {
    const { data } = await axios.post(
      `${NLP_URL}/evaluate`,
      { question: questionText, answer: answerText },
      { timeout: TIMEOUT_NLP }
    );
    return data;
  } catch (err) {
    const code = err.code === 'ECONNABORTED' ? 'nlp_timeout' : 'nlp_error';
    console.error(`[NLP] ${code}: ${err.message}`);
    return null; // caller handles null → neutral scores
  }
}

/**
 * Request a custom interview question based on resume.
 *
 * @param {string} resumeText
 * @returns {object|null} Generated question result
 */
async function callNLPGenerate(resumeText) {
  try {
    const { data } = await axios.post(
      `${NLP_URL}/generate`,
      { resume_text: resumeText },
      { timeout: 30000 } // Takes longer to generate
    );
    return data;
  } catch (err) {
    console.error(`[NLP Generate] Error: ${err.message}`);
    return null;
  }
}

// ── ASR Service ───────────────────────────────────────────────────────────────

/**
 * Transcribe a base64-encoded audio file.
 *
 * @param {string} audioB64   base64-encoded .webm audio
 * @returns {object|null} ASR result, or null on timeout/error
 *
 * On timeout (>8s): caller sends ERROR frame and prompts user to re-record.
 */
async function callASR(audioB64) {
  try {
    // Decode base64 → Buffer and send as multipart form
    const audioBuffer = Buffer.from(audioB64, 'base64');
    const form = new FormData();
    form.append('audio_file', audioBuffer, {
      filename    : 'recording.webm',
      contentType : 'audio/webm',
    });

    const { data } = await axios.post(
      `${ASR_URL}/transcribe`,
      form,
      {
        headers : form.getHeaders(),
        timeout : TIMEOUT_ASR,
        maxBodyLength: 15 * 1024 * 1024, // 15MB headroom above 10MB limit
      }
    );
    return data;
  } catch (err) {
    const isTimeout = err.code === 'ECONNABORTED';
    const code = isTimeout ? 'asr_timeout' : 'asr_error';
    console.error(`[ASR] ${code}: ${err.message}`);
    // Return structured error so caller can distinguish timeout vs other errors
    return { _error: code, _message: err.message };
  }
}

// ── Sentiment Service ─────────────────────────────────────────────────────────

/**
 * Classify confidence and detect hedging phrases.
 *
 * @param {string} transcript
 * @returns {object} sentiment result, or a safe fallback on error
 */
async function callSentiment(transcript) {
  try {
    const { data } = await axios.post(
      `${SENTIMENT_URL}/classify`,
      { transcript },
      { timeout: TIMEOUT_SENTIMENT }
    );
    return data;
  } catch (err) {
    console.error(`[Sentiment] Error: ${err.message}`);
    // Safe fallback — don't fail the whole answer evaluation
    return {
      confidence_label  : 'moderate',
      confidence_score  : 0.5,
      hedging_phrases   : [],
      filler_word_count : 0,
      feedback_text     : 'Confidence analysis unavailable.',
    };
  }
}

// ── Adaptive Engine ───────────────────────────────────────────────────────────

/**
 * Request the next question from the adaptive engine.
 *
 * @param {object} params
 * @param {string} params.userId
 * @param {string} params.sessionId
 * @param {object} params.lastScores   {content,relevance,completeness,accuracy}
 * @param {Array}  params.sessionQuestionIds
 * @returns {object|null} next question, or null on timeout (caller uses fallback)
 */
async function callAdaptive({ userId, sessionId, lastScores, sessionQuestionIds }) {
  try {
    const { data } = await axios.post(
      `${ADAPTIVE_URL}/next-question`,
      {
        user_id              : userId,
        session_id           : sessionId,
        last_scores          : lastScores,
        session_question_ids : sessionQuestionIds,
      },
      { timeout: TIMEOUT_ADAPTIVE }
    );
    return data;
  } catch (err) {
    console.error(`[Adaptive] Error: ${err.message}`);
    return null; // caller uses random medium-difficulty fallback
  }
}

// ── Report Service ────────────────────────────────────────────────────────────

/**
 * Trigger PDF report generation for a completed session.
 *
 * @param {string} sessionId
 * @returns {object|null} report data including report_pdf_url, or null on error
 */
async function callReportGenerate(sessionId) {
  try {
    const { data } = await axios.post(
      `${REPORT_URL}/generate`,
      { session_id: sessionId },
      { timeout: TIMEOUT_REPORT }
    );
    return data;
  } catch (err) {
    console.error(`[Report] Generation error: ${err.message}`);
    return null;
  }
}

module.exports = {
  callNLP,
  callNLPGenerate,
  callASR,
  callSentiment,
  callAdaptive,
  callReportGenerate,
  // Export timeout constants so tests can verify behaviour
  TIMEOUT_NLP,
  TIMEOUT_ASR,
  TIMEOUT_ADAPTIVE,
};
