#!/usr/bin/env bash
# =============================================================================
# run_all_evals.sh  —  Run all evaluation scripts and print Table II summary
#
# Replicates the paper's reported metrics:
#   ┌─────────────────────────────────────┬────────────┬────────────┬────────┐
#   │ Metric                              │  Target    │  Paper     │ Status │
#   ├─────────────────────────────────────┼────────────┼────────────┼────────┤
#   │ Answer eval accuracy                │  ≥ 94.0%   │  94.2%     │   ✓    │
#   │ Answer eval macro F1                │  ≥ 94.0%   │  94.2%     │   ✓    │
#   │ Sentiment classification accuracy   │  ≥ 91.0%   │  91.8%     │   ✓    │
#   │ ASR WER — clear speech              │  ≤  4.0%   │   3.2%     │   ✓    │
#   │ ASR WER — overall                   │  ≤  6.0%   │   5.3%     │   ✓    │
#   └─────────────────────────────────────┴────────────┴────────────┴────────┘
#
# Usage:
#   # Synthetic mode (no model weights or audio needed):
#   bash scripts/run_all_evals.sh --synthetic
#
#   # Real model mode (requires trained weights + test data):
#   bash scripts/run_all_evals.sh \
#     --nlp_test    data/nlp_test.jsonl \
#     --nlp_model   models/bert_scorer.pt \
#     --sent_test   data/sentiment_test.jsonl \
#     --sent_model  models/confidence_lstm.pt \
#     --sent_vocab  models/vocab.json \
#     --asr_manifest data/asr_test/manifest.jsonl \
#     --asr_dir      data/asr_test/
#
#   # Mixed: synthetic NLP + real ASR:
#   bash scripts/run_all_evals.sh --synthetic \
#     --asr_manifest data/asr_test/manifest.jsonl \
#     --asr_dir      data/asr_test/
#
# Options:
#   --synthetic          Use synthetic data for all three scripts
#   --nlp_n N            Synthetic NLP sample count (default 1000)
#   --sent_n N           Synthetic sentiment sample count (default 2000)
#   --asr_n N            Synthetic ASR utterance count (default 100)
#   --seed N             Random seed for reproducibility (default 42)
#   --output_dir DIR     Directory for JSON result files (default: eval_results/)
#   --verbose            Print confusion matrices
# =============================================================================

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BOLD='\033[1m';   RESET='\033[0m'

pass() { echo -e "${GREEN}✓ PASS${RESET}"; }
fail() { echo -e "${RED}✗ FAIL${RESET}"; }
info() { echo -e "${BOLD}$*${RESET}"; }

# ── Defaults ───────────────────────────────────────────────────────────────────
SYNTHETIC=false
NLP_TEST=""; NLP_MODEL="models/bert_scorer.pt"
SENT_TEST=""; SENT_MODEL="models/confidence_lstm.pt"; SENT_VOCAB="models/vocab.json"
ASR_MANIFEST=""; ASR_DIR="."
NLP_N=1000; SENT_N=2000; ASR_N=100; SEED=42
OUTPUT_DIR="eval_results"; VERBOSE=""

# ── Arg parsing ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --synthetic)      SYNTHETIC=true ;;
        --nlp_test)       NLP_TEST="$2";       shift ;;
        --nlp_model)      NLP_MODEL="$2";      shift ;;
        --sent_test)      SENT_TEST="$2";      shift ;;
        --sent_model)     SENT_MODEL="$2";     shift ;;
        --sent_vocab)     SENT_VOCAB="$2";     shift ;;
        --asr_manifest)   ASR_MANIFEST="$2";   shift ;;
        --asr_dir)        ASR_DIR="$2";        shift ;;
        --nlp_n)          NLP_N="$2";          shift ;;
        --sent_n)         SENT_N="$2";         shift ;;
        --asr_n)          ASR_N="$2";          shift ;;
        --seed)           SEED="$2";           shift ;;
        --output_dir)     OUTPUT_DIR="$2";     shift ;;
        --verbose)        VERBOSE="--verbose"  ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ── Locate project root ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo ""
info "============================================================"
info "  Interview Bot — Evaluation Suite"
info "  $(date)"
info "  Mode: $([ "$SYNTHETIC" = true ] && echo 'SYNTHETIC' || echo 'REAL MODEL')"
info "============================================================"
echo ""

OVERALL_PASS=true

# ── Helper: run a script and extract the pass status ──────────────────────────
run_eval() {
    local label="$1"; local script="$2"; local args="$3"
    local outfile="$OUTPUT_DIR/${label}_${TIMESTAMP}.json"

    echo ""
    info "── ${label} ──────────────────────────────────────────────"
    # Run; capture exit code without aborting the whole script
    if python3 $script $args --output_json "$outfile"; then
        echo -e "  Output saved → ${outfile}"
    else
        echo -e "${RED}  Script exited with error${RESET}"
        OVERALL_PASS=false
    fi
}

# ── 1. NLP evaluation ─────────────────────────────────────────────────────────
if [ "$SYNTHETIC" = true ] || [ -n "$NLP_TEST" ]; then
    if [ "$SYNTHETIC" = true ]; then
        NLP_ARGS="--synthetic --n ${NLP_N} --seed ${SEED} ${VERBOSE}"
    else
        NLP_ARGS="--test_path ${NLP_TEST} --model_path ${NLP_MODEL} ${VERBOSE}"
    fi
    run_eval "nlp" "nlp_service/evaluate.py" "$NLP_ARGS"
    NLP_RESULT="$OUTPUT_DIR/nlp_${TIMESTAMP}.json"
else
    echo -e "${YELLOW}  Skipping NLP eval (no --nlp_test or --synthetic)${RESET}"
    NLP_RESULT=""
fi

# ── 2. Sentiment evaluation ───────────────────────────────────────────────────
if [ "$SYNTHETIC" = true ] || [ -n "$SENT_TEST" ]; then
    if [ "$SYNTHETIC" = true ]; then
        SENT_ARGS="--synthetic --n ${SENT_N} --seed ${SEED} ${VERBOSE}"
    else
        SENT_ARGS="--test_path ${SENT_TEST} --model_path ${SENT_MODEL} \
                   --vocab_path ${SENT_VOCAB} ${VERBOSE}"
    fi
    run_eval "sentiment" "sentiment_service/evaluate_sentiment.py" "$SENT_ARGS"
    SENT_RESULT="$OUTPUT_DIR/sentiment_${TIMESTAMP}.json"
else
    echo -e "${YELLOW}  Skipping Sentiment eval (no --sent_test or --synthetic)${RESET}"
    SENT_RESULT=""
fi

# ── 3. ASR evaluation ─────────────────────────────────────────────────────────
if [ "$SYNTHETIC" = true ] || [ -n "$ASR_MANIFEST" ]; then
    if [ "$SYNTHETIC" = true ]; then
        ASR_ARGS="--synthetic --n ${ASR_N} --seed ${SEED}"
    else
        ASR_ARGS="--manifest ${ASR_MANIFEST} --data_dir ${ASR_DIR}"
    fi
    run_eval "asr" "asr_service/evaluate_asr.py" "$ASR_ARGS"
    ASR_RESULT="$OUTPUT_DIR/asr_${TIMESTAMP}.json"
else
    echo -e "${YELLOW}  Skipping ASR eval (no --asr_manifest or --synthetic)${RESET}"
    ASR_RESULT=""
fi

# ── 4. Summary table (Table II) ───────────────────────────────────────────────
echo ""
info "============================================================"
info "  TABLE II — PERFORMANCE METRICS SUMMARY"
info "============================================================"

python3 - <<PYEOF
import json, sys, os

def load(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

nlp  = load("${NLP_RESULT}")
sent = load("${SENT_RESULT}")
asr  = load("${ASR_RESULT}")

GREEN = "\033[0;32m"; RED = "\033[0;31m"; RESET = "\033[0m"; BOLD = "\033[1m"

def verdict(passed):
    return f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"

def row(metric, target, paper, actual, passed):
    act_str = f"{actual:.1f}%" if actual is not None else "  N/A  "
    v       = verdict(passed) if actual is not None else f"  —  "
    print(f"  {metric:<38s} {target:<12s} {paper:<10s} {act_str:<10s} {v}")

print(f"\n  {BOLD}{'Metric':<38s} {'Target':<12s} {'Paper':<10s} {'Actual':<10s} {'Status'}{RESET}")
print(f"  {'-'*38}   {'-'*10}   {'-'*8}   {'-'*8}   {'-'*8}")

# NLP
if nlp:
    nlp_acc = nlp.get("macro", {}).get("accuracy")
    nlp_f1  = nlp.get("macro", {}).get("f1")
    row("Answer eval accuracy",     "≥ 94.0%", "94.2%", nlp_acc,
        nlp.get("targets_met", {}).get("accuracy", False))
    row("Answer eval macro F1",     "≥ 94.0%", "94.2%", nlp_f1,
        nlp.get("targets_met", {}).get("f1",       False))
else:
    row("Answer eval accuracy",     "≥ 94.0%", "94.2%", None, False)
    row("Answer eval macro F1",     "≥ 94.0%", "94.2%", None, False)

# Sentiment
if sent:
    s_acc = sent.get("accuracy")
    row("Sentiment accuracy",       "≥ 91.0%", "91.8%", s_acc,
        sent.get("targets_met", {}).get("accuracy", False))
else:
    row("Sentiment accuracy",       "≥ 91.0%", "91.8%", None, False)

# ASR
if asr:
    clear_wer   = asr.get("per_type", {}).get("clear",   {}).get("wer_micro")
    overall_wer = asr.get("overall",  {}).get("wer_micro")
    row("ASR WER — clear speech",   "≤  4.0%",  "3.2%", clear_wer,
        asr.get("targets_met", {}).get("wer_clear",   False))
    row("ASR WER — overall",        "≤  6.0%",  "5.3%", overall_wer,
        asr.get("targets_met", {}).get("wer_overall", False))
else:
    row("ASR WER — clear speech",   "≤  4.0%",  "3.2%", None, False)
    row("ASR WER — overall",        "≤  6.0%",  "5.3%", None, False)

print()

# All-pass verdict
results = []
if nlp:
    results += list(nlp.get("targets_met", {}).values())
if sent:
    results += list(sent.get("targets_met", {}).values())
if asr:
    results += list(asr.get("targets_met", {}).values())

if results:
    all_pass = all(results)
    if all_pass:
        print(f"  {GREEN}{BOLD}All targets met — results match paper Table II ✓{RESET}")
    else:
        failed = sum(1 for r in results if not r)
        print(f"  {RED}{BOLD}{failed} target(s) not met.{RESET}")
PYEOF

echo ""
info "============================================================"
info "  Results saved to: ${OUTPUT_DIR}/"
info "============================================================"
echo ""
