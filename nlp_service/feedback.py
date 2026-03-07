"""
Feedback text generation — pure Python, no ML dependencies.

Extracted from model.py so it can be tested and imported
without requiring torch or transformers.
"""


def build_feedback_summary(scores: dict[str, int]) -> str:
    """
    Generate a short natural-language feedback summary from the four
    dimension scores (0–4 each). Used in the /evaluate API response.
    """
    content      = scores["content"]
    relevance    = scores["relevance"]
    completeness = scores["completeness"]
    accuracy     = scores["accuracy"]
    overall      = (content + relevance + completeness + accuracy) / 4.0

    weakest     = min(scores, key=lambda k: scores[k])
    weakest_val = scores[weakest]

    if overall >= 3.5:
        quality = "Strong answer"
    elif overall >= 2.5:
        quality = "Good answer"
    elif overall >= 1.5:
        quality = "Adequate answer"
    else:
        quality = "Weak answer"

    tips = {
        "content": {
            0: "Your answer lacks technical substance — include specific facts, definitions, or examples.",
            1: "Add more technical detail to demonstrate deeper knowledge.",
            2: "Good technical content; strengthen with concrete examples or edge cases.",
            3: "Strong technical content; consider addressing trade-offs for a perfect score.",
            4: "Excellent technical depth.",
        },
        "relevance": {
            0: "Your answer does not address the question — re-read the question and focus your response.",
            1: "Stay more focused on what was asked; avoid tangents.",
            2: "Mostly on-topic; remove off-topic sections to improve clarity.",
            3: "Very relevant; minor tightening would make it perfect.",
            4: "Perfectly on-topic.",
        },
        "completeness": {
            0: "Very incomplete — cover the main aspects of the question.",
            1: "Address more parts of the question; several key points are missing.",
            2: "Most points covered; expand on the details for a more complete picture.",
            3: "Nearly complete; add one or two finishing details.",
            4: "Comprehensive coverage.",
        },
        "accuracy": {
            0: "Contains significant factual errors — review the core concepts before re-attempting.",
            1: "Some inaccuracies present; double-check your facts and terminology.",
            2: "Mostly accurate with minor errors; verify the specific details you're unsure of.",
            3: "Accurate with very minor imprecision.",
            4: "Fully accurate.",
        },
    }

    if overall >= 3.5:
        if weakest_val < 4:
            tip = tips[weakest][weakest_val]
            return f"{quality}. To reach a perfect score: {tip}"
        return f"{quality}. All dimensions rated excellent — well done."

    tip = tips[weakest][weakest_val]
    return f"{quality} (overall {overall:.1f}/4). Focus on {weakest}: {tip}"
