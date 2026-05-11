import os
import subprocess
import re

from src.prompt_builder import build_medgemma_prompt


def clean_medgemma_output(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    text = re.sub(r"^[0-9\W_]*(output|response|answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", " ")
    text = text.replace("`", " ")
    text = re.sub(r"\s+", " ", text).strip()

    stop_patterns = [
        r"disclaimer:",
        r"final answer:",
        r"important considerations:",
        r"please consult with",
        r"always consult with",
        r"i am an ai",
        r"medical advice",
    ]

    lowered = text.lower()
    cutoff_positions = []

    for pattern in stop_patterns:
        match = re.search(pattern, lowered)
        if match:
            cutoff_positions.append(match.start())

    if cutoff_positions:
        text = text[:min(cutoff_positions)].strip()

    sentence_candidates = re.split(r'(?<=[.!?])\s+', text)

    cleaned_sentences = []
    seen = set()

    for sent in sentence_candidates:
        sent = sent.strip()
        if not sent:
            continue

        sent = re.sub(r"^[^A-Za-z]+", "", sent).strip()
        sent = re.sub(r"\s+", " ", sent)

        normalized = sent.lower().strip(" .,!?:;")
        if len(normalized) < 10:
            continue
        if normalized in seen:
            continue

        seen.add(normalized)
        cleaned_sentences.append(sent)

        if len(cleaned_sentences) == 2:
            break

    final_text = " ".join(cleaned_sentences).strip()

    if final_text and final_text[-1] not in ".!?":
        final_text += "."

    return final_text

def run_medgemma_cli(prompt: str, explanation_payload: dict, max_tokens: int = 80) -> str:
    image_path = explanation_payload.get("image_path")
    if not image_path:
        raise ValueError(
            "MedGemma CLI backend requires explanation_payload['image_path'] with a real local image file."
        )

    cmd = [
        "medgemma",
        "ask",
        "--no-stream",
        "--max-tokens",
        str(max_tokens),
        "--image",
        image_path,
        prompt,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"MedGemma CLI failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    output_text = result.stdout.strip()
    if not output_text:
        return ""

    return clean_medgemma_output(output_text)


def build_fallback_findings_from_classifier(classifier_output: dict) -> str:
    predicted_label = classifier_output.get("predicted_label", "unknown")

    if predicted_label == "fractured":
        return (
            "This X-ray appears to show a visible bony region. "
            "The visible appearance may be consistent with fracture."
        )

    if predicted_label == "non_fractured":
        return (
            "This X-ray appears to show a visible bony region. "
            "The visible appearance does not clearly show fracture."
        )

    return (
        "This X-ray appears to show a visible bony region. "
        "The visual appearance is limited for automated interpretation."
    )

def is_weak_findings(findings_text: str) -> bool:
    if not findings_text:
        return True

    normalized = findings_text.strip().lower()

    weak_patterns = [
        "unable to provide",
        "cannot provide",
        "medical advice",
        "consult with",
        "qualified healthcare professional",
        "important considerations",
        "clinical context",
        "comparison to previous images",
        "i am an ai",
    ]

    if len(normalized) < 20:
        return True

    return any(pattern in normalized for pattern in weak_patterns)

def generate_real_medgemma_findings(explanation_payload: dict, prompt: str) -> str:
    classifier_output = explanation_payload.get("classifier_output", {})

    findings_text = run_medgemma_cli(
        prompt=prompt,
        explanation_payload=explanation_payload,
        max_tokens=80,
    )

    if is_weak_findings(findings_text):
        findings_text = build_fallback_findings_from_classifier(classifier_output)

    return findings_text


def build_integrated_impression(classifier_output: dict) -> str:
    predicted_label = classifier_output.get("predicted_label", "unknown")
    prob_fractured = classifier_output.get("prob_fractured", None)

    if predicted_label == "fractured":
        if prob_fractured is not None:
            return (
                f"The CNN classifier favors fracture with high confidence "
                f"(fracture probability: {prob_fractured:.4f}). "
                f"Overall, the image may be consistent with fracture, but this is not a definitive diagnosis."
            )
        return (
            "The CNN classifier favors fracture. "
            "Overall, the image may be consistent with fracture, but this is not a definitive diagnosis."
        )

    if predicted_label == "non_fractured":
        if prob_fractured is not None:
            return (
                f"The CNN classifier does not favor fracture "
                f"(fracture probability: {prob_fractured:.4f}). "
                f"Overall, there is no strong automated evidence of fracture, although subtle findings can still be missed."
            )
        return (
            "The CNN classifier does not favor fracture. "
            "Overall, there is no strong automated evidence of fracture, although subtle findings can still be missed."
        )

    return (
        "The CNN classifier did not provide a confident fracture decision. "
        "This result should be interpreted with caution."
    )


def format_final_explanation(explanation_payload: dict, findings_text: str) -> str:
    classifier_output = explanation_payload.get("classifier_output", {})

    predicted_label = classifier_output.get("predicted_label", "unknown")
    decision_label = classifier_output.get("decision_label", predicted_label)
    prob_fractured = classifier_output.get("prob_fractured", None)
    selected_threshold = classifier_output.get("selected_threshold", None)

    if prob_fractured is None:
        classifier_context = (
            f"The CNN classifier labeled this image as '{decision_label}'."
        )
    else:
        classifier_context = (
            f"The CNN classifier labeled this image as '{decision_label}' "
            f"with fracture probability {prob_fractured:.4f}."
        )

    if selected_threshold is not None:
        classifier_context += f" The decision threshold was {selected_threshold:.4f}."

    integrated_impression = build_integrated_impression(classifier_output)

    final_text = f"""Image findings:
{findings_text}

Classifier context:
{classifier_context}

Integrated impression:
{integrated_impression}

Caution:
This is an AI-assisted output. The CNN model provides the fracture decision, while MedGemma provides supportive image description. The result must be reviewed by a qualified clinician.
"""
    return final_text.strip()


def run_medgemma_pipeline(explanation_payload: dict, use_mock: bool = False) -> dict:
    prompt = build_medgemma_prompt(explanation_payload)

    findings_text = generate_real_medgemma_findings(
        explanation_payload=explanation_payload,
        prompt=prompt,
    )

    final_explanation = format_final_explanation(
        explanation_payload=explanation_payload,
        findings_text=findings_text,
    )

    return {
        "mode": "real",
        "backend": os.getenv("MEDGEMMA_BACKEND", "medgemma_cli"),
        "model_id": os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it"),
        "prompt": prompt,
        "findings": findings_text,
        "explanation": final_explanation,
    }