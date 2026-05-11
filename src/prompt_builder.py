def build_medgemma_prompt(explanation_payload: dict) -> str:
    image_info = explanation_payload.get("image_info", {})
    classifier_output = explanation_payload.get("classifier_output", {})

    filename = image_info.get("filename", "unknown_image")
    original_size = image_info.get("original_size", None)
    original_mode = image_info.get("original_mode", "unknown")

    predicted_label = classifier_output.get("predicted_label", "unknown")
    decision_label = classifier_output.get("decision_label", predicted_label)
    prob_fractured = classifier_output.get("prob_fractured", None)

    size_text = f"{original_size}" if original_size is not None else "unknown"
    prob_text = f"{prob_fractured:.4f}" if prob_fractured is not None else "unknown"

    prompt = f"""
You are a medical X-ray visual description assistant.

Look at the X-ray and answer naturally.

Your job:
- Identify the likely visible body part or region.
- Briefly describe whether the visible appearance may suggest fracture.
- Use the CNN result only as supportive context.
- Do not override the CNN decision.
- Do not provide a definitive diagnosis.
- Do not include disclaimers.
- Keep the answer short and natural.

Output rules:
- Write exactly 2 short sentences.
- First sentence: say which body part or region the X-ray most likely shows.
- Second sentence: briefly describe whether the visible appearance may be consistent with fracture.
- Use cautious wording like "most likely", "appears to show", or "may be consistent with".
- Do not use labels like "Visible region:", "Appearance:", or "Uncertainty:".
- Do not write bullet points.
- Do not write more than 2 sentences.

Image context:
- Filename: {filename}
- Image size: {size_text}
- Image mode: {original_mode}

CNN context:
- Predicted label: {predicted_label}
- Decision label: {decision_label}
- Fracture probability: {prob_text}
""".strip()

    return prompt