from pathlib import Path
from PIL import Image

from src.classifier_inference import load_model, predict_single_image
from src.prompt_builder import build_medgemma_prompt

# --------------------------------------------------
# EDIT THIS PATH TO A REAL LOCAL X-RAY IMAGE
# --------------------------------------------------
TEST_IMAGE_PATH = Path("/Users/pranjalparmar/Documents/git-repos/bone-fracture-classification/FracAtlas/images/Fractured/IMG0000059.jpg")

if not TEST_IMAGE_PATH.exists():
    raise FileNotFoundError(f"Test image not found: {TEST_IMAGE_PATH}")

print("=== LOADING MODEL ===")
assets = load_model()
print("Model loaded successfully.")

print("\n=== LOADING IMAGE ===")
image = Image.open(TEST_IMAGE_PATH)
print(f"Image path: {TEST_IMAGE_PATH}")
print(f"Original size: {image.size}")
print(f"Original mode: {image.mode}")

print("\n=== RUNNING CLASSIFIER ===")
result = predict_single_image(image, assets)
for k, v in result.items():
    print(f"{k}: {v}")

explanation_payload = {
    "image_info": {
        "filename": TEST_IMAGE_PATH.name,
        "original_size": list(image.size),
        "original_mode": image.mode,
    },
    "classifier_output": result,
}

print("\n=== BUILDING PROMPT ===")
prompt = build_medgemma_prompt(explanation_payload)
print(prompt)