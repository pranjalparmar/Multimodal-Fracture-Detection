
# Bone Fracture Classifier + MedGemma Chat

A small Streamlit app for X-ray analysis.
It combines a CNN fracture classifier with MedGemma-based image description so the result feels more readable than a plain label.

## What it does

- Upload an X-ray image
- Predict fractured vs non-fractured with a CNN
- Generate a short natural-language description with MedGemma
- Show a combined interpretation in a chat-style UI

## Stack

- Python
- Streamlit
- CNN classifier
- MedGemma

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

This project is a prototype for experimentation and thesis work.
It is not a medical device and should not be used for diagnosis.

## Screenshots

### Fracture example



### Non-fracture example