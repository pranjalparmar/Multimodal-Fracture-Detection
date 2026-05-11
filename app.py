import os
import tempfile
from datetime import datetime

import streamlit as st
from PIL import Image

from src.classifier_inference import load_model, predict_single_image
from src.medgemma_pipeline import run_medgemma_pipeline


st.set_page_config(
    page_title="Fracture Classifier + MedGemma Chat",
    page_icon="🩻",
    layout="wide",
)


st.title("Bone Fracture Classifier + MedGemma Chat")
st.caption(
    "Conversational X-ray analysis using a CNN fracture decision and MedGemma-supported image description."
)


@st.cache_resource
def get_model_assets():
    return load_model()


def save_uploaded_file_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name


def add_message(role: str, content: str):
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
    )


def reset_chat():
    st.session_state.messages = []
    st.session_state.current_image = None
    st.session_state.current_image_name = None
    st.session_state.analysis_ready = False


if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_image" not in st.session_state:
    st.session_state.current_image = None

if "current_image_name" not in st.session_state:
    st.session_state.current_image_name = None

if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False


model_assets = get_model_assets()


with st.sidebar:
    st.header("Session")
    st.write("Mode: `real`")
    st.write(f"MEDGEMMA_BACKEND: `{os.getenv('MEDGEMMA_BACKEND', 'not_configured')}`")
    st.write(f"MEDGEMMA_MODEL_ID: `{os.getenv('MEDGEMMA_MODEL_ID', 'google/medgemma-4b-it')}`")

    st.markdown("### Upload X-ray")
    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=["png", "jpg", "jpeg"],
        key="chat_uploader",
    )

    if uploaded_file is not None:
        st.session_state.current_image = uploaded_file
        st.session_state.current_image_name = uploaded_file.name
        st.session_state.analysis_ready = True
        st.success(f"Loaded: {uploaded_file.name}")

    if st.button("Reset conversation"):
        reset_chat()
        st.rerun()


if st.session_state.current_image is not None:
    image = Image.open(st.session_state.current_image)
    with st.expander("Current uploaded image", expanded=True):
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.image(image, caption=st.session_state.current_image_name)
        with col_b:
            st.markdown("### Image details")
            st.write(f"Filename: {st.session_state.current_image_name}")
            st.write(f"Size: {image.size[0]} x {image.size[1]}")
            st.write(f"Mode: {image.mode}")
else:
    st.info("Upload a PNG, JPG, or JPEG X-ray from the sidebar to begin.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Ask something like: 'Is it fractured? Which body part is visible?'")

if prompt:
    add_message("user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.current_image is None:
        assistant_text = "Please upload an X-ray image first so I can analyze it."
        add_message("assistant", assistant_text)
        with st.chat_message("assistant"):
            st.markdown(assistant_text)
    else:
        uploaded_file = st.session_state.current_image
        image = Image.open(uploaded_file)
        temp_image_path = None

        try:
            with st.chat_message("assistant"):
                with st.spinner("Running CNN prediction and MedGemma description..."):
                    result = predict_single_image(image, model_assets)
                    temp_image_path = save_uploaded_file_to_temp(uploaded_file)

                    explanation_payload = {
                        "image_path": temp_image_path,
                        "image_info": {
                            "filename": uploaded_file.name,
                            "original_size": list(image.size),
                            "original_mode": image.mode,
                        },
                        "classifier_output": result,
                    }

                    llm_result = run_medgemma_pipeline(
                        explanation_payload=explanation_payload,
                        use_mock=False,
                    )

                    predicted_label = result.get("decision_label", "unknown")
                    prob_fractured = result.get("prob_fractured", None)
                    selected_threshold = result.get("selected_threshold", None)

                    cnn_lines = [
                        "### CNN decision",
                        f"- Predicted label: **{predicted_label}**",
                    ]

                    if prob_fractured is not None:
                        cnn_lines.append(f"- Fracture probability: **{prob_fractured:.4f}**")

                    if selected_threshold is not None:
                        cnn_lines.append(f"- Decision threshold: **{selected_threshold:.4f}**")

                    findings_text = llm_result.get("findings", "").strip()
                    explanation_text = llm_result.get("explanation", "").strip()

                    classifier_context = ""
                    integrated_impression = ""
                    caution_text = ""

                    if explanation_text:
                        if "Classifier context:" in explanation_text:
                            classifier_context = explanation_text.split("Classifier context:")[1]
                            if "Integrated impression:" in classifier_context:
                                classifier_context = classifier_context.split("Integrated impression:")[0].strip()

                        if "Integrated impression:" in explanation_text:
                            integrated_impression = explanation_text.split("Integrated impression:")[1]
                            if "Caution:" in integrated_impression:
                                integrated_impression = integrated_impression.split("Caution:")[0].strip()

                        if "Caution:" in explanation_text:
                            caution_text = explanation_text.split("Caution:")[1].strip()

                    response_parts = [
                        "\n".join(cnn_lines),
                    ]

                    if findings_text:
                        response_parts.append(
                            "### MedGemma description\n"
                            f"{findings_text}"
                        )

                    if classifier_context:
                        response_parts.append(
                            "### Classifier context\n"
                            f"{classifier_context}"
                        )

                    if integrated_impression:
                        response_parts.append(
                            "### Combined interpretation\n"
                            f"{integrated_impression}"
                        )

                    if caution_text:
                        response_parts.append(
                            "### Caution\n"
                            f"{caution_text}"
                        )

                    response_text = "\n\n".join(response_parts).strip()

                    st.markdown(response_text)

            add_message("assistant", response_text)

        except Exception as e:
            error_text = f"Error during prediction/explanation: {e}"
            add_message("assistant", error_text)
            with st.chat_message("assistant"):
                st.error(error_text)

        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)