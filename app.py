import streamlit as st
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech
import torch
import numpy as np
import io
import soundfile as sf

# Model Description
model_description = """
Image Description and text-to-speech model.




"""

@st.cache_resource
def initialize_image_captioning():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def initialize_speech_synthesis():
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    return processor, model

def generate_caption(processor, model, image):
    # Convert the Streamlit camera input to a PIL image
    image = Image.open(image)  # Convert the uploaded file to a PIL Image
    
    # Process the image using the BLIP processor
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    
    # Decode the generated output into a caption
    output_caption = processor.decode(out[0], skip_special_tokens=True)
    return output_caption

def generate_speech(processor, model, caption):
    inputs = processor(text=caption, return_tensors="pt")

    # Generate a dummy speaker embedding of size (1, 768)
    dummy_speaker_embedding = torch.randn(1, 768).to(inputs["input_ids"].device)

    # Generate speech from the caption
    with torch.no_grad():
        speech = model.generate(input_ids=inputs["input_ids"], speaker_embeddings=dummy_speaker_embedding)

    # Process the speech output
    audio_np = speech.squeeze().cpu().numpy()
    audio_np = audio_np / np.max(np.abs(audio_np))  # Normalize audio

    # Save to BytesIO instead of a file
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_np, 22050)  # Save to BytesIO
    audio_buffer.seek(0)

    return audio_buffer

def main():
    st.set_page_config(
        page_title="Image-to-Speech",
        page_icon="📸",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by Alim Tleuliyev")
    st.sidebar.markdown("Contact: [alim.tleuliyev@nu.edu.kz](mailto:alim.tleuliyev@nu.edu.kz)")
    st.sidebar.markdown("GitHub: [Repo](https://github.com/AlimTleuliyev/image-to-audio)")

    st.markdown(
        """
        <style>
        .container {
            max-width: 800px;
        }
        .title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .description {
            margin-bottom: 30px;
        }
        .instructions {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.markdown("<div class='title'>Image Captioning and Text-to-Speech</div>", unsafe_allow_html=True)

    # Model Description
    st.markdown("<div class='description'>" + model_description + "</div>", unsafe_allow_html=True)

    # Instructions
    with st.expander("Instructions"):
        st.markdown("1. Click on 'Take a Photo' to capture a real-time image.")
        st.markdown("2. Click the 'Generate Caption and Speech' button.")
        st.markdown("3. The generated caption will be displayed, and you can click the 'Speak Caption' button to hear it.")

    # Capture image from the webcam
    image = st.camera_input("Take a Photo")

    # Generate caption and play sound button
    if image is not None:
        # Display the captured image
        st.image(image, caption='Captured Image', use_column_width=True)

        # Initialize image captioning models
        caption_processor, caption_model = initialize_image_captioning()

        # Generate caption
        with st.spinner("Generating Caption..."):
            output_caption = generate_caption(caption_processor, caption_model, image)

        # Display the caption
        st.subheader("Caption:")
        st.write(output_caption)

        # Add a button to trigger the Web Speech API
        if st.button("Speak Caption"):
            st.write(
                f"""
                <script>
                    var msg = new SpeechSynthesisUtterance("{output_caption}");
                    window.speechSynthesis.speak(msg);
                </script>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
