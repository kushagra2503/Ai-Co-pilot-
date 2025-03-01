import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
import numpy as np
import base64
import pyautogui
import cv2
from gtts import gTTS
from io import BytesIO

# Configure Gemini API
genai.configure(api_key="AIzaSyCFWyeKi7S2gSa7_UVb4JSB9KCeh6OVvP0")

# Function to encode image
def encode_image(image):
    _, buffer = cv2.imencode(".jpeg", image)
    return base64.b64encode(buffer).decode()

# Screenshot Capture Function
def capture_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return encode_image(screenshot)

# Text-to-Speech Function
def text_to_speech(text):
    tts = gTTS(text, lang="de")  # Change "de" to your preferred language
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    return audio_bytes.getvalue()

# Speech-to-Text Function
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio, language="en")  # Change "en" to "de" for German
    except sr.UnknownValueError:
        return "Could not recognize speech."
    except sr.RequestError:
        return "Could not request speech recognition results."

# Gemini Assistant Function
def get_gemini_response(prompt, screenshot=None):
    model = genai.GenerativeModel("gemini-2.0-flash")  # Default model
    image_data = {"mime_type": "image/jpeg", "data": screenshot} if screenshot else None
    response = model.generate_content([prompt, image_data] if screenshot else [prompt])
    return response.text.strip() if response and response.text else "I couldn't understand."

# Streamlit UI
st.title("💬 Gen AI Voice Assistant with Screen Capture Feature")
st.write("Speak a command or type your input below!")

# Text Input Box
user_input = st.text_input("Type your question here:")

# Sidebar Screenshot Option
screenshot_enabled = st.sidebar.checkbox("Include Screenshot?", value=False)

# Voice Input Button
if st.button("🎤 Speak Now"):
    voice_input = speech_to_text()  # Convert speech to text
    if voice_input:
        st.session_state["user_input"] = voice_input  # Store voice input
        st.write("**You said:**", voice_input)
    else:
        st.warning("Speech not recognized. Try again.")

# Store text input in session state if provided
if user_input:
    st.session_state["user_input"] = user_input

# Gemini AI Response
if st.button("Ask Bot"):
    final_input = st.session_state.get("user_input", "")  # Get stored input (voice or text)

    if final_input:
        st.write("📡 Sending request to Bot...")
        screenshot_data = capture_screenshot() if screenshot_enabled else None
        response_text = get_gemini_response(final_input, screenshot_data)

        st.write("**🤖 Bot's Response:**", response_text)
        audio_bytes = text_to_speech(response_text)
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.warning("Please enter text or use voice input.")
