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
genai.configure(api_key="YOUR_GEMINI_API_KEY")

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
    tts = gTTS(text, lang="en")  # Change language if needed
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
        return recognizer.recognize_google(audio, language="en")
    except sr.UnknownValueError:
        return "Could not recognize speech."
    except sr.RequestError:
        return "Could not request speech recognition results."

# Gemini Assistant Function with System Prompt
def get_gemini_response(prompt, system_prompt, screenshot=None):
    model = genai.GenerativeModel("gemini-2.0-flash")  # Default model
    image_data = {"mime_type": "image/jpeg", "data": screenshot} if screenshot else None
    
    full_prompt = f"{system_prompt}\nUser: {prompt}"
    
    response = model.generate_content([full_prompt, image_data] if screenshot else [full_prompt])
    
    return response.text.strip() if response and response.text else "I couldn't understand."

# Streamlit UI
st.title("💬 Gen AI Voice Assistant with System Prompts")
st.write("Speak, type, or select a predefined prompt!")

# Sidebar Screenshot Option
screenshot_enabled = st.sidebar.checkbox("Include Screenshot?", value=False)

# System Prompt Selection
st.sidebar.subheader("🛠️ System Prompt (AI Personality)")
system_prompt_options = {
    "Default AI": "You are a helpful AI assistant.",
    "Concise AI": "Answer in a brief and direct manner.",
    "Friendly AI": "Be cheerful and engaging in your responses.",
    "Tech Support": "Provide technical assistance for software and hardware issues.",
    "Teacher": "Explain concepts in simple terms as if teaching a beginner.",
}
selected_system_prompt = st.sidebar.selectbox("Choose AI Personality:", list(system_prompt_options.keys()))
system_prompt = system_prompt_options[selected_system_prompt]

# Predefined Prompt Suggestions
prompt_options = [
    "What's the latest tech news?",
    "Summarize this webpage for me.",
    "Tell me a fun fact.",
    "How do I fix a slow computer?",
    "Explain quantum computing in simple terms.",
    "Translate 'Hello, how are you?' into German.",
]
selected_prompt = st.selectbox("💡 Choose a question:", ["Select a prompt..."] + prompt_options)

# Text Input Box
user_input = st.text_input("Or type your question here:")

# Voice Input Button
if st.button("🎤 Speak Now"):
    voice_input = speech_to_text()  # Convert speech to text
    if voice_input:
        st.session_state["user_input"] = voice_input  # Store voice input
        st.write("**You said:**", voice_input)
    else:
        st.warning("Speech not recognized. Try again.")

# Store text input or selected prompt in session state
if selected_prompt != "Select a prompt...":
    st.session_state["user_input"] = selected_prompt
elif user_input:
    st.session_state["user_input"] = user_input

# Gemini AI Response
if st.button("Ask Bot"):
    final_input = st.session_state.get("user_input", "")  # Get stored input (voice, text, or prompt)

    if final_input:
        st.write(f"📡 Sending request to Bot using **{selected_system_prompt}** mode...")
        screenshot_data = capture_screenshot() if screenshot_enabled else None
        response_text = get_gemini_response(final_input, system_prompt, screenshot_data)

        st.write("**🤖 Bot's Response:**", response_text)
        audio_bytes = text_to_speech(response_text)
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.warning("Please enter text, select a prompt, or use voice input.")
