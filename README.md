# 💬 AI Co-pilot Voice Assistant with Screen Capture Feature

This is a **Streamlit-powered AI assistant** that integrates **Google Gemini AI** with voice and text input. It allows users to **speak or type queries**, captures optional **screenshots**, and provides **text and voice responses** using Google Text-to-Speech (gTTS).

---

## 🚀 Features
✅ **Voice & Text Input** - Ask questions by speaking or typing.  
✅ **Gemini AI Integration** - Uses **Gemini 2.0 Flash** for AI responses.  
✅ **Screenshot Capture** - Optionally send a screenshot with your query.  
✅ **Text-to-Speech (TTS)** - Gemini’s response is read aloud.  
✅ **Streamlit UI** - Simple, interactive, and user-friendly interface.

---

## 📌 Installation
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/kushagra2503/Ai-Co-pilot-
cd gemini-ai-voice-assistant
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Set Up Google Gemini API Key**
Replace `YOUR_GEMINI_API_KEY` in the code with your **Google Gemini API Key**.

```python
genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

---

## 🏃‍♂️ Running the App
```bash
streamlit run app.py
```

---

## 🛠️ Requirements
- Python 3.8+
- Streamlit
- Google Generative AI SDK
- SpeechRecognition
- PyAudio (For voice input)
- PyAutoGUI (For screenshots)
- OpenCV
- gTTS (Google Text-to-Speech)

To install missing dependencies:
```bash
pip install streamlit google-generativeai SpeechRecognition PyAudio pyautogui opencv-python gtts numpy
```

---

## 🎤 How to Use
1. **Start the Streamlit app** with `streamlit run app.py`.
2. **Enter a question in the text box** or click **'Speak Now'** to use voice input.
3. **Enable 'Include Screenshot' (Optional)** in the sidebar.
4. **Click 'Ask Bot'** to get a response.
5. **Listen to the response** using built-in **text-to-speech**.

---

## 📸 Screenshots (Optional Feature)
- If enabled, **captures your screen** and sends it along with your query.
- Helps Gemini provide more contextual answers.

---

## 🤖 Future Improvements
🔹 Add support for multiple AI models.  
🔹 Improve accuracy of speech recognition.  
🔹 Implement real-time chat with Gemini.  
🔹 Add improvents for deploying it in a cloud platform

---

## 📜 License
This project is licensed under the **GNU License**.

---

### 🎯 **Contributions & Feedback**
If you’d like to **contribute** or have **suggestions**, feel free to open an **issue** or **pull request** on GitHub!

🚀 Happy Coding! 😊

