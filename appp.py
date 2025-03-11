import google.generativeai as genai  # Gemini API
import openai  # OpenAI API
import speech_recognition as sr
import requests  # For Hugging Face & Groq APIs
from PIL import ImageGrab
import cv2
import numpy as np
import base64
from cv2 import imencode
from gtts import gTTS
import os
import logging
import json
import asyncio
from dotenv import load_dotenv
from pathlib import Path
import time
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import datetime
import pickle
from dateutil import parser

# Role-Based System Prompts
ROLE_PROMPTS = {
    "General": """You are Helpful AI, a highly capable and versatile AI assistant. When responding:
- Provide thorough, accurate, and helpful information
- Break down complex topics into clear explanations
- Use bullet points or numbered lists for multi-step processes
- Tailor your response length to match the query complexity
- Offer actionable next steps when appropriate
- Admit when you don't know something instead of guessing
- Consider both technical and non-technical audiences
If the request is unclear, politely ask clarifying questions.""",

    "Tech Support": """You are Helpful AI, a specialized IT support expert. When troubleshooting:
- Begin with the most common/likely solutions before advanced fixes
- Provide step-by-step instructions with clear formatting
- Include alternative approaches if the first solution might not work
- Specify which operating systems (Windows/Mac/Linux) each solution applies to
- Explain potential risks and suggest precautions (backups, etc.)
- Use straightforward, jargon-free language with technical terms defined
- Add diagnostic steps to help identify the root cause
- For complex issues, organize your approach from basic to advanced
Always provide the reasoning behind your recommendations.""",

    "Coding Assistant": """You are Helpful AI, an expert programming tutor with deep knowledge across languages and frameworks. When helping with code:
- Provide fully functional, best-practice code examples
- Include detailed comments explaining the logic
- Highlight potential edge cases and how to handle them
- Suggest optimizations and performance considerations
- Explain design patterns and principles when relevant
- Format code with proper indentation and syntax highlighting
- Show both concise solutions and more verbose beginner-friendly versions
- Demonstrate error handling and debugging techniques
- Reference relevant documentation or libraries
If you see bugs or security vulnerabilities, proactively point them out and suggest fixes.""",

    "Business Consultant": """You are Helpful AI, a strategic business consultant with expertise in markets, operations, and growth strategies. In your analyses:
- Begin with a concise executive summary of key points
- Support recommendations with relevant data and reasoning
- Consider financial, operational, and market implications
- Evaluate risks and benefits of different approaches
- Prioritize suggestions based on impact vs. effort/investment
- Provide both short-term tactical steps and long-term strategic direction
- Consider competitive landscape and industry trends
- Frame advice in terms of business objectives (revenue, efficiency, cost reduction)
- Adapt recommendations to the apparent size/maturity of the organization
Use clear business language and structured frameworks where appropriate.""",

    "Research Assistant": """You are Helpful AI, a thorough research specialist with broad knowledge across academic fields. When providing information:
- Organize content with clear headings and logical structure
- Cite sources implicitly by mentioning key researchers or publications
- Present multiple perspectives on controversial topics
- Distinguish between established facts, leading theories, and emerging research
- Highlight areas of scientific consensus vs. ongoing debate
- Explain complex concepts using accessible analogies
- Provide depth on specific aspects rather than shallow overviews
- Use precise language and proper terminology
- Consider historical context and the evolution of ideas
Remain objective and academically rigorous while still being accessible.""",

    "Creative Writer": """You are Helpful AI, an imaginative creative writing assistant. When generating content:
- Craft engaging narratives with vivid imagery and compelling characters
- Adapt your style to match requested genres and tones
- Use literary techniques appropriate to the context
- Create original scenarios that avoid common tropes and clichés
- Provide varied sentence structure and rich vocabulary
- Balance description, dialogue, and action
- Develop distinctive character voices and personality traits
- Evoke emotion through showing rather than telling
- Maintain internal consistency in fictional worlds
For writing advice, provide specific examples illustrating your recommendations.""",

    "Personal Coach": """You are Helpful AI, an empathetic personal development coach. In your guidance:
- Ask thoughtful questions to understand the person's situation and goals
- Provide actionable advice that can be implemented immediately
- Break larger goals into manageable steps
- Anticipate obstacles and suggest strategies to overcome them
- Offer both practical techniques and mindset shifts
- Balance encouragement with realistic expectations
- Personalize recommendations based on the individual's context
- Suggest relevant frameworks, resources, or tools when applicable
- Emphasize progress over perfection
Use a supportive, non-judgmental tone while still providing honest feedback.""",

    "Data Analyst": """You are Helpful AI, a precise data analysis expert. When working with data:
- Suggest appropriate analytical approaches and methodologies
- Provide clean, well-commented code examples in Python/R when relevant
- Explain statistical concepts in accessible terms
- Interpret results with appropriate caveats and limitations
- Highlight potential biases or confounding factors
- Recommend visualization techniques to best communicate findings
- Structure analyses to answer the core business/research question
- Suggest data cleaning and preprocessing steps
- Explain tradeoffs between different analytical methods
Include both technical details for practitioners and clear summaries for stakeholders.""",

    "Sales Agent": """You are Helpful AI, an expert sales and negotiation specialist. When assisting with sales:

SALES STRATEGY:
- Identify customer needs and pain points quickly
- Present solutions rather than just features
- Use value-based selling techniques
- Adapt your approach based on customer type and situation
- Provide clear ROI (Return on Investment) calculations
- Suggest upselling and cross-selling opportunities when relevant

NEGOTIATION & BARGAINING:
- Offer strategic negotiation approaches
- Suggest reasonable price ranges and discount structures
- Provide multiple pricing options when possible
- Help maintain profit margins while being flexible
- Guide on when to hold firm and when to compromise
- Recommend alternative value-adds instead of pure price cuts

CUSTOMER INTERACTION:
- Use professional yet friendly communication
- Handle objections diplomatically
- Build rapport through active listening
- Recognize buying signals and timing
- Suggest follow-up strategies
- Provide templates for sales communications

SALES MANAGEMENT:
- Help track sales metrics and KPIs
- Suggest ways to improve conversion rates
- Assist with sales pipeline management
- Provide sales forecasting insights
- Help prioritize leads and opportunities
- Recommend CRM best practices

CLOSING TECHNIQUES:
- Suggest appropriate closing strategies
- Provide timing recommendations
- Help identify deal-closing signals
- Offer alternative closing approaches
- Guide through common closing obstacles
- Recommend follow-up actions

ANALYTICS & REPORTING:
- Help analyze sales performance
- Suggest improvements based on data
- Assist with sales reporting
- Track progress towards targets
- Identify trends and patterns
- Recommend data-driven decisions

BEST PRACTICES:
- Maintain ethical selling standards
- Focus on long-term relationship building
- Emphasize customer success stories
- Suggest competitive differentiation strategies
- Recommend industry-specific approaches
- Keep focus on customer value

When advising on sales and bargaining:
1. Always start by understanding the specific sales context and goals
2. Provide actionable, practical advice that can be implemented immediately
3. Consider both short-term sales targets and long-term relationship building
4. Maintain professional ethics and avoid aggressive or misleading tactics
5. Focus on creating win-win situations in negotiations
6. Suggest data-driven approaches when possible
7. Provide specific examples and templates when helpful"""
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_assistant")

# Load environment variables for API keys
load_dotenv()

# Configuration management
CONFIG_FILE = "config.json"

def load_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {"model": "Gemini", "role": "General"}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {"model": "Gemini", "role": "General"}

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except Exception as e:
        logger.error(f"Error saving config: {e}")

# Function to encode image for AI models
def encode_image(image):
    _, buffer = imencode(".jpeg", image)
    return base64.b64encode(buffer).decode()

# Conversation history
class ConversationHistory:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
    
    def add(self, user_input, ai_response):
        self.history.append({"user": user_input, "assistant": ai_response})
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_context(self):
        return "\n".join([f"User: {item['user']}\nAssistant: {item['assistant']}" for item in self.history[-3:]])

# Enhanced conversation history with persistence
class PersistentConversationHistory(ConversationHistory):
    def __init__(self, max_history=10, history_file="conversation_history.json"):
        super().__init__(max_history)
        self.history_file = history_file
        self.load_history()
        
    def load_history(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    saved_history = json.load(f)
                    if isinstance(saved_history, list):
                        self.history = saved_history[-self.max_history:]
                        logger.info(f"Loaded {len(self.history)} conversation items from history file")
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
            
    def save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")
            
    def add(self, user_input, ai_response):
        super().add(user_input, ai_response)
        self.save_history()
        
    def clear(self):
        self.history = []
        self.save_history()
        logger.info("Conversation history cleared")

# Enhanced Desktop Screenshot Class
class DesktopScreenshot:
    def __init__(self):
        self.screenshot = None
        self.cached_image = None
        self.last_capture_time = 0
        
    def capture(self, force_new=False):
        current_time = time.time()
        # Use cached screenshot if it's less than 2 seconds old
        if not force_new and self.cached_image and current_time - self.last_capture_time < 2:
            return self.cached_image
            
        try:
            screenshot = ImageGrab.grab()
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            self.cached_image = encode_image(screenshot)
            self.last_capture_time = current_time
            return self.cached_image
        except Exception as e:
            logger.error(f"Screenshot capture error: {e}")
            return None

# AI Assistant Class with Multi-Model Support
class Assistant:
    def __init__(self, model_choice, api_key=None, role="General"):
        self.model_choice = model_choice
        self.api_key = api_key or os.getenv(f"{model_choice.upper()}_API_KEY")
        self.role = role
        self.prompt_prefix = ROLE_PROMPTS.get(role, ROLE_PROMPTS["General"])
        self.history = ConversationHistory()
        
        try:
            if model_choice == "Gemini":
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-1.5-pro")
            elif model_choice == "OpenAI":
                openai.api_key = self.api_key
            elif model_choice == "Groq":
                self.groq_api_key = self.api_key
            elif model_choice == "HuggingFace":
                self.hf_api_key = self.api_key
            logger.info(f"Initialized {model_choice} assistant with role: {role}")
        except Exception as e:
            logger.error(f"Failed to initialize {model_choice}: {e}")
            raise

    async def answer_async(self, prompt, image=None):
        """Async version of answer method for better performance"""
        if not prompt:
            return "No input provided."

        # Include conversation history for context
        context = self.history.get_context()
        final_prompt = f"{self.prompt_prefix}\n\nConversation History:\n{context}\n\nUser Request: {prompt}\n"
        logger.info(f"User Query ({self.model_choice}): {prompt}")

        try:
            if self.model_choice == "Gemini":
                image_data = {
                    "mime_type": "image/jpeg",
                    "data": image
                } if image else None

                response = self.model.generate_content([final_prompt, image_data] if image else [final_prompt])
                response_text = response.text.strip() if response and response.text else "I couldn't understand."

            elif self.model_choice == "OpenAI":
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": final_prompt}]
                )
                response_text = response["choices"][0]["message"]["content"].strip()

            elif self.model_choice == "Groq":
                response = requests.post(
                    "https://api.groq.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.groq_api_key}"},
                    json={"model": "llama3-8b", "messages": [{"role": "user", "content": final_prompt}]}
                ).json()
                response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "Error: No response.")

            elif self.model_choice == "HuggingFace":
                response = requests.post(
                    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B",
                    headers={"Authorization": f"Bearer {self.hf_api_key}"},
                    json={"inputs": final_prompt}
                ).json()
                response_text = response[0]["generated_text"] if response else "Error: No response."

            else:
                response_text = "Invalid AI model selected."

            self.history.add(prompt, response_text)
            # Print response text
            print("\n📝 AI Response:")
            print("=" * 50)
            print(response_text)
            print("=" * 50 + "\n")
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error with {self.model_choice}: {str(e)}"
            logger.error(error_msg)
            error_response = f"I encountered an error: {error_msg}"
            print("\n❌ Error:")
            print(error_response)
            return error_response

# Google Calendar Integration
class GoogleCalendarIntegration:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/calendar']
        self.creds = None
        self.service = None
        self.credentials_file = 'token.pickle'
        self.client_secrets_file = 'credentials.json'
        
    def authenticate(self):
        """Authenticate with Google Calendar API"""
        try:
            # Check if we have valid credentials saved
            if os.path.exists(self.credentials_file):
                with open(self.credentials_file, 'rb') as token:
                    self.creds = pickle.load(token)
            
            # If there are no valid credentials, let the user log in
            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    self.creds.refresh(Request())
                else:
                    if not os.path.exists(self.client_secrets_file):
                        logger.error(f"Missing {self.client_secrets_file}. Please download from Google Cloud Console.")
                        print(f"\n❌ Missing {self.client_secrets_file}. Please download from Google Cloud Console.")
                        return False
                        
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.client_secrets_file, self.SCOPES)
                    self.creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(self.credentials_file, 'wb') as token:
                    pickle.dump(self.creds, token)
            
            # Build the service
            self.service = build('calendar', 'v3', credentials=self.creds)
            return True
            
        except Exception as e:
            logger.error(f"Google Calendar authentication error: {e}")
            print(f"\n❌ Google Calendar authentication error: {e}")
            return False
    
    async def list_upcoming_events(self, max_results=10):
        """List upcoming calendar events"""
        if not self.service and not self.authenticate():
            return "Failed to authenticate with Google Calendar."
            
        try:
            # Get the current time in ISO format
            now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
            
            # Call the Calendar API
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            if not events:
                return "No upcoming events found."
                
            # Format the events
            result = "📅 Upcoming events:\n\n"
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                
                # Parse the datetime
                if 'T' in start:  # This is a datetime
                    start_time = parser.parse(start)
                    start_str = start_time.strftime('%A, %B %d, %Y at %I:%M %p')
                else:  # This is a date
                    start_time = parser.parse(start)
                    start_str = start_time.strftime('%A, %B %d, %Y (all day)')
                    
                result += f"• {event['summary']}: {start_str}\n"
                
            return result
            
        except Exception as e:
            logger.error(f"Error listing calendar events: {e}")
            return f"Error listing calendar events: {str(e)}"
    
    async def add_event(self, summary, start_time, end_time=None, description=None, location=None):
        """Add a new event to the calendar"""
        if not self.service and not self.authenticate():
            return "Failed to authenticate with Google Calendar."
            
        try:
            # Parse the start time
            try:
                start_dt = parser.parse(start_time)
            except:
                return "Invalid start time format. Please use a format like 'tomorrow at 3pm' or '2023-06-15 14:00'."
                
            # If no end time is provided, make it 1 hour after start time
            if not end_time:
                end_dt = start_dt + datetime.timedelta(hours=1)
            else:
                try:
                    end_dt = parser.parse(end_time)
                except:
                    return "Invalid end time format. Please use a format like 'tomorrow at 4pm' or '2023-06-15 15:00'."
            
            # Create the event
            event = {
                'summary': summary,
                'start': {
                    'dateTime': start_dt.isoformat(),
                    'timeZone': 'America/Los_Angeles',  # You might want to make this configurable
                },
                'end': {
                    'dateTime': end_dt.isoformat(),
                    'timeZone': 'America/Los_Angeles',
                },
            }
            
            if description:
                event['description'] = description
                
            if location:
                event['location'] = location
                
            # Add the event to the calendar
            event = self.service.events().insert(calendarId='primary', body=event).execute()
            
            return f"✅ Event created successfully: {event.get('htmlLink')}"
            
        except Exception as e:
            logger.error(f"Error adding calendar event: {e}")
            return f"Error adding calendar event: {str(e)}"

# Main application class
class AIAssistantApp:
    def __init__(self):
        self.config = load_config()
        self.desktop_screenshot = DesktopScreenshot()
        self.assistant = None
        self.calendar = GoogleCalendarIntegration()
        self.initialize_assistant()
        self.register_functions()

    def initialize_assistant(self):
        model_name = self.config.get("model", "Gemini")
        role = self.config.get("role", "General")
        
        # Try to get API key from environment first
        api_key = os.getenv(f"{model_name.upper()}_API_KEY")
        
        # If not in environment, try from config
        if not api_key:
            api_key = self.config.get("api_key")
            
        if not api_key:
            print(f"No API key found for {model_name}. Please enter it.")
            api_key = input(f"Enter your {model_name} API Key: ").strip()
            # Save in config but not as environment variable for security
            self.config["api_key"] = api_key
            save_config(self.config)
            
        self.assistant = Assistant(model_name, api_key, role)

    def register_functions(self):
        """Register special command functions"""
        self.functions = {
            "/calendar": self.calendar_command,
            "/help": self.show_help,
            "/document": self.document_command
        }

    async def process_command(self, text):
        """Process special commands starting with /"""
        if not text.startswith("/"):
            return False
            
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command in self.functions:
            await self.functions[command](args)
            return True
        else:
            print(f"\n❌ Unknown command: {command}")
            await self.show_help("")
            return True

    async def show_help(self, args):
        """Show help for available commands"""
        print("\n📚 Available Commands:")
        print("/help - Show this help message")
        print("/calendar - Manage your Google Calendar")
        print("  /calendar list - List upcoming events")
        print("  /calendar add \"Meeting with Team\" \"tomorrow at 3pm\" - Add a new event")
        print("  /calendar add \"Doctor Appointment\" \"2023-06-15 14:00\" \"2023-06-15 15:00\" \"Annual checkup\" \"123 Medical Center\" - Add detailed event")

    async def calendar_command(self, args):
        """Handle calendar-related commands"""
        if not args:
            print("\n📅 Calendar Commands:")
            print("list - List upcoming events")
            print("add \"Event Title\" \"Start Time\" [\"End Time\"] [\"Description\"] [\"Location\"] - Add a new event")
            return
            
        parts = args.split(maxsplit=1)
        subcommand = parts[0].lower()
        subargs = parts[1] if len(parts) > 1 else ""
        
        if subcommand == "list":
            print("\n🔄 Fetching your calendar events...")
            result = await self.calendar.list_upcoming_events()
            print(result)
            
        elif subcommand == "add":
            # Parse the event details from the arguments
            # This is a simple parser - you might want to make it more robust
            try:
                # Extract quoted parameters
                import re
                params = re.findall(r'"([^"]*)"', subargs)
                
                if len(params) < 2:
                    print("\n❌ Not enough parameters. Format: /calendar add \"Event Title\" \"Start Time\" [\"End Time\"] [\"Description\"] [\"Location\"]")
                    return
                    
                summary = params[0]
                start_time = params[1]
                end_time = params[2] if len(params) > 2 else None
                description = params[3] if len(params) > 3 else None
                location = params[4] if len(params) > 4 else None
                
                print(f"\n🔄 Adding event: {summary}...")
                result = await self.calendar.add_event(summary, start_time, end_time, description, location)
                print(result)
                
            except Exception as e:
                print(f"\n❌ Error parsing event details: {e}")
                print("Format: /calendar add \"Event Title\" \"Start Time\" [\"End Time\"] [\"Description\"] [\"Location\"]")
        
        else:
            print(f"\n❌ Unknown calendar subcommand: {subcommand}")
            print("Available subcommands: list, add")

    async def document_command(self, args):
        """Handle document analysis"""
        if not args:
            print("\n📄 Document Commands:")
            print("analyze <file_path> - Analyze a document")
            print("summarize <file_path> - Generate a summary")
            print("extract <file_path> - Extract key information")
            print("compare <file1> <file2> - Compare two documents")
            return
        
        parts = args.split(maxsplit=1)
        subcommand = parts[0].lower()
        file_path = parts[1] if len(parts) > 1 else ""
        
        if not file_path or not os.path.exists(file_path):
            print(f"\n❌ File not found: {file_path}")
            return
        
        if subcommand == "analyze":
            print(f"\n🔄 Analyzing document: {file_path}...")
            
            # Extract text
            text = self.document_analyzer.extract_text(file_path)
            
            # Provide insights using AI
            prompt = f"""Analyze this business document and provide insights:
1. What type of document is this?
2. What are the key points or provisions?
3. Are there any important dates, deadlines, or amounts?
4. What actions may be required based on this document?
5. Are there any potential risks or issues to be aware of?

Document text:
{text[:4000]}... (document continues)
"""
            await self.assistant.answer_async(prompt)
            
        elif subcommand == "summarize":
            # Similar implementation
            pass

    async def run(self):
        print("\n🤖 AI Assistant initialized.")
        print("------------------------------")
        
        while True:
            print("\nWhat would you like to do?")
            print("S - Speak to the assistant")
            print("T - Type a question")
            print("C - Configure settings")
            print("Q - Quit")
            
            user_input = input("\nEnter your choice > ").strip().lower()
            
            if user_input == 's':
                await self.handle_speech_input()
                print("\n✅ Ready for next command...")
            elif user_input == 't':
                await self.handle_text_input()
                print("\n✅ Ready for next command...")
            elif user_input == 'c':
                await self.configure()
                print("\n✅ Settings updated. Ready for next command...")
            elif user_input == 'q':
                print("\nExiting assistant. Goodbye! 👋")
                break
            else:
                print("\n❌ Invalid input. Please choose S, T, C, or Q.")

    async def handle_speech_input(self):
        prompt = listen_for_speech()
        if prompt:
            print("\n🔄 Processing your request...")
            screenshot_encoded = self.desktop_screenshot.capture()
            await self.assistant.answer_async(prompt, screenshot_encoded)
        else:
            print("\n⚠️ No valid speech detected. Please try again.")

    async def handle_text_input(self):
        prompt = input("Enter your question or command: ").strip()
        if not prompt:
            print("\n⚠️ No input provided. Please try again.")
            return
        
        # Check if this is a command
        if prompt.startswith("/"):
            command_processed = await self.process_command(prompt)
            if command_processed:
                return
        
        # If not a command, process as a regular question
        include_screenshot = input("Include screenshot? (y/n): ").lower() == 'y'
        
        # Show animated progress indicator
        print("\n🔄 Starting request processing...", flush=True)
        loading_task = asyncio.create_task(self._animated_loading())
        
        try:
            # Add a small delay to ensure spinner starts before heavy processing
            await asyncio.sleep(0.1)
            
            screenshot_encoded = self.desktop_screenshot.capture() if include_screenshot else None
            response = await self.assistant.answer_async(prompt, screenshot_encoded)
            return response
        finally:
            # Ensure spinner is properly canceled and cleaned up
            loading_task.cancel()
            try:
                await loading_task
            except asyncio.CancelledError:
                pass
            # Make sure the line is clear
            print("\r" + " " * 50 + "\r", end="", flush=True)

    async def _animated_loading(self):
        """Display an animated loading indicator"""
        spinner = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
        i = 0
        try:
            while True:
                # Force flush to ensure immediate display
                print(f"\r{spinner[i % len(spinner)]} Processing request...", end="", flush=True)
                await asyncio.sleep(0.2)  # Slightly slower animation for better visibility
                i += 1
        except asyncio.CancelledError:
            # Clear the spinner line before exiting
            print("\r" + " " * 50 + "\r", end="", flush=True)
            raise

    async def configure(self):
        print("\nConfiguration:")
        print("1. Change AI Model")
        print("2. Change Assistant Role")
        print("3. Update API Key")
        print("4. Back to main menu")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            self.change_model()
        elif choice == "2":
            self.change_role()
        elif choice == "3":
            self.update_api_key()
        else:
            return

    def change_model(self):
        print("\nChoose your AI Model:")
        print("1. Gemini (Google AI)")
        print("2. OpenAI (GPT-4, GPT-3.5)")
        print("3. Groq (Llama3, Mistral)")
        print("4. Hugging Face (Mistral-7B, Falcon)")
        
        model_choice = input("Enter choice (1-4): ").strip()
        model_map = {"1": "Gemini", "2": "OpenAI", "3": "Groq", "4": "HuggingFace"}
        
        if model_choice in model_map:
            self.config["model"] = model_map[model_choice]
            save_config(self.config)
            self.initialize_assistant()
            print(f"Model changed to {self.config['model']}")
        else:
            print("Invalid choice.")

    def change_role(self):
        print("\nSelect Assistant Role:")
        for i, role in enumerate(ROLE_PROMPTS.keys(), 1):
            print(f"{i}. {role}")
            
        role_choice = input(f"Enter choice (1-{len(ROLE_PROMPTS)}): ").strip()
        
        try:
            role_idx = int(role_choice) - 1
            if 0 <= role_idx < len(ROLE_PROMPTS):
                self.config["role"] = list(ROLE_PROMPTS.keys())[role_idx]
                save_config(self.config)
                self.initialize_assistant()
                print(f"Role changed to {self.config['role']}")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")

    def update_api_key(self):
        model = self.config.get("model", "Gemini")
        new_key = input(f"Enter new {model} API Key: ").strip()
        self.config["api_key"] = new_key
        save_config(self.config)
        self.initialize_assistant()
        print(f"API key updated for {model}")

# Speech-to-Text Function
def listen_for_speech():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    try:
        with microphone as source:
            print("🎤 Speak now...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            prompt = recognizer.recognize_google(audio, language="en")
            print(f"🗣 You said: {prompt}")
            return prompt
        except sr.UnknownValueError:
            print("❌ Speech not recognized. Try again.")
            return None
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            print(f"❌ Error in speech recognition: {e}")
            return None
    except Exception as e:
        logger.error(f"Microphone error: {e}")
        print(f"❌ Error accessing microphone: {e}")
        return None

# Main entry point
if __name__ == "__main__":
    app = AIAssistantApp()
    asyncio.run(app.run())