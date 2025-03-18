from fastapi import FastAPI, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel # type: ignore
from typing import List, Dict, Optional, Any
import google.generativeai as genai # type: ignore
import os
from dotenv import load_dotenv # type: ignore
import json
import requests # type: ignore

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Configure the Gemini API with safety settings
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model with specific configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    generation_config=generation_config,
    safety_settings=safety_settings
)

class Message(BaseModel):
    role: str
    content: str

class UserProfile(BaseModel):
    name: str
    age: str
    hobbies: str
    additionalInfo: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message]
    userProfile: Optional[UserProfile] = None

class ChatResponse(BaseModel):
    reply: str
    updated_context: Optional[Dict[str, Any]] = None

def extract_user_info(message: str, current_context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract potential user information from the message."""
    updated_context = current_context.copy()
    
    # Check for name mentions
    name_indicators = ["my name is", "my age is", "I am", "I'm", "call me"]
    for indicator in name_indicators:
        if indicator.lower() in message.lower():
            try:
                name_start = message.lower().index(indicator.lower()) + len(indicator)
                name = message[name_start:].strip().split()[0]
                updated_context["user_name"] = name
            except:
                pass
    
    return updated_context

def format_context_for_prompt(context: Dict[str, Any]) -> str:
    """Format the shared context into a prompt prefix."""
    if not context:
        return ""
    
    context_parts = []
    if "user_name" in context:
        context_parts.append(f"The user's name is {context['user_name']}.")
    
    return " ".join(context_parts) + "\n\n" if context_parts else ""

@app.get("/list-models")
async def list_models():
    """List available Gemini models"""
    try:
        response = requests.get(
            "https://generativelanguage.googleapis.com/v1/models",
            headers={"x-goog-api-key": GEMINI_API_KEY}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Format the conversation history
        formatted_history = []
        
        # Add user profile context as an assistant message if available
        if request.userProfile:
            profile_context = f"""I should remember the following about you:
Name: {request.userProfile.name}
Age: {request.userProfile.age}
Hobbies: {request.userProfile.hobbies}
Additional Info: {request.userProfile.additionalInfo}

I'll use this information to provide personalized responses throughout our conversation."""
            
            formatted_history.append({
                "role": "model",
                "parts": [{"text": profile_context}]
            })
            
            # Add user acknowledgment
            formatted_history.append({
                "role": "user",
                "parts": [{"text": "Yes, please remember my profile information."}]
            })
        
        # Add conversation history
        for msg in request.history:
            formatted_history.append({
                "role": "user" if msg.role == "user" else "model",
                "parts": [{"text": msg.content}]
            })
        
        # Add the current message with user profile context
        current_message = request.message
        if request.userProfile and not formatted_history:  # Only add reminder for first message
            current_message = f"{request.message}\n\nNote: I'm {request.userProfile.name}, {request.userProfile.age} years old, and I enjoy {request.userProfile.hobbies}."
        
        formatted_history.append({
            "role": "user",
            "parts": [{"text": current_message}]
        })
        
        # Prepare the request to Gemini API
        api_url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"
        
        payload = {
            "contents": formatted_history,
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1000,
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 404:
                # Handle model not found error
                available_models_response = requests.get(
                    "https://generativelanguage.googleapis.com/v1/models",
                    headers={"x-goog-api-key": GEMINI_API_KEY}
                )
                available_models = available_models_response.json().get("models", [])
                model_names = [model["name"] for model in available_models]
                raise HTTPException(
                    status_code=404,
                    detail=f"Model not found. Available models: {', '.join(model_names)}"
                )
            
            response.raise_for_status()
            data = response.json()

            # Extract the response text
            if "candidates" in data and len(data["candidates"]) > 0:
                response_text = data["candidates"][0]["content"]["parts"][0]["text"]
                return {
                    "reply": response_text,
                    "updated_context": None
                }
            else:
                print("Unexpected API response:", data)  # Log the unexpected response
                raise HTTPException(status_code=500, detail="No response generated from the API")

        except requests.exceptions.RequestException as e:
            print(f"API Request Error: {str(e)}")  # Log the API error
            if hasattr(e.response, 'json'):
                try:
                    error_data = e.response.json()
                    print("API Error Response:", error_data)  # Log the error response
                    error_message = error_data.get('error', {}).get('message', str(e))
                    raise HTTPException(status_code=e.response.status_code, detail=error_message)
                except ValueError:
                    pass
            raise HTTPException(status_code=500, detail=f"Error from Gemini API: {str(e)}")

    except Exception as e:
        print(f"Unexpected error: {str(e)}")  # Log any unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/")
async def root():
    return {"message": "FastAPI Gemini Chatbot is running!"}
