from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pyttsx3
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

# Enable CORS to allow requests from frontend (e.g., React on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust the origin as necessary (for development, it's React on localhost)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize TTS engine (Optional, remove if you don't want TTS on the server side)
engine = pyttsx3.init()

# Define the template for the prompt
template = """
Answer the question below.

Here is the conversation history:
{context}
Question: {question}

Answer:
"""

# Initialize the model and prompt
model = OllamaLLM(model="llama3.1")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# File to save the context
CONTEXT_FILE = "conversation_context.txt"

def load_context():
    """Load the context from the file."""
    if os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE, "r") as file:
            return file.read()
    return ""

def save_context(context):
    """Save the context to the file by appending to it."""
    with open(CONTEXT_FILE, "a") as file:
        file.write(context)

# Request model for incoming user message
class ChatRequest(BaseModel):
    message: str

# Endpoint to handle chatbot requests
@app.post("/chatbot")
async def chatbot(request: ChatRequest):
    user_input = request.message
    context = load_context()  # Load previous conversation context
    
    # Generate response using the chatbot model
    try:
        result = chain.invoke({"context": context, "question": user_input})
        response = result.strip()  # Clean up the response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")
    
    # Update context and save
    context += f"\nUser: {user_input}\nBot: {response}\n"
    save_context(context)
    
    # Optional: Text-to-speech response (remove if not needed)
    engine.say(response)
    engine.runAndWait()

    return {"response": response}

# Endpoint to clear the context (optional)
@app.post("/clear-context")
async def clear_context():
    if os.path.exists(CONTEXT_FILE):
        os.remove(CONTEXT_FILE)
    return {"status": "context cleared"}
