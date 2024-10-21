import os
import uuid
import faiss
import numpy as np
import pandas as pd
import textwrap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# Load environment variables if needed (optional)
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="ILATE LMS Chatbot API")

# Configure CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the dataframe with precomputed embeddings
EMBEDDINGS_FILE = "data_with_embeddings.feather"
if not os.path.exists(EMBEDDINGS_FILE):
    raise FileNotFoundError(f"{EMBEDDINGS_FILE} not found.")

df = pd.read_feather(EMBEDDINGS_FILE)

# Number of passages to be retrieved
TOP_N = 3

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

# In-memory session store
store = {}


# Pydantic models for request and response
class StartSessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    user_input: str


class ChatResponse(BaseModel):
    response: str


# Helper Functions


def find_best_passages(query: str, dataframe: pd.DataFrame, top_n: int = TOP_N) -> list:
    """
    Finds the top_n most relevant passages for the given query using embeddings.
    """
    query_embedding = genai.embed_content(
        model="models/text-embedding-004", content=query
    )["embedding"]
    dot_products = np.dot(np.stack(dataframe["Embeddings"]), query_embedding)
    top_indices = np.argsort(dot_products)[-top_n:][::-1]
    return dataframe.iloc[top_indices]["answer"].tolist()


def make_prompt(query: str, relevant_passages: list, top_n: int = TOP_N) -> str:
    """
    Constructs the prompt to be sent to the Gemini model, ensuring the chatbot
    only answers ILATE-related queries.
    """
    escaped_passages = [
        passage.replace("'", "").replace('"', "").replace("\n", " ")
        for passage in relevant_passages
    ]
    joined_passages = "\n\n".join(
        f"PASSAGE {i+1}: {passage}" for i, passage in enumerate(escaped_passages)
    )

    prompt = textwrap.dedent(
        f"""
    Persona: You are an ILATE AI Chatbot, a specialized assistant knowledgeable in ILATE Learning Management System (LMS) and ILATE organization matters. You are responsible for providing accurate and helpful information about ILATE's features, courses, functionalities, and policies.

    Task: You are ONLY allowed to answer questions related to the ILATE LMS or ILATE organization. If a user asks a question that is not related to ILATE or its services, politely refuse to answer, explaining that your role is limited to providing information about ILATE. Do not attempt to answer general knowledge or unrelated questions.

    Format: Provide clear and concise answers related to ILATE. If a question is outside of the ILATE scope, respond with: "I'm sorry, I can only assist with ILATE-related queries." Always guide users back to ILATE-related topics when necessary.

    Context: Below are {top_n} passages that might be relevant to the user's query. Only use them if they are relevant to the question. Avoid using irrelevant context.

    Passages: {joined_passages}

    QUESTION: '{query}'

    ANSWER:
    """
    )
    return prompt


def is_ilate_related(query: str) -> bool:
    """
    Determines if the user's query is related to ILATE based on predefined keywords.
    """
    ilate_keywords = [
        "ILATE",
        "LMS",
        "course",
        "courses",
        "features",
        "policies",
        "support",
        "platform",
        "organization",
        "learning",
        "institute",
    ]
    return any(keyword.lower() in query.lower() for keyword in ilate_keywords)


# Endpoint to start a new session
@app.post("/start-session", response_model=StartSessionResponse)
async def start_session():
    """
    Initializes a new chat session and returns a unique session_id.
    """
    session_id = str(uuid.uuid4())
    # Initialize chat history with the welcome message
    store[session_id] = [
        {
            "role": "model",
            "parts": "Welcome to the ILATE AI Chatbot! How can I assist you today?",
        }
    ]
    return StartSessionResponse(session_id=session_id)


# Endpoint to handle chat messages
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Processes user input and returns the chatbot's response.
    """
    session_id = chat_request.session_id
    user_input = chat_request.user_input.strip()

    # Validate session
    if session_id not in store:
        raise HTTPException(
            status_code=404, detail="Session not found. Please start a new session."
        )

    # Initialize chat history
    chat_history = store[session_id]

    # Check if the query is ILATE-related
    if not is_ilate_related(user_input):
        response_text = "I'm sorry, I can only assist with ILATE-related queries."
        # Optionally, you can still append this to history
        chat_history.append({"role": "user", "parts": user_input})
        chat_history.append({"role": "model", "parts": response_text})
        return ChatResponse(response=response_text)

    # Find relevant passages
    relevant_passages = find_best_passages(user_input, df, TOP_N)
    prompt = make_prompt(user_input, relevant_passages, TOP_N)

    # Append user message to chat history
    chat_history.append({"role": "user", "parts": user_input})

    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        chat = model.start_chat(history=chat_history)

        # Send the prompt to Gemini and get the response
        response = chat.send_message(prompt)
        response_text = response.text.strip()

        # Append model response to chat history
        chat_history.append({"role": "model", "parts": response_text})

        return ChatResponse(response=response_text)

    except genai.types.generation_types.StopCandidateException as e:
        # Handle the specific exception
        response_text = (
            "The model stopped generating content prematurely. "
            "Please try a different question or reduce the query length."
        )
        chat_history.append({"role": "model", "parts": response_text})
        return ChatResponse(response=response_text)

    except Exception as e:
        # Handle other exceptions
        raise HTTPException(status_code=500, detail=str(e))


# Optional: Clear chat history (if you want an endpoint for that)
# @app.post("/clear-chat")
# async def clear_chat(session_id: str):
#     if session_id in store:
#         store[session_id] = [
#             {
#                 "role": "model",
#                 "parts": "Welcome to the ILATE AI Chatbot! How can I assist you today?"
#             }
#         ]
#         return {"detail": "Chat history cleared."}
#     else:
#         raise HTTPException(status_code=404, detail="Session not found.")

# Run the app using: uvicorn your_script_name:app --host 0.0.0.0 --port 8000
