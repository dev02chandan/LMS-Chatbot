import streamlit as st
import google.generativeai as genai
import os
import faiss
import numpy as np
import pandas as pd
import textwrap

# App title and configuration
st.set_page_config(page_title="LMS Chatbot")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Add logo
logo_path = "logo.jpg"
if os.path.exists(logo_path):
    st.image(logo_path, width=200)

# Load the dataframe with precomputed embeddings
df = pd.read_feather("data_with_embeddings.feather")

# Number of passages to be retrieved
top_n = 3


# Function to find the best passages
def find_best_passages(query, dataframe, top_n=top_n):
    query_embedding = genai.embed_content(
        model="models/text-embedding-004", content=query
    )["embedding"]
    dot_products = np.dot(np.stack(dataframe["Embeddings"]), query_embedding)
    top_indices = np.argsort(dot_products)[-top_n:][::-1]
    return dataframe.iloc[top_indices]["answer"].tolist()


def make_prompt(query, relevant_passages, top_n=5):
    # Escape any single or double quotes in the passages and join them for context
    escaped_passages = [
        passage.replace("'", "").replace('"', "").replace("\n", " ")
        for passage in relevant_passages
    ]
    joined_passages = "\n\n".join(
        f"PASSAGE {i+1}: {passage}" for i, passage in enumerate(escaped_passages)
    )

    # Updated task-specific prompt with strict focus on ILATE-related questions
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


# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "model",
            "parts": "Welcome to the ILATE AI Chatbot! How can I assist you today?",
        }
    ]

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome to the ILATE AI Chatbot! How can I assist you today?",
        }
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome to the ILATE AI Chatbot! How can I assist you today?",
        }
    ]
    st.session_state.chat_history = [
        {
            "role": "model",
            "parts": "Welcome to the ILATE AI Chatbot! How can I assist you today?",
        }
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


def generate_gemini_response(query):
    try:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "parts": query})

        relevant_passages = find_best_passages(query, df)
        prompt = make_prompt(query, relevant_passages)

        # Create chat with history included
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        chat = model.start_chat(history=st.session_state.chat_history)

        # Get response and update chat history
        response = chat.send_message(prompt)
        st.session_state.chat_history.append({"role": "model", "parts": response.text})

        return response.text

    except genai.types.generation_types.StopCandidateException as e:
        st.error(
            "The model stopped generating content prematurely. Please try a different question or reduce the query length."
        )
        return str(
            e
        )  # Optionally log or return the exception message for further debugging


# User-provided prompt
if prompt := st.chat_input(disabled=not gemini_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_gemini_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
