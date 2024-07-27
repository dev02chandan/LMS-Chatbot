# ILATE LMS RAG-Based Chatbot

A RAG-based chatbot for ILATE Learning Management System. Check the app [here](https://ilatelms.streamlit.app/)

## Features

- RAG Implementation with Gemini API.
- Provides information about ILATE, its services, and related topics.
- Precomputes embeddings for faster responses.
- Retrieves and presents the top n relevant passages for comprehensive answers.
- User-friendly interface with a conversational tone.

## Prerequisites

- Python 3.7 or higher
- Streamlit
- Google Generative AI library
- FAISS (for similarity search)
- pandas

## Setup Instructions

### Step 1: Prepare the Data

1. Create a JSON file named `data.json` in the following format:

```json
[
    {
        "userType": "Student",
        "question": "What is ILATE?",
        "answer": "ILATE is a comprehensive Learning Management System offering courses for students from 8th grade onwards, including IBDP, IGCSE, MYP, A Levels, SAT/ACT, and A/ML."
    },
    {
        "userType": "Teacher",
        "question": "What can teachers do on the platform?",
        "answer": "Teachers can take attendance, upload lecture notes, handwritten notes, extensive worksheets, question banks, and test series. They can also make announcements."
    },
    ...
]
```

### Step 2: Create a Virtual Environment

1. Create a virtual environment:

```bash
python3 -m venv env
```

2. Activate the virtual environment:

- On Windows:

```bash
.\env\Scripts\activate
```

- On macOS and Linux:

```bash
source env/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

### Step 3: Precompute Embeddings

Run the setup script to compute embeddings and save the DataFrame:

```bash
python3 setup.py
```

### Step 4: Run the Streamlit Application

Run the Streamlit app:

```bash
streamlit run app.py
```

## Files

- `data.json`: The JSON file containing data about ILATE LMS. (You can add your data)
- `setup.py`: Script to compute embeddings and save the data. (For RAG)
- `app.py`: The main Streamlit application.

## Usage

1. Add your data to `data.json` in the specified format.
2. Run `setup.py` to precompute embeddings and save the data.
3. Run `app.py` to start the Streamlit application and interact with the ILATE LMS Chatbot.

## License

This project is licensed under the MIT License.
