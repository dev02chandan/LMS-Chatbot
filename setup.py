import pandas as pd
import google.generativeai as genai
import os

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=api_key)

# Load JSON data
data = pd.read_json('data.json')

df = pd.DataFrame(data)
df.columns = ['question', 'answer']

# Get the embeddings of each answer and add to an embeddings column in the dataframe
def embed_fn(question, answer):
    content = f'Question: {question} \n\n Answer: {answer}'
    return genai.embed_content(model='models/text-embedding-004', content=content)["embedding"]

df['Embeddings'] = df.apply(lambda row: embed_fn(row['question'], row['answer']), axis=1)

# Save the dataframe to a feather file
df.to_feather('data_with_embeddings.feather')