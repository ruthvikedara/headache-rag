from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
import pandas as pd
import numpy as np
import atexit


app = Flask(__name__)

# Initialize your models and database
llama3_8b = "/Users/ruthvik/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

llm = LlamaCpp(
    model_path=llama3_8b,
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def create_vector_database(csv_file, model):
  """Creates a vector database from a CSV file using a custom sentence transformer model.

  Args:
    csv_file: Path to the CSV file.
    model_name: Name of the Hugging Face model.

  Returns:
    A dictionary where keys are question embeddings and values are a tuple of question and answer.
  """

  df = pd.read_csv(csv_file)


  def encode_sentence(sentence):
      outputs = model.encode(sentence)
      return outputs

  embeddings = [encode_sentence(question) for question in df['question']]

  vector_database = {}
  for i, embedding in enumerate(embeddings):
      vector_database[tuple(embedding)] = (df['question'][i], df['answer'][i])

  return vector_database

db = create_vector_database('data2/headache_qna.csv', embed_model)

def similarity_search(query,vector_database,model=embed_model, k=2):
  """Performs a similarity search on the vector database.

  Args:
    query_embedding: The embedding of the query.
    vector_database: The vector database as a dictionary.
    k: The number of top similar items to return.

  Returns:
    A list of tuples containing the question and answer for the top k similar items.
  """

  query_embedding = encode_sentence(query, model)
  similarities = []
  for embedding, (question, answer) in vector_database.items():
    similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
    similarities.append((question, answer, similarity))

  similarities.sort(key=lambda x: x[2], reverse=True)
  for s in similarities[:k]:
    print(f"Question: {s[0]}")
    print(f"Answer: {s[1]}")
    print(f'Cosine Similarity: {s[2]:.4f}')
    print('-'*200)
  return similarities[:k]

def encode_sentence(sentence, model):
  outputs = model.encode(sentence)
  return outputs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/similarity_search', methods=['POST'])
def perform_similarity_search():
    user_message = request.json['message']
    
    # Perform similarity search
    similarity_results = similarity_search(user_message, db)
    
    # Format similarity results
    formatted_results = []
    for question, answer, similarity in similarity_results:
        formatted_results.append({
            'question': question,
            'answer': answer,
            'similarity': f'{similarity:.4f}'
        })
    
    return jsonify({'similarity_results': formatted_results})

@app.route('/llm_response', methods=['POST'])
def generate_llm_response():
    user_message = request.json['message']
    
    # Generate LLM response
    prompt = f'''
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful, confident assistant to a neurologist in an Indian corporate hospital. Your goal is to answer questions that patients might have related only to headaches.
    Only give them useful, helpful, positive advice. Try and reassure the patient. Give your answer with respect to the Indian context you are dealing with.
    Do not answer if you don't know something, say that you don't know. Ask them to talk to the physician to know more.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>

    {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    '''

    llm_output = llm.invoke(prompt)
    
    return jsonify({'llm_response': llm_output})

# Cleanup function to close the LLM model
def cleanup():
    global llm
    if llm:
        llm.close()
        llm = None

# Register the cleanup function to be called on exit
atexit.register(cleanup)

if __name__ == '__main__':
    app.run(debug=True)