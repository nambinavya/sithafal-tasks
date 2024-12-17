# Required Libraries
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai


openai.api_key = "your_openai_api_key"


def scrape_website(url):
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

   
    paragraphs = soup.find_all('p')
    content = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
    
   
    chunks = []
    chunk = ""
    for paragraph in content:
        if len(chunk.split()) + len(paragraph.split()) <= 300:
            chunk += " " + paragraph
        else:
            chunks.append(chunk.strip())
            chunk = paragraph
    if chunk:
        chunks.append(chunk.strip())
    
    return chunks


def generate_embeddings(chunks, model):
   
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return np.array(embeddings)


def store_embeddings_in_faiss(embeddings, chunks):
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    metadata = {i: chunks[i] for i in range(len(chunks))}

    return index, metadata


def retrieve_relevant_chunks(query, model, index, metadata, top_k=5):
    
    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding)

   
    distances, indices = index.search(query_embedding, top_k)

    

    retrieved_chunks = [metadata[i] for i in indices[0]]
    return retrieved_chunks



def generate_response(query, retrieved_chunks):
    

 
    context = "\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nUser Query: {query}\n\nAnswer:"

  

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        temperature=0
    )
    return response.choices[0].text.strip()


def rag_pipeline(url, query):
   

    model = SentenceTransformer('all-MiniLM-L6-v2')

   

    chunks = scrape_website(url)

   

    embeddings = generate_embeddings(chunks, model)

    

    index, metadata = store_embeddings_in_faiss(embeddings, chunks)

   

    retrieved_chunks = retrieve_relevant_chunks(query, model, index, metadata)

   

    response = generate_response(query, retrieved_chunks)

    return response



if __name__ == "__main__":
   

    target_url = "https://en.wikipedia.org/wiki/Natural_language_processing"
    user_query = "What are the common applications of natural language processing?"

 
 
    answer = rag_pipeline(target_url, user_query)
    print("\nResponse:\n", answer)
