import pdfplumber
import os
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def extract_text_from_pdfs(pdf_folder):
   
    texts = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(pdf_folder, file_name)
            print(f"Extracting from: {file_name}")
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:  
                        texts.append(extracted_text)
    return texts

def chunk_texts(texts, chunk_size=500):
    
    chunks = []
    for text in texts:
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
    return chunks

def create_vector_store(chunks):
   
  
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding_model)
    return vector_store

def generate_response(query, vector_store):
   
   
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    response = qa_chain.run(query)
    return response

def main(pdf_folder, query):
    
    
    extracted_texts = extract_text_from_pdfs(pdf_folder)

   
    chunks = chunk_texts(extracted_texts)

    
    vector_store = create_vector_store(chunks)

   
    response = generate_response(query, vector_store)
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
  
    pdf_folder_path = "/Users/navya/Desktop/pdfs"  
    user_query = "What is the unemployment rate for Bachelor's degrees?"  
    main(pdf_folder_path, user_query)
