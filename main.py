# ==============================================================================
# main.py: FINAL 'LOAD-ONLY' FastAPI Application
# This API only loads pre-existing resources. It does not build them.
# ==============================================================================
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from huggingface_hub import InferenceClient
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Load Environment Variables ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set.")

# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    question: str
class AnswerResponse(BaseModel):
    answer: str

# --- RAG Initialization (Runs ONCE on startup) ---
print("Initializing RAG chain...")
try:
    # 1. Load the PRE-BUILT Retriever
    print("Loading pre-built FAISS index...")
    vector_store_path = "nasa_data/faiss_index"
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(f"FAISS index not found at {vector_store_path}. Ensure the folder is in the correct location.")
    
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    print("Retriever loaded successfully.")

    # 2. Initialize the direct InferenceClient
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    client = InferenceClient(model=repo_id, token=hf_token)

    # 3. Create our reliable LLM call function
    def call_chat_api(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(messages=messages, max_tokens=1024, temperature=0.1)
        return response.choices[0].message.content

    # 4. Create Prompt Template
    template = """
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt_template = ChatPromptTemplate.from_template(template)

    # 5. Assemble the final RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | RunnableLambda(lambda x: x.to_string())
        | RunnableLambda(call_chat_api)
        | StrOutputParser()
    )
    print("✅ RAG Chain initialized successfully.")
except Exception as e:
    print(f"❌ An error occurred during initialization: {e}")
    rag_chain = None

# --- FastAPI Application ---
app = FastAPI(title="NASA Farmer RAG Chatbot API")

@app.post("/ask", response_model=AnswerResponse)
async def ask_rag_system(request: QuestionRequest):
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG chain is not initialized. Check server logs.")
    try:
        response = rag_chain.invoke(request.question)
        return AnswerResponse(answer=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process question: {e}")

@app.get("/", summary="Health Check")
async def root():
    return {"status": "ok"}