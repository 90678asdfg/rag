import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

from langchain.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS
from langchain.chains import retrieval_qa
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "Llama 2"  # Ensure this is the correct model name
OLLAMA_SERVER_URL = "http://45.114.48.221:11434/"  # Local server URL

# Initialize the model and embeddings
if MODEL.startswith("gpt"):
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
else:
    model = Ollama(model=MODEL, base_url=OLLAMA_SERVER_URL)
    embeddings = OllamaEmbeddings(base_url=OLLAMA_SERVER_URL)

# Load and split the PDF documents from the specified folder
pdf_folder_path = r"C:\Users\ADMIN\Desktop\rag\files"
documents = []

logger.info("Loading PDF documents")
for filename in os.listdir(pdf_folder_path):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder_path, filename)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

logger.info("Splitting documents into chunks")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

logger.info("Creating FAISS vector store")
vectorstore = FAISS.from_texts([text.page_content for text in texts], embeddings)

# Define the retrieval model
retriever = vectorstore.as_retriever(k=4)

# Create a PromptTemplate
template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)
parser = StrOutputParser()

# Load the QA chain
logger.info("Loading QA chain")
qa_chain = load_qa_chain(llm=model, chain_type="stuff")

# Combine the components into a RetrievalQA chain
qa_retrieval_chain = retrieval_qa(
    retriever=retriever,
    combine_documents_chain=qa_chain,
    input_key="question",
    output_key="answer"
)

# Function to query the system
def query_rag_system(question):
    response = qa_retrieval_chain({"question": question})
    return response["answer"]

# Example query
if __name__ == "__main__":
    question = "What are the main points discussed in the education system?"
    answer = query_rag_system(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
