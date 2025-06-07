from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Initialize HuggingFace embedding model (MiniLM is a good default)
embedding_model = OpenAIEmbeddings()

# Store embeddings in FAISS
vectorstore = FAISS.load_local("main_faiss_huggingface_index", embedding_model, allow_dangerous_deserialization=True)

# Step 1: Create LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # or "gpt-4" if you have access
    temperature=0,
    
)

# Step 2: Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Step 3: Ask a question
query = "What is backtracking?"
result = qa_chain(query)

# Optionally, show sources
print("\n📄 Sources:")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("source", "Unknown"), "\n", doc.page_content[:200])

# Show answer
print("🧠 Answer:", result["result"])