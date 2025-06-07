from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Your sample text
text = """
The solar system consists of the Sun and everything bound to it by gravity.
This includes planets, moons, asteroids, comets, and meteoroids.
Earth is the third planet from the Sun and the only known planet to support life.
""" * 5

# Split the text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_text(text)

# Initialize HuggingFace embedding model (MiniLM is a good default)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store embeddings in FAISS
vectorstore = FAISS.from_texts(chunks, embedding_model)

# Run a test search
query = "What planet supports life?"
results = vectorstore.similarity_search(query, k=2)

# Save to local folder
vectorstore.save_local("sample_faiss_huggingface_index")

# Load later
#vectorstore = FAISS.load_local("sample_faiss_huggingface_index", embedding_model)


# Show results
print("🔍 Search results:")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:\n{doc.page_content}")
