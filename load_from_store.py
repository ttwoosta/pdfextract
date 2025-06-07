from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Your sample text
text = """
The solar system consists of the Sun and everything bound to it by gravity.
This includes planets, moons, asteroids, comets, and meteoroids.
Earth is the third planet from the Sun and the only known planet to support life.
""" * 5


# Initialize HuggingFace embedding model (MiniLM is a good default)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# Store embeddings in FAISS
vectorstore = FAISS.load_local("sample_faiss_huggingface_index", embedding_model, allow_dangerous_deserialization=True)

# Run a test search
query = "What planet supports life?"
results = vectorstore.similarity_search(query, k=2)

# Show results
print("🔍 Search results:")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:\n{doc.page_content}")
