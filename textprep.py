from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import sys
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Your long text (e.g., extracted from a PDF)
# Ensure the user provided a PDF path
if len(sys.argv) < 2:
    print("Usage: python extract_pdf.py <path_to_pdf>")
    sys.exit(1)

# Get the PDF path from the first command-line argument
txt_path = sys.argv[1]

# Validate the file
if not os.path.isfile(txt_path):
    print(f"Error: File not found: {txt_path}")
    sys.exit(1)

with open(txt_path, "r", encoding="utf-8") as f:
    text = f.read()
    

# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # Ideal size for embeddings
    chunk_overlap=100,  # Helps preserve context across chunks
)

# Split the document into chunks
chunks = text_splitter.split_text(text)

# Print a few chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"--- Chunk {i+1} ---")
    print(chunk)



# Generate embeddings
embeddings = OpenAIEmbeddings()
doc_embeddings = embeddings.embed_documents(chunks)

# Store in FAISS (or another vector store)
vectorstore = FAISS.from_texts(chunks, embeddings)

# Save to local folder
vectorstore.save_local("main_faiss_huggingface_index")