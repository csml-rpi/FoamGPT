from langchain_community.vectorstores import FAISS
import argparse
from pathlib import Path
from langchain_community.embeddings.openai import OpenAIEmbeddings

# Step 1: Parse command-line arguments
parser = argparse.ArgumentParser(description="Process OpenFOAM case data and store embeddings in FAISS.")
parser.add_argument("--db_name", type=str, required=True, help="Name of the FAISS database to retrieve from")
parser.add_argument("--db_path", type=str, default=str(Path(__file__).resolve().parent.parent),
                    help="Path to the database directory (default: '../database')")

args = parser.parse_args()

database_path = args.db_path  # Get the database path from arguments


# Step 1: Define the path to the FAISS database
persist_directory = f"{database_path}/faiss/{args.db_name}"

# Step 2: Load the FAISS database
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)

# Step 3: Retrieve all stored documents
documents = vectordb.docstore._dict.values()  # Extract stored documents

# Step 4: Print the contents
print(f"📂 Loaded {len(documents)} documents from the FAISS database.\n")

for i, doc in enumerate(documents):
    if i > 10:
        break
    print(f"Document {i + 1}:")
    print(f"Page Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("-" * 80)
