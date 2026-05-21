import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb


# =========================
# 1. Load API key
# =========================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing. Please add it to your .env file.")

client = OpenAI(api_key=api_key)


# =========================
# 2. Settings
# =========================

KNOWLEDGE_FILE = "knowledge_base.txt"
CHROMA_FOLDER = "chroma_db"
COLLECTION_NAME = "financial_market_knowledge"

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"


# =========================
# 3. Load knowledge base
# =========================

def load_knowledge_base(filename):
    """
    Reads the knowledge base text file.
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{filename} was not found. Make sure it is in the same folder as app.py."
        )


# =========================
# 4. Split text into chunks
# =========================

def chunk_text(text, chunk_size=500, overlap=80):
    """
    Splits long text into smaller chunks.

    chunk_size = how many characters per chunk
    overlap = repeated characters between chunks so context is not lost
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

    return chunks


# =========================
# 5. Create embeddings
# =========================

def create_embedding(text):
    """
    Converts text into a vector using OpenAI embeddings.
    A vector is a list of numbers that represents the meaning of the text.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


# =========================
# 6. Set up Chroma vector database
# =========================

def setup_vector_database(chunks):
    """
    Stores text chunks and their embeddings inside ChromaDB.
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_FOLDER)

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # Check if the database already has data
    existing_count = collection.count()

    if existing_count > 0:
        return collection

    print("Creating vector database. This may take a moment...")

    for index, chunk in enumerate(chunks):
        embedding = create_embedding(chunk)

        collection.add(
            ids=[f"chunk_{index}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"source": KNOWLEDGE_FILE, "chunk_number": index}]
        )

    print("Vector database created successfully.\n")

    return collection


# =========================
# 7. Retrieve relevant chunks
# =========================

def retrieve_relevant_chunks(question, collection, number_of_results=3):
    """
    Searches ChromaDB for the most relevant chunks.
    """
    question_embedding = create_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results
    )

    documents = results["documents"][0]
    return documents


# =========================
# 8. Generate final answer with LLM
# =========================

def generate_answer(question, retrieved_chunks):
    """
    Sends the user's question and retrieved document chunks to the LLM.
    The LLM writes a simple answer using only the retrieved information.
    """
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a beginner-friendly financial market tutor.

Answer the user's question using only the information in the context below.

Rules:
- Use simple language.
- Give a clear explanation.
- Do not give personal financial advice.
- Do not tell the user exactly what to buy or sell.
- If the answer is not in the context, say: "I do not have enough information in my knowledge base to answer that."

Context:
{context}

User question:
{question}
"""

    response = client.responses.create(
        model=LLM_MODEL,
        input=prompt
    )

    return response.output_text


# =========================
# 9. Main chatbot loop
# =========================

def main():
    text = load_knowledge_base(KNOWLEDGE_FILE)
    chunks = chunk_text(text)

    collection = setup_vector_database(chunks)

    print("Welcome to the Full RAG Financial Market Bot!")
    print("Ask me financial market questions in simple words.")
    print("Type 'exit' to stop the bot.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Bot: Goodbye!")
            break

        if question == "":
            print("Bot: Please type a question.\n")
            continue

        retrieved_chunks = retrieve_relevant_chunks(question, collection)

        answer = generate_answer(question, retrieved_chunks)

        print("\nBot:", answer)
        print()


if __name__ == "__main__":
    main()