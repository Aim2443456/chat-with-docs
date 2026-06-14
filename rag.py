import os
import pymupdf
import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded!")

def extract_text(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages = []
    page_count = len(doc)
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"page": i+1, "text": text})
    doc.close()
    print(f"Extracted text from {page_count} pages")
    return pages

def chunk_pages(pages, chunk_size=200):
    chunks = []
    for page in pages:
        words = page["text"].split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "page": page["page"]
                })
    print(f"Created {len(chunks)} chunks")
    return chunks

def create_vector_db(chunks):
    print("Creating embeddings...")
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"Vector database created with {index.ntotal} vectors")
    return index, chunks

def search(query, index, chunks, top_k=3):
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            results.append({
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "distance": distances[0][i]
            })
    return results

def ask(question, index, chunks):
    relevant_chunks = search(question, index, chunks)
    context = ""
    citations = []
    for chunk in relevant_chunks:
        context += f"\n[Source - Page {chunk['page']}]:\n{chunk['text']}\n"
        citations.append(f"Page {chunk['page']}")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant that answers questions
based ONLY on the provided document context.
If the answer is not in the context, say 'I could not find that in the document.'
Always mention which page your answer comes from."""
            },
            {
                "role": "user",
                "content": f"Context from document:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    answer = response.choices[0].message.content
    return answer, citations

def load_document(pdf_path):
    pdf_path = pdf_path.strip().strip('"')
    if not os.path.exists(pdf_path):
        print(f"Error: File not found - {pdf_path}")
        return None, None
    if not pdf_path.lower().endswith(".pdf"):
        print("Error: File must be a PDF")
        return None, None
    pages = extract_text(pdf_path)
    if not pages:
        print("No text found in PDF")
        return None, None
    chunks = chunk_pages(pages)
    index, chunks = create_vector_db(chunks)
    return index, chunks

def main():
    print("\nCHAT WITH YOUR DOCS - RAG System")
    print("=" * 50)
    print("Upload any PDF and chat with it!")
    print("Type 'quit' to exit.")
    print("Type 'new' to load a different document.")
    print("=" * 50)

    index = None
    chunks = None
    current_doc = None

    while True:
        if index is None:
            print("\nHow would you like to load your PDF?")
            print("1. Type the full file path")
            print("2. Type just the filename (if PDF is in this folder)")
            print("3. Quit")

            choice = input("\nEnter choice (1-3): ").strip()

            if choice == "1":
                pdf_path = input("\nEnter full file path:\n> ").strip()
                pdf_path = pdf_path.strip('"')

            elif choice == "2":
                filename = input("\nEnter filename (include .pdf):\n> ").strip()
                pdf_path = os.path.join(os.getcwd(), filename)

            elif choice == "3":
                print("Goodbye!")
                break

            else:
                print("Invalid choice. Try again.")
                continue

            print(f"\nLoading document: {pdf_path}")
            index, chunks = load_document(pdf_path)

            if index is None:
                print("Failed to load document. Try again.")
                continue

            current_doc = os.path.basename(pdf_path)
            print(f"\nDocument loaded: {current_doc}")
            print("You can now ask questions about it.")
            print("Type 'new' to load a different document.")
            print("Type 'quit' to exit.")
            print("=" * 50)

        question = input("\nYour question: ").strip()

        if question.lower() == "quit":
            print("Goodbye!")
            break

        elif question.lower() == "new":
            index = None
            chunks = None
            current_doc = None
            print("\nLoading new document...")
            continue

        elif not question:
            continue

        print("\nSearching document...")
        answer, citations = ask(question, index, chunks)

        print(f"\nAnswer:\n{answer}")
        print(f"\nSources: {', '.join(set(citations))}")
        print("-" * 50)

if __name__ == "__main__":
    main()
