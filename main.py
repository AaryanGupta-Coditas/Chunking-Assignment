import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_pdf_text(file_path: str):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f) #to read text from PDF files.
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text #complete text of the PDF as a string

def chunk_text(text: str, chunk_size: int = 300):
    #smaller chunks improve retrieval granularity
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks
#Breaks the text into smaller pieces of 300 characters each.
#Returns a list of chunks.

def main():
    pdf_text = load_pdf_text("sample.pdf")
    print(f"Total text length: {len(pdf_text)}")

    chunks = chunk_text(pdf_text)
    print(f"Total chunks: {len(chunks)}")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    print("Embeddings shape:", embeddings.shape)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    print("Total vectors in index:", index.ntotal)

    # query = "Explain about artificial intelligence"
    query  = input("Enter your query :")
    query_vector = model.encode([query])
    k = 3
    distances, indices = index.search(np.array(query_vector), k)

    print("\n Query Results:")
    for idx, dist in zip(indices[0], distances[0]):
        print(f"\nChunk #{idx} (Distance: {dist:.2f}):\n{chunks[idx][:200]}")

if __name__ == "__main__":
    main()
