from sentence_transformers import SentenceTransformer
import faiss

with open("data/knowledge.txt", "r", encoding="utf-8") as f:
    texts = [t.strip() for t in f if t.strip()]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "memory/index.faiss")

with open("memory/texts.txt", "w", encoding="utf-8") as f:
    for t in texts:
        f.write(t + "\n")

print("Memory built.")
