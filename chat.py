import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("memory/index.faiss")

with open("memory/texts.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

print("Mini-ChatGPT ready. Type 'exit' to quit.\n")

while True:
    question = input("You: ")
    if question.lower() == "exit":
        break

    q_emb = embedder.encode([question])
    _, I = index.search(q_emb, k=2)

    context = " ".join([texts[i] for i in I[0]])

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the information below.

Context:
{context}

Question:
{question}

Answer:
"""

    response = generator(
        prompt,
        max_length=180,
        do_sample=True,
        temperature=0.7
    )

    print("\nAI:", response[0]["generated_text"].split("Answer:")[-1].strip(), "\n")
