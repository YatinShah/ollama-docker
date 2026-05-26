#pip install ollama qdrant-client

# not sogood example for very large docs!
import uuid
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# --- CONFIGURATION ---
OLLAMA_URL = "http://192.168.1.50:11434"
QDRANT_URL = "http://192.168.1.50:6333"
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "agent_memories"

# --- INITIALIZATION ---
# Connect to the local services deployed via Docker Compose
ollama_client = ollama.Client(host=OLLAMA_URL)
qdrant_client = QdrantClient(url=QDRANT_URL)

# Ensure the memory vault exists in Qdrant. 
# nomic-embed-text outputs 768-dimensional vectors.
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

def query_agent_with_memory(user_input: str) -> str:
    # 1. Generate an embedding vector for the incoming user message
    embed_response = ollama_client.embeddings(model=EMBED_MODEL, prompt=user_input)
    query_vector = embed_response['embedding']
    
    # 2. Query Qdrant for the top 2 most semantically relevant past memories
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=2
    )
    
    # Extract text from match payloads into a single context string
    historical_context = "\n".join([res.payload["text"] for res in search_results]) if search_results else "None"
    
    # 3. Construct an augmented prompt feeding historical context to the LLM
    system_prompt = (
        "You are an AI Agent with long-term memory. Use the retrieved historical context "
        "to inform your answers. If the context is irrelevant, ignore it.\n"
        f"--- RELEVANT PAST MEMORIES ---\n{historical_context}\n-------------------------"
    )
    
    print(f"💡 [System] Retrieved Context: {historical_context[:60]}...")
    
    # 4. Generate answer from Ollama
    response = ollama_client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    agent_reply = response['message']['content']
    
    # 5. Commit the new user statement into long-term memory for next time
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()), # Generate a unique ID for the memory point
                vector=query_vector,
                payload={"text": f"User said: {user_input} -> Agent replied: {agent_reply}"}
            )
        ]
    )
    
    return agent_reply

# --- TEST THE PIPELINE ---
if __name__ == "__main__":
    print("Conversation 1:")
    print(query_agent_with_memory("Remember this secret password: 'XC-998'."))
    
    print("\nConversation 2 (Testing long-term recall):")
    print(query_agent_with_memory("What was that secret password I told you about?"))
