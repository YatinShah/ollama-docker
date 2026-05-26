#good for large documents, since it uses dockument splitting with overlap
#pip install langchain-text-splitters
import uuid
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter



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


# Initialize your existing home lab clients
ollama_client = ollama.Client(host="http://192.168.1.50:11434")
qdrant_client = QdrantClient(url="http://192.168.1.50:6333")

# Define a large text document simulating home lab logs or wiki pages
large_document = """
LAB SERVER RUNBOOK - UPDATE 2026
Section 1: Network Configuration. All primary server hosts are mapped inside the 10.0.0.0/24 subnet. 
The main gateway router is managed at 10.0.0.1. The AI agent orchestration pipeline uses static IP 10.0.0.50.
Section 2: Database Backups. Postges and Qdrant volume backups are scheduled nightly at 02:00 AM.
All backup files are compressed into tarball assets and pushed to local NAS storage array mounted at /mnt/nas/backups.
Section 3: Emergency Recovery. In the event of a total hypervisor crash, hardware node 1 must be booted first.
"""

# Initialize the Splitter. It attempts to split by paragraphs (\n\n), then sentences (\n), then words ( ).
# This preserves structured information within the same context chunk.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,      # Small chunk size optimal for your fast 3B local model
    chunk_overlap=20     # 20-character safety margin so boundaries maintain contextual meaning
)

# Break the huge document into an iterable list of small text fragments
chunks = text_splitter.split_text(large_document)

print(f"📄 Processed document into {len(chunks)} optimized memory blocks.")

# Iterate and upload each chunk to Qdrant separately
for index, chunk_text in enumerate(chunks):
    print(f" -> Processing Chunk {index + 1}: '{chunk_text[:40]}...'")
    
    # Compute vector embedding for this precise snippet
    embed_res = ollama_client.embeddings(model="nomic-embed-text", prompt=chunk_text)
    vector = embed_res['embedding']
    
    # Store this fragment inside Qdrant
    qdrant_client.upsert(
        collection_name="agent_memories",
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk_text} # The agent retrieves this exact chunk later
            )
        ]
    )

print("✅ Server document successfully ingested into long-term memory!")

----same as example1.yp ---------
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
