#pip install ollama qdrant-client
#Hybrid Document Store Complete Python ScriptThis end-to-end operational script creates an internal SQLite database for raw storage and pairs it with Qdrant for semantic search index lookups.
# this is a general enterprise grade approach for large docs.

import os
import sqlite3
import uuid
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# --- CONFIGURATION ---
OLLAMA_URL = "http://192.168.1.50:11434"
QDRANT_URL = "http://192.168.1.50:6333"
VECTOR_COLLECTION = "hybrid_indices"
SQLITE_DB_PATH = "local_doc_store.db"

# --- SERVICE INITIALIZATION ---
ollama_client = ollama.Client(host=OLLAMA_URL)
qdrant_client = QdrantClient(url=QDRANT_URL)

# 1. Initialize SQLite Relational Database for large text block lookups
sql_conn = sqlite3.connect(SQLITE_DB_PATH)
cursor = sql_conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_store (
        chunk_id TEXT PRIMARY KEY,
        large_context TEXT,
        source_name TEXT
    )
""")
sql_conn.commit()

# 2. Initialize Qdrant Collection for dense geometric vectors (768-dim)
if not qdrant_client.collection_exists(VECTOR_COLLECTION):
    qdrant_client.create_collection(
        collection_name=VECTOR_COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )


# --- CORE PIPELINE FUNCTIONS ---

def ingest_document(large_context_block: str, search_snippet: str, source: str):
    """
    Saves the data using Strategy 3: 
    Vectors go to Qdrant, heavy raw text blocks go to SQLite.
    """
    # Generate a unique key joining both database entries
    shared_uuid = str(uuid.uuid4())
    
    # Step A: Insert massive raw context block safely into relational SQLite
    cursor.execute(
        "INSERT INTO document_store (chunk_id, large_context, source_name) VALUES (?, ?, ?)",
        (shared_uuid, large_context_block, source)
    )
    sql_conn.commit()
    
    # Step B: Generate vector math ONLY for the small, highly dense search snippet
    embed_res = ollama_client.embeddings(model="nomic-embed-text", prompt=search_snippet)
    vector = embed_res['embedding']
    
    # Step C: Upload vector and the pointer ID to Qdrant
    qdrant_client.upsert(
        collection_name=VECTOR_COLLECTION,
        points=[
            PointStruct(
                id=shared_uuid, # Match SQLite Primary Key
                vector=vector,
                payload={"snippet_preview": search_snippet} # Keep payloads featherweight
            )
        ]
    )
    print(f" Saved index pointer {shared_uuid} to Qdrant & Full Text to SQLite.")


def retrieve_large_context(user_query: str) -> str:
    """
    Queries Qdrant for the ID, then runs a rapid local lookup inside SQLite
    to extract the heavy narrative text block.
    """
    # 1. Embed query
    embed_res = ollama_client.embeddings(model="nomic-embed-text", prompt=user_query)
    query_vector = embed_res['embedding']
    
    # 2. Search vector database for the top entry match
    search_results = qdrant_client.search(
        collection_name=VECTOR_COLLECTION,
        query_vector=query_vector,
        limit=1
    )
    
    if not search_results:
        return "No corresponding memories found."
        
    # Extract the pointer ID string
    matched_id = search_results[0].id
    print(f"🎯 Match found in Qdrant! ID: {matched_id} (Score: {search_results[0].score:.4f})")
    
    # 3. Pull heavy context structure from local SQLite instantly using indexing
    cursor.execute("SELECT large_context FROM document_store WHERE chunk_id = ?", (matched_id,))
    row = cursor.fetchone()
    
    return row[0] if row else "Context could not be reconstructed from database."


# --- SIMULATED EXECUTION ---
if __name__ == "__main__":
    print("--- INGESTION PHASE ---")
    
    # Large context block we want returned to the user or agent model
    full_chapter = (
        "PROXMOX CLUSTER MAINTENANCE GUIDE\n"
        "To perform structural partition upgrades on the main storage pool, engineers must run "
        "the utility scripts located in /opt/proxmox/scripts/rebuild_zfs.sh. "
        "CRITICAL WARNING: Make sure all running agent micro-containers are paused using "
        "'docker compose stop' prior to running this script, or database file corruption WILL occur. "
        "The estimated recovery downtime window for this disk array operation spans 45 minutes."
    )
    
    # Tiny, highly targeted string used for mathematical embedding search
    targeted_hook = "How to run upgrade scripts on Proxmox ZFS storage arrays"
    
    ingest_document(
        large_context_block=full_chapter, 
        search_snippet=targeted_hook, 
        source="proxmox_guide.txt"
    )
    
    print("\n--- RETRIEVAL PHASE ---")
    query = "What happens if I forget to stop containers during storage upgrades?"
    print(f"Querying system: '{query}'\n")
    
    # Fetch massive surrounding context blocks cleanly
    resulting_context = retrieve_large_context(query)
    
    print("\n[Retrieved Big Data Block Results]:")
    print(resulting_context)
    
    # Close connections on script exit
    sql_conn.close()

