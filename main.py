import os
import logging
import httpx
import mysql.connector
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load env variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ecommerce AI Service", description="AI components for embeddings, vector search, and LLM chat.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "ecommerce-index")

OSS_LLM_URL = os.getenv("CUSTOM_API_BASE")
OSS_LLM_MODEL = os.getenv("CUSTOM_MODEL", "gpt-oss:20b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")

# --- MySQL Config ---
MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_PORT = int(os.getenv("DB_PORT", 3306))
MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD")
MYSQL_DB = os.getenv("DB_NAME")
MYSQL_TABLE = os.getenv("MYSQL_TABLE", "amazon_products")
MYSQL_ID_COLUMN = os.getenv("MYSQL_ID_COLUMN", "asin")
MYSQL_TITLE_COLUMN = os.getenv("MYSQL_TITLE_COLUMN", "title")

# Initialize models and clients
logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
pc = Pinecone(api_key=PINECONE_API_KEY)
llm_client = OpenAI(base_url=OSS_LLM_URL, api_key=os.getenv("CUSTOM_API_KEY", "dummy")) if OSS_LLM_URL else None

@app.get("/health")
def health():
    return {"status": "ok", "service": "ai_service", "model": EMBEDDING_MODEL_NAME}

@app.post("/embed-description")
async def embed_description(
    product_id: str = Body(...), 
    description: str = Body(...)
):
    """
    Generate embedding for a single product description and store in Pinecone.
    Payload: { "product_id": "ASIN123", "description": "High-quality leather bag..." }
    """
    if not product_id or not description:
        raise HTTPException(status_code=400, detail="Missing product_id or description")
    
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # 1. Generate embedding from description string
        embedding = embedder.encode(description).tolist()
        
        # 2. Store in Pinecone with minimal metadata
        index.upsert(vectors=[{
            "id": product_id,
            "values": embedding,
            "metadata": {"id": product_id} # Minimal metadata to keep it valid
        }])
        
        logger.info(f"Successfully embedded description for product: {product_id}")
        return {"message": "Embedding created and stored", "product_id": product_id}
    
    except Exception as e:
        logger.error(f"Error in embed_description: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed-titles-bulk")
async def embed_titles_bulk():
    """
    Fetch first 1000 product titles from MySQL, generate embeddings, and store in Pinecone.
    No request body needed — reads DB config from environment variables.
    """
    try:
        # 1. Connect to MySQL and fetch 1000 products distributed across categories
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor(dictionary=True)

        # Fetch 50 products from each of the top 20 categories for diversity
        top_categories = (91,84,270,114,118,95,110,112,116,123,105,122,173,120,113,97,88,107,51,228)
        rows = []
        for cat_id in top_categories:
            cursor.execute(
                f"SELECT `{MYSQL_ID_COLUMN}`, `{MYSQL_TITLE_COLUMN}` FROM `{MYSQL_TABLE}` "
                f"WHERE category_id = %s AND `{MYSQL_TITLE_COLUMN}` IS NOT NULL AND `{MYSQL_TITLE_COLUMN}` != '' "
                f"LIMIT 50",
                (cat_id,)
            )
            rows.extend(cursor.fetchall())

        rows = rows[:1000]  # Ensure max 1000
        cursor.close()
        conn.close()

        if not rows:
            raise HTTPException(status_code=404, detail="No products found in database")

        logger.info(f"Fetched {len(rows)} products from MySQL for bulk embedding")

        # 2. Generate embeddings in batch for efficiency
        titles = [str(row[MYSQL_TITLE_COLUMN]) for row in rows]
        product_ids = [str(row[MYSQL_ID_COLUMN]) for row in rows]
        embeddings = embedder.encode(titles, batch_size=64, show_progress_bar=False).tolist()

        # 3. Ensure Pinecone index exists, create if not
        existing_indexes = [i.name for i in pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing_indexes:
            logger.info(f"Index '{PINECONE_INDEX_NAME}' not found. Creating...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=768,  # all-mpnet-base-v2 outputs 768-dim vectors
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"Index '{PINECONE_INDEX_NAME}' created successfully.")
        # 4. Upsert into Pinecone in batches of 100
        index = pc.Index(PINECONE_INDEX_NAME)
        batch_size = 100
        upserted_count = 0

        for i in range(0, len(embeddings), batch_size):
            batch = [
                {
                    "id": product_ids[j],
                    "values": embeddings[j],
                    "metadata": {"id": product_ids[j], "title": titles[j]}
                }
                for j in range(i, min(i + batch_size, len(embeddings)))
            ]
            index.upsert(vectors=batch)
            upserted_count += len(batch)
            logger.info(f"Upserted batch {i // batch_size + 1}: {upserted_count}/{len(embeddings)} vectors")
        return {
            "message": "Bulk title embedding complete",
            "total_fetched": len(rows),
            "total_upserted": upserted_count
        }

    except mysql.connector.Error as db_err:
        logger.error(f"MySQL error in embed_titles_bulk: {db_err}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(db_err)}")
    except Exception as e:
        logger.error(f"Error in embed_titles_bulk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(message: str = Body(..., embed=True)):
    """
    Use the GPT OSS model for chat.
    """
    if not llm_client:
        raise HTTPException(status_code=500, detail="OSS LLM not configured")
    
    try:
        response = llm_client.chat.completions.create(
            model=OSS_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for an ecommerce platform."},
                {"role": "user", "content": message}
            ]
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search(query: str = Query(...), top_k: int = Query(default=10)):
    """
    Semantic search: Query -> Embedding -> Pinecone Search.
    """
    try:
        query_vector = embedder.encode(query).tolist()
        index = pc.Index(PINECONE_INDEX_NAME)
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        
        matches = []
        for match in results.get("matches", []):
            matches.append({
                "product_id": match.id,
                "score": match.score,
                "metadata": match.metadata
            })
            
        return {"query": query, "matches": matches}
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embedded-categories")
async def embedded_categories():
    """
    Returns all distinct category names of products currently embedded in Pinecone.
    Uses a dummy vector search to fetch stored ASINs, then cross-references with MySQL.
    """
    try:
        # 1. Use a zero vector to fetch up to 1000 IDs from Pinecone
        index = pc.Index(PINECONE_INDEX_NAME)
        dummy_vector = [0.0] * 768
        results = index.query(vector=dummy_vector, top_k=1000, include_metadata=False)
        asin_list = [match.id for match in results.get("matches", [])]

        if not asin_list:
            return {"total_embedded": 0, "categories": []}

        # 2. Cross-reference with MySQL to get category names
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor(dictionary=True)
        placeholders = ",".join(["%s"] * len(asin_list))
        cursor.execute(f"""
            SELECT DISTINCT c.id as category_id, c.category_name
            FROM amazon_products p
            JOIN amazon_categories c ON p.category_id = c.id
            WHERE p.asin IN ({placeholders})
            ORDER BY c.category_name
        """, asin_list)
        categories = cursor.fetchall()
        cursor.close()
        conn.close()

        return {
            "total_embedded": len(asin_list),
            "total_categories": len(categories),
            "categories": categories
        }

    except Exception as e:
        logger.error(f"Error in embedded_categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)