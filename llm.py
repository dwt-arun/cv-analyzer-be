from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from openai import OpenAI
# ---------------------------------------------
# LOAD MODELS / CLIENT
# ---------------------------------------------

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise Exception("OPENAI_API_KEY not found in environment variables.")

openai_base_url = os.getenv("OPENAI_BASE_URL", None)

openai_client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_base_url  
)

embedding_model = SentenceTransformer("all-mpnet-base-v2")