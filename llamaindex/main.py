from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os 
from dotenv import load_dotenv
import sys
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")
# print(os.getenv("API_KEY"))
# os.environ["OPENAI_BASE_URL"] = os.getenv("API_URL")
# base_url = "https://api.groq.com/openai/v1/"

# embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = OpenAI(
    model = "mistral-saba-24b",
    api_key = os.getenv("API_KEY"),
    base_url = os.getenv("API_URL")
)


doc = SimpleDirectoryReader("my_doc").load_data()

index = VectorStoreIndex.from_documents(doc)

query_engine = index.as_query_engine()

response = query_engine.query("What is the Return Policy")

print(response)