from langchain_huggingface import HuggingFaceEndpoint ,ChatHuggingFace 
# from langchain_core import 
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mixtral-8x7B-Instruct-v0.1",
    task= "text-generation",
    temperature= 0.5
)
# design the schema
class Review(TypedDict):
    summary  : str
    sentiment : str

model = ChatHuggingFace(llm = llm)
structured_model =  model.with_structured_output(Review)



result =  structured_model.invoke("""The hardware is great but the software feels bloated. There are too mant pre-installed apps that I cant remove. Also the UI looks outdated compared to other brands. Hoping for a software update to fix that.""")

print(result)