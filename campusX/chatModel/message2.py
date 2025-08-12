# Solving the problem of context of caht 

from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage , SystemMessage , AIMessage

load_dotenv()

llm =  HuggingFaceEndpoint(
        repo_id= "mistralai/Mixtral-8x7B-Instruct-v0.1",
    task= "text-generation"
)

model = ChatHuggingFace(llm =llm)

messages = [
    SystemMessage(content= "You are a Helpful Assistant generate an crisp and short Answere"),
    HumanMessage(content= "Tell me about the Langchain")

]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)