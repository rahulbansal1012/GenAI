from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
llm = HuggingFaceEndpoint(    repo_id= "mistralai/Mixtral-8x7B-Instruct-v0.1",
    task= "text-generation")

# Chat Template
chat_template = ChatPromptTemplate([
    ('system' , "you are a helpful customer support agent"),
    MessagesPlaceholder(variable_name= 'chat_history'),
    ('human' , "{query}")
])

chat_history = []
# load chat history

with open('chat_history.txt') as f :
    chat_history.extend(f.readlines())

# ccreate prompt

prompt = chat_template.invoke(
    {'chat_history' : chat_history ,"query" : "What is te update of refund"
     }
     )

print(prompt)
model = ChatHuggingFace(llm = llm)

