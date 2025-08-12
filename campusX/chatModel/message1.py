from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage , AIMessage , HumanMessage
load_dotenv()

llm =  HuggingFaceEndpoint(

repo_id= "mistralai/Mixtral-8x7B-Instruct-v0.1",
task= "text-generation"

)

model  = ChatHuggingFace(llm =  llm)

# creating a history to store the context
chat_history = [
    SystemMessage(content="You are helpful AI assistant and generate response in one line.")
]
while True:
    user_input = input("User:")
    chat_history.append(HumanMessage(content=user_input))
    # print(type(chat_history))
    if user_input == "exit" :
        break
    result = model.invoke((chat_history))
    chat_history.append(AIMessage(content= result.content))
    print("AI :", result.content)