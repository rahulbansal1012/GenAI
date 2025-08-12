from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint 
from langchain_core.prompts import PromptTemplate ,load_prompt
from dotenv import  load_dotenv
import streamlit as st
import os

load_dotenv()
# define the llm
llm = HuggingFaceEndpoint(
    repo_id= "mistralai/Mixtral-8x7B-Instruct-v0.1",
    task= "text-generation"
)
# initialize the Model
model = ChatHuggingFace(llm = llm)
# Define the UI
st.header("Research Tool")

paper_input = st.selectbox("Select Research paper" , ["Attention is all you need", "Word2Wec","GPT-3 Language Models are few-shot Learners"])
style_input =  st.selectbox("Select Explanation Style:" , ["Beginner-Friendly" , "Technical" , "Code-Oriented"])
length_input  = st.selectbox("Select Expalantion length" , ["Short (1-2 Paragraphs)" ,"Medium (2-3 Paragraph)"])
# user_input  = st.text_input("Enter Your prompt")

# Dynamic Prompt template

prompt_template  = load_prompt("template.json")

formated_prompt = prompt_template.format(paper_input =  {paper_input}, style_input = {style_input} , length_input = {length_input} )
print(formated_prompt)

if st.button("Summarize"):
# Invoking model on based of user input
    result  = model.invoke(formated_prompt)
    st.write(result.content)

# print(os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))
# print(result.content)