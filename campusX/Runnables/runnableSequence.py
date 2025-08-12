from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

llm  = HuggingFaceEndpoint(
            repo_id= "mistralai/Mixtral-8x7B-Instruct-v0.1",task= "text-generation"
)

model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(
    template= "Your Task is to generate the Random joke on {topic}\n"
    ,input_variables= ['topic']
)

template2 = PromptTemplate(
    template= "Your task is to summarise the joke {text}" ,
    input_variables= ['text']
)

parser =  StrOutputParser()

chain  =  RunnableSequence(template1 , model , parser , template2 , model , parser)
print(chain.invoke({'topic' : 'maths'}) )