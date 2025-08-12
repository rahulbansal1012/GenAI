from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
llm = HuggingFaceEndpoint(
      repo_id= "mistralai/Mixtral-8x7B-Instruct-v0.1",
    task= "text-generation",
)

model = ChatHuggingFace(llm = llm)
# Define the Prompt Template :: 
template1 =  PromptTemplate(
    template= "You task is to generate a summised version on the topic : {topic}" ,
    input_variables = ['topic']
)

template2 = PromptTemplate(
    template="you task is summarized these text: {text}  in 2 lines"
    ,input_variables=['text']
)

# Define the parser
parser =  StrOutputParser() 
# define chain
chain  = template1 | model | parser | template2 | model | parser

result  = chain.invoke({"topic" : 'Fast Food In India'})
print(result)
print(type(result))