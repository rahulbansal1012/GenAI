from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm  =  HuggingFaceEndpoint(
repo_id= "mistralai/Mixtral-8x7B-Instruct-v0.1",task= "text-generation"
)
model = ChatHuggingFace(llm = llm)
loader  = TextLoader("cricket.txt" ,encoding='utf-8')
template =  PromptTemplate(
    template= "Your task is to summarize the text : {text}"
)


docs = loader.load()
print(docs[0])
print(type(docs))
print(type(docs[0]))

parser =  StrOutputParser()

chain  =  template | model | parser
res = chain.invoke(docs[0].page_content)
print(res)