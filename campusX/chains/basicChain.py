from langchain_huggingface import HuggingFaceEndpoint ,  ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel,Field


load_dotenv()

llm = HuggingFaceEndpoint(repo_id= "mistralai/Mixtral-8x7B-Instruct-v0.1",task= "text-generation")

# class schema(BaseModel):
    # car_companies : list[str] = Field(description = "List out the car companies in a list of string")
    # content : dict[(str,str)] = Field(description="List out the financial and quaterly result of these company and provide in a dictory mapped to the car company")





model  = ChatHuggingFace(llm = llm)
parser  = StrOutputParser()
template1 =  PromptTemplate(
    template= "Your task is to generate detail on  {topic} " ,
    input_variables= ['topic'] ,
    # partial_variables= {"format_des": parser.get_format_instructions()}
)

template2 = PromptTemplate(
    template= "Your task is to summarized the {text}.",
    input_variables= ['text'],
    # partial_variables= {'format_des' : parser.get_format_instructions()}
)
# prompt  = template1.invoke("their financial analysis")
chain  = template1 | model | parser | template2 |model|parser
result = chain.invoke({'topic': 'GTA - 5'})
print(result)
print(type(result))
print(chain.get_graph().print_ascii())