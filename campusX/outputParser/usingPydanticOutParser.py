from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen3-Coder-480B-A35B-Instruct",task= "text-generation"
)

model = ChatHuggingFace(llm = llm)



class person(BaseModel):
    name : str = Field(description = "Name of the person")
    age : int = Field(description = "Age o the Person")
    city : str = Field(description = "Name of city the person belongs to")


parser  = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template= "You task is to generate a fictinal {topic}  which will have age city and name   \n and provide output in following format {format_description}",
    input_variables= ['topic'],
    partial_variables= {'format_description' : parser.get_format_instructions()} 
   
)
prompt = template.invoke({'topic' : 'indian'})
print("Prompt :" , prompt)
res =  model.invoke(prompt)
final_res =  parser.parse(res.content)

print(final_res)
