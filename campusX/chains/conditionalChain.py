from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel,Field
load_dotenv()

llm = HuggingFaceEndpoint(
        repo_id= "mistralai/Mixtral-8x7B-Instruct-v0.1",task= "text-generation"
)

class schema(BaseModel):
    Sentiment: str = Field(description= "Categorized Sentiment as positive or negative.") 
    response :  str =  Field(description= "Describe the response in few words")
    

model  = ChatHuggingFace(llm = llm)
parser  = PydanticOutputParser(pydantic_object= schema)

temalte1 =  PromptTemplate(
    template="Classify the Sentiemtn of the following Feedback text in the feedbck text either positive or negative \n {feedback} \n {format_inst}"
    ,input_variables= ['feedback'],
    partial_variables= {'format_inst' : parser.get_format_instructions()}
)

classifier_chain =  temalte1 | model  | parser

template2 = PromptTemplate(
    template= "Write an appropriate resposne to positive feedback \n {feedback} and output format is  {inst_format} "
    ,
    partial_variables= {'inst_format' : parser.get_format_instructions}
    , 
    input_variables= ['feedback']
)


template3 = PromptTemplate(
    template= "Write an appropriate resposne to Negative feedback \n {feedback} and output format is  {inst_format} "
    ,
    partial_variables= {'inst_format' : parser.get_format_instructions}
    , 
    input_variables= ['feedback']
)

branch_Chain = RunnableBranch(
    (lambda x : x.Sentiment == 'positive' , template2 | model | parser),
    (lambda x : x.Sentiment == 'negative' , template3 | model | parser),
    RunnableLambda(lambda x : "could not determine sentiment")
)

chain = classifier_chain  | branch_Chain

result = chain.invoke({'feedback' : "Car is good with mileage and comfortable"})
print(result)