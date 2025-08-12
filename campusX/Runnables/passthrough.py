from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence , RunnablePassthrough ,RunnableParallel
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm =  HuggingFaceEndpoint(
            repo_id= "mistralai/Mixtral-8x7B-Instruct-v0.1",task= "text-generation"
)
model = ChatHuggingFace(llm  = llm)

tempalte1 = PromptTemplate(
    template= "Your task is to generate a joke on the topic {topic}",
    input_variables= ['topic']
)

tempalte2 =  PromptTemplate(
    template= "your task is to summarise and expalin the joke {text} ",
    input_variables= ['text']
)
parser  =  StrOutputParser()

joke_gen_Chain = RunnableSequence(tempalte1 , model , parser)
joke_summ = RunnableSequence(tempalte2 , model , parser)

parallel_chain = RunnableParallel(
    {
        'joke' :  RunnablePassthrough(),
        'explain' : RunnableSequence(tempalte2 , model , parser)
    }
)

result_chain =  RunnableSequence(joke_gen_Chain , parallel_chain)
result_chain.invoke({'topic' : 'cricket'})

