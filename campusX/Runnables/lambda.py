from langchain_huggingface  import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence , RunnableLambda , RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = HuggingFaceEndpoint(
            repo_id= "mistralai/Mixtral-8x7B-Instruct-v0.1",task= "text-generation"

)

model = ChatHuggingFace(llm = llm)
 
parser =  StrOutputParser()
template1 = PromptTemplate(
    template= "Your task is to generate a joke on the topic {topic}",
    input_variables= ['topic']
)
def counter(text):
    return len(text.split())

joke_gen_Chain =  RunnableSequence(template1 , model , parser)
parallel_Chain = RunnableParallel(
    {'joke' : RunnablePassthrough() , 
    'word_count' : RunnableLambda(counter)}
)
finalChain = RunnableSequence(joke_gen_Chain, parallel_Chain )

print(finalChain.invoke({'topic' : 'AI'}))