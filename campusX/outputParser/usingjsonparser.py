from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

llm  = HuggingFaceEndpoint(  repo_id= "mistralai/Mixtral-8x7B-Instruct-v0.1",task= "text-generation")

model = ChatHuggingFace(llm = llm)
parser  =  JsonOutputParser()

template1 = PromptTemplate(
    template="your task is to provide content on topic: {topic} \n generate output in following json format: {format_instruction}"
    ,input_variables= ['topic'] ,
    partial_variables= {'format_instruction' : parser.get_format_instructions()}
)

template2 = PromptTemplate(
    template= "Your task is to summarized the text: {text} in 3 lines.\n generate output in following json format :  {format_instruction}",
    input_variables= ['text'],
    partial_variables= {"format_instruction" : parser.get_format_instructions()}
    )


# === Step-by-step ===
# Step 1: Prompt 1 → model → parser
prompt1_text = template1.format(topic="Sports Cars In India")
llm_output1 = model.invoke(prompt1_text)
parsed_result1 = parser.invoke(llm_output1)

print("First parsed output:\n", parsed_result1)

# Let's say we want to summarize the "introduction" field
intro_text = parsed_result1.get("introduction", "")

# Step 2: Prompt 2 → model → parser
prompt2_text = template2.format(text=intro_text)
llm_output2 = model.invoke(prompt2_text)
parsed_result2 = parser.invoke(llm_output2)

print("\nFinal summarized result:\n", parsed_result2)

# chain = template1 | model | parser | template2 | model | parser

# print("Template1 : " ,template1)
# print("Template2 : ",template2)
# result  = chain.invoke({
#     "topic" : "Sports Cars In India"
# })

# print(result)
# print(type(result))

