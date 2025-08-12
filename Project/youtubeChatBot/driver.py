from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings,ChatHuggingFace ,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()
# fucntion for Extracting text from the transcript
def fetched_transcript_Formater(transcript):
    output_string = " "
    for i in transcript_output :
        output_string+= (" " + i.get('text'))
    return output_string.replace("\n" , " ")

# Creatring the instance of Trancript Class
transcripter = YouTubeTranscriptApi()
video_id_inputted  = "1bUy-1hGZpI" 

# Data Extraction ->Data loading
transcript_output =  transcripter.fetch(
video_id= video_id_inputted,
languages=['hi', 'en']
).to_raw_data()

formated_transcript = fetched_transcript_Formater(transcript_output)
# print("Transcript :: ", type(formated_transcript))

#Data Indexing/Splitter 

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 200,
     
)
chunks = splitter.create_documents([formated_transcript])
# print("chunks:: ", chunks)
print("length of chunks" , len(chunks))

# Create Embedding of these chunks
embedding_model = HuggingFaceEmbeddings(
   model_name = "sentence-transformers/all-mpnet-base-v2"
)

# Design the vector DB and adding chunks to it 
vectorDb  = Chroma.from_documents(
    collection_name= "myYtDb",
    embedding = embedding_model,
    documents= chunks
)

db_data = vectorDb.get(
    include= ['documents', "embeddings"]
)
# print("generated_embedding :: ", db_data)
print("embedding Generated->")

# Retriever Stage:
retriever = vectorDb.as_retriever(
    search_type = 'similarity',
    search_kwargs = {'k' : 4}
)


#Augmentation Stage:

tempalte = PromptTemplate(
    template=  """
    You are a helpful AI Assistant.
    Answere only from the transcript context.
    If context is insufficent just prompt i am unable to figure out amswere.
    {context}
    question : {question}
    """ ,
    input_variables= ['context' , 'question']
)
user_input_question = input("Enter the question please : ")
retrieved_docs = retriever.invoke(user_input_question)

# Formating retreived Docs:
formated_page_content = ""
for i, doc in enumerate(retrieved_docs):
    content = (doc.page_content)
    formated_page_content += content

prompt  = tempalte.invoke({'context' : formated_page_content , 'question' : user_input_question })

# print("Retrieved Doc::" , retrieved_docs)
print("Document Retrieved -> ")
print("prompt::" , prompt)

llm  = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)
model = ChatHuggingFace(
    llm = llm 
)
output_llm  = model.invoke(prompt)
print("Results :", output_llm)


    

# print(formated_page_content)