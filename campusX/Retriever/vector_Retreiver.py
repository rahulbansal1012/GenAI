from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever


my_docs = [
      Document(
        page_content="LangGraph is the best framework for stateful agents.",
        metadata={"source": "tweet"}
    ),
    Document(
        page_content="OpenAI released GPT-4 with vision capabilities.",
        metadata={"source": "news"}
    ),
]

embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-mpnet-base-v2"
)


vector_store =  Chroma.from_documents(
    collection_name= "My_Retriver",
    documents= my_docs ,
    embedding= embedding_model,
)

my_collection  = vector_store._collection
items = my_collection.get(
    include=['documents']
)
for i,doc in enumerate(items['documents']):
    print(f"data at {i+1}" , doc )

# here we have defined the retreiver
retreiver = vector_store.as_retriever()
query = "What is langchain"
result = retreiver.invoke(query)
print(result) 