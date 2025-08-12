from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document 

embedding_model =  HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-mpnet-base-v2"

)

vector_store =  Chroma(
    collection_name= "Testing_collection" , 
    embedding_function= embedding_model
)

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

res = vector_store.add_documents(documents = my_docs)

collection = vector_store._collection
print("Collections::" , collection)

items = collection.get(include=['documents'])
print("type of Item" , type(items))
print("Embeddings::" , items)

print(res)
for i, doc in enumerate(items["documents"]):
    print(f"Document {i+1}: {doc}")