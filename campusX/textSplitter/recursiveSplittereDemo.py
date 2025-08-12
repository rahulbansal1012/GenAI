from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader  = PyPDFLoader(r"C:\Users\bansa\Desktop\GenAi\campusX\Cri.pdf")
docs = loader.load()

print("Number of Docs:" , len(docs))
# print("page content::" , docs[2])

splitter  = RecursiveCharacterTextSplitter(
    chunk_size =  300 ,
    chunk_overlap = 0 ,
    
)
res =  splitter.split_documents(docs)
print(len(res))
print("Type: " , type(res))
print(res[3])