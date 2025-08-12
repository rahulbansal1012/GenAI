from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader  = PyPDFLoader(r"C:\Users\bansa\Desktop\GenAi\campusX\Cri.pdf")
docs = loader.load()

print("Number of Docs:" , len(docs))
# print("page content::" , docs[2])

splitter  = CharacterTextSplitter(
    chunk_size =  100 ,
    chunk_overlap = 0 ,
    separator= ''
)
res =  splitter.split_documents(docs)
print(len(res))
print("Type: " , type(res))
print(res[3])