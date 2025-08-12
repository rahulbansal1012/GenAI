from langchain_community.document_loaders import PyPDFLoader
import os 
from pypdf import PdfReader
print(os.getcwd())
print("is File Exist " , os.path.exists("Cri.pdf"))

loader =  PyPDFLoader(r"C:\Users\bansa\Desktop\GenAi\campusX\Cri.pdf")
docs = loader.load()

print((docs))