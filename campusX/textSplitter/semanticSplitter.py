from langchain.text_splitter import CharacterTextSplitter 
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# loader  = PyPDFLoader(r"C:\Users\bansa\Desktop\GenAi\campusX\Cri.pdf")
# docs = loader.load()

huggicFaceEmbedder =  HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# print("Number of Docs:" , len(docs))
# print("page content::" , docs[2])

splitter  = SemanticChunker(
    huggicFaceEmbedder,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. 
The sun was bright, and the air smelled of earth and fresh grass. 
The Indian Premier League (IPL) is the biggest cricket league in the world. 
People all over the world watch the matches and cheer for their favourite teams.
"""

chunks = splitter.split_text(sample)
# res =  splitter.split_documents(docs)
# print(len(chunks))
# print("Type: " , type(chunks))
# print(chunks)
for i , chunk in enumerate(chunks):
    print(f"\nChunks {i+1}:\n{chunk}")