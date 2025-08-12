from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader

loader =  DirectoryLoader(
    path = r'C:\Users\bansa\Desktop\harryPotter-All',
    glob = '*.pdf',
    loader_cls= PyPDFLoader
)

docs  = loader.lazy_load()
# print(len(docs))
print(docs)
for i, doc in enumerate(docs):
    print(f"Document #{i+1}")
    print(doc.page_content)
    if i == 2:  # Only show first 3 documents
        break