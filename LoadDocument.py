import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
 
#Load environment variables
load_dotenv("environment.env", override=True)
   
# method to load a pdf document and chunk 
def LoadDocument_And_Split():
    print("Document loading started")
    loader = PyPDFLoader(r'Power Platform Licensing Guide October 2023.pdf')
    pages = loader.load_and_split()

    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = text_splitter.split_documents(pages)
    print("Document chunking complete")
    return texts

texts = LoadDocument_And_Split()

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'
print("Uploading document chunks along with embedding to Chroma vector db")
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
print("Upload completed")
vectordb.persist()


