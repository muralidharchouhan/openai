import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

#Load environment variables
load_dotenv("environment.env", override=True)

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

print("Initializing chroma db")
# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

print("cleaning started")
# To cleanup, you can delete the collection
test = vectordb.delete_collection()
print("cleaning complete")
