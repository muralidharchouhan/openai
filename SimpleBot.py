import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv


load_dotenv("environment.env", override=True)

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
while True:
    print("Question: ")
    query = input()
    # docs = vectordb.similarity_search(query)
    embedding_vector = embedding.embed_query(query)
    docs = vectordb.similarity_search_by_vector(embedding_vector)
    
    retriever = vectordb.as_retriever()
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    print(chain.invoke(query))
    

    