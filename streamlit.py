import os
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

st.title("BRAIN Demo")
load_dotenv("environment.env", override=True)

openai.api_key = os.environ["OPENAI_API_KEY"]

persist_directory = 'db'
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    embedding_vector = embedding.embed_query(prompt)
    docs = vectordb.similarity_search_by_vector(embedding_vector)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        retriever = vectordb.as_retriever()
        template = """Answer the question based only on the following context:

        {context}

        Question: {question}
        """
        promptlang = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI()

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | promptlang
            | model
            | StrOutputParser()
        )


        query = prompt
        print(prompt)
        full_response = chain.invoke(query)
        
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
