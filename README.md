A simple chatbot using OpenAI, Langchain, ChromaDB (Vector DB), streamlit and Python. We will use locally persistnet directory for storing chroma db. 

Usecase: A bot that can answer any licensing questions from a specific document(s).   

Run the code in following order:
1. Install required packages using install.bat
2. Use LoadDocument.py for loading a sample pdf file to Chroma DB
3. You can either use SimpleBot.py or streamlit.py to create a chatbot experience that can answer questions from the document. Only difference is streamlit.py will generate GUI for chatbot.
4. Finally run Cleanup.py to clear the contents of locally stored chroma db. 

Here are some links to everything we used in this exercise,

https://code.visualstudio.com/
https://www.python.org/downloads/
https://www.trychroma.com/
https://python.langchain.com/docs/get_started/introduction
