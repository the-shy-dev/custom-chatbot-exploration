import os
from dotenv import load_dotenv
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

class Chatbot:
    def __init__(self, db_folder: str = "./vectorstore"):
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")

        if not os.path.exists(db_folder) or len(os.listdir(db_folder)) == 0:
            raise ValueError("No precomputed ChromaDB found! Run `python prepare_vectordb.py` first.")

        print("Loading precomputed VectorDB...")
        self.vectorstore = Chroma(persist_directory=db_folder, embedding_function=self.embedding)

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question")

        self.llm = OpenAI(temperature=0.5, streaming=True)

        self.retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.5})

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory
        )

    def stream_response(self, query):
        response = self.chain.stream({"question": query})
        for chunk in response:
            yield chunk["answer"]

chatbot = Chatbot(db_folder="./vectorstore")
