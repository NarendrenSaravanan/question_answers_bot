
import os
import time
import backoff
from langchain.document_loaders import PyPDFLoader, JSONLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from openai import RateLimitError
from typing import List


class QaGenerator:
    def __init__(self) -> None:
        # self.__supported
        pass

    def __load_docs(self, document_path, document_type):
        if document_type == "pdf":
            loader = PyPDFLoader(document_path)
        elif document_type == "json":
            loader = JSONLoader(file_path=document_path,
                                jq_schema='.', text_content=False)
        documents = loader.load()
        return documents

    def __initialize_db(self, document_path, document_type):
        documents = self.__load_docs(document_path, document_type)
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(disallowed_special=())
        db = Chroma.from_documents(texts, embeddings)
        return db

    def __get_chatgpt_model(self, db):
        retriever = db.as_retriever(
            search_type="mmr",  # You can also experiment with "similarity"
            search_kwargs={"k": 8},
            max_tokens_limit=10000,
            top_n=5
        )
        llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
        # First we need a prompt that we can pass into an LLM to generate this search query
        prompt = ChatPromptTemplate.from_messages(
            [
                ("placeholder", "{chat_history}"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
                ),
            ]
        )
        retriever_chain = create_history_aware_retriever(
            llm, retriever, prompt)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's questions based on the below context:\n\n{context}",
                ),
                ("placeholder", "{chat_history}"),
                ("user", "{input}"),
            ]
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        model = create_retrieval_chain(retriever_chain, document_chain)
        return model

    @backoff.on_exception(backoff.expo, RateLimitError)
    def __invoke_model(self, model, question):
        return model.invoke({"input": question})

    def __generate_answers(self, db, questions):
        qa_model = self.__get_chatgpt_model(db)
        resp = {}
        for question in questions:
            result = self.__invoke_model(qa_model, question)
            print(f"Question: {question} \n")
            print(f"Answer: {result['answer']} \n")
            resp[question] = result['answer']
        return resp

    def execute(self, document_path: str, document_type: str, questions: List[str]):
        if "OPENAI_API_KEY" not in os.environ:
            raise Exception("OPENAI_API_KEY is required to run this module")
        db = self.__initialize_db(document_path, document_type)
        return {"result": self.__generate_answers(db, questions)}
