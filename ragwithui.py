import streamlit as st
import bs4
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Web Page Info
st.set_page_config(page_title="RAG Assistant", page_icon="🤖")
st.title("🌐 Article Assistant")

load_dotenv()
os.environ["USER_AGENT"] = "Mozilla/5.0"

# Sidebar: It opens a control panel on the left side of the page.
with st.sidebar:
    st.header("Options")
    url = st.text_input("Article Web Page :",
                        value="https://lilianweng.github.io/posts/2023-06-23-agent/")
    process_btn = st.button("Process Data")


# RAG function we use cache to store things in memory
@st.cache_resource
def setup_rag(url):
    loader = WebBaseLoader(web_paths=(url,))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectordb = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectordb.as_retriever()


# Main Process
# spinner : It displays a spinning loading icon while the process is in progress.
if process_btn:
    with st.spinner("I reading the article and creating database..."):
        st.session_state.retriever = setup_rag(url)
        st.success("Ready for Quensions !")

# Questions-Answers Area
question = st.text_input("What would you like to know about the article ?")

if question:
    if "retriever" in st.session_state:
        # Prompt ve Model Prepare
        prompt = ChatPromptTemplate.from_template(
            "You are an assistant for article\nQuestion: {question} \nContext: {context} \nAnswer:"
        )
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

        # Chain
        context_docs = st.session_state.retriever.invoke(question)
        context_text = "\n\n".join(doc.page_content for doc in context_docs)

        full_prompt = prompt.format(context=context_text, question=question)
        response = llm.invoke(full_prompt)

        st.markdown("### Answer:")
        st.write(response.content)
    else:
        st.warning("Please first proccess data")