import bs4
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# OpenAI model version
ai_model = ChatOpenAI(model="gpt-3.5-turbo-0125")

# 1. It extracts and load online data(web pages),
# Removes unnecessary parts(bs_kwargs),
# Uses clean content.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# 1.1. Corrects the data that was moved to the next line.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 2. The document to be used is divided into small pieces for each use.
text_splitter_langchain = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter_langchain.split_documents(docs)

# 3. The divided chunks values are saved to Vector Database (Langchain Chroma Database).
vectordb = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# 3.1. To extract data from VectorDB
retriever = vectordb.as_retriever()

# 4. By using ready-made prompts from the Smith-LangChain website(https://eu.smith.langchain.com/hub/moodlehq/wiki-rag),
# Holding the model in a specific role,
# It is made to respond more effectively.
prompt = ChatPromptTemplate.from_template(
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ai_model
    | StrOutputParser()
)

if __name__ == "__main__":
    for chunk in rag_chain.stream("What is Task execution?"):
        print(chunk, end="", flush=True)