import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_API_KEY")

REPO_URL = "https://github.com/GovTechSG/developer.gov.sg"  # Source URL
DOCS_FOLDER = "hci"  # Folder to check out to
REPO_DOCUMENTS_PATH = "painting/"  # Set to "" to index the whole data folder
DOCUMENT_BASE_URL = (
    "https://ko.wikipedia.org/wiki/%EB%AA%A8%EB%82%98%EB%A6%AC%EC%9E%90"  # Actual URL
)
DATA_STORE_DIR = "data_store"  # Folder to save/load the database

import os
import pathlib
import re

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

name_filter = "**/*.md"
separator = "\n### "  # This separator assumes Markdown docs from the repo uses ### as logical main header most of the time
chunk_size_limit = 1000
max_chunk_overlap = 20

repo_path = pathlib.Path(os.path.join(DOCS_FOLDER, REPO_DOCUMENTS_PATH))  # 두개 합쳐주는 역할
document_files = list(repo_path.glob(name_filter))  # 모든 md 파일에 대한 경로를 가져옴


def convert_path_to_doc_url(doc_path):
    # Convert from relative path to actual document url
    return re.sub(
        f"{DOCS_FOLDER}/{REPO_DOCUMENTS_PATH}/(.*)\.[\w\d]+",
        f"{DOCUMENT_BASE_URL}/\\1",
        str(doc_path),
    )


documents = [
    Document(
        page_content=open(file, "r").read(),
        metadata={"source": convert_path_to_doc_url(file)},
    )
    for file in document_files
]


text_splitter = CharacterTextSplitter(
    separator=separator, chunk_size=chunk_size_limit, chunk_overlap=max_chunk_overlap
)
split_docs = text_splitter.split_documents(documents)


# create a GPT-4 encoder instance
# import tiktoken
# enc = tiktoken.encoding_for_model("gpt-4")

# total_word_count = sum(len(doc.page_content.split()) for doc in split_docs)
# total_token_count = sum(len(enc.encode(doc.page_content)) for doc in split_docs)

# print(f"\nTotal word count: {total_word_count}")
# print(f"\nEstimated tokens: {total_token_count}")
# print(f"\nEstimated cost of embedding: ${total_token_count * 0.0004 / 1000}")


# vector store 생성
# Download the files `$DATA_STORE_DIR/index.faiss` and `$DATA_STORE_DIR/index.pkl` to local

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(split_docs, embeddings)

vector_store.save_local(DATA_STORE_DIR)

# Upload the files `$DATA_STORE_DIR/index.faiss` and `$DATA_STORE_DIR/index.pkl` to local
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

if os.path.exists(DATA_STORE_DIR):
    vector_store = FAISS.load_local(DATA_STORE_DIR, OpenAIEmbeddings())
else:
    print(
        f"Missing files. Upload index.faiss and index.pkl files to {DATA_STORE_DIR} directory first"
    )


###
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

system_template = """Use the following pieces of context to answer the users question.
Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)


###
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

chain_type_kwargs = {"prompt": prompt}
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0, max_tokens=256
)  # Modify model_name if you have access to GPT-4
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)


def print_result(result, question):
    output_text = f"""### Question:
  {question}
  ### Answer:
  {result['answer']}
  ### Sources:
  {result['sources']}
  ### All relevant sources:
  {' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))}
  """
    return output_text


# query = "What is Sogang-hci project?"
# query = "what technique was used in the monaliza?"
# result = chain(query)
# print_result(result, query)
