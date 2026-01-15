import os
import uuid

import boto3
import streamlit as st

# ---------- AWS + S3 ----------
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# ---------- LangChain imports that match your installed packages ----------
# Bedrock LLM + embeddings live in langchain_community
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock

# PromptTemplate now lives in langchain_core.prompts
from langchain_core.prompts import PromptTemplate

# FAISS vector store
from langchain_community.vectorstores import FAISS

# PDF loader (not actually used on the client but harmless)
from langchain_community.document_loaders import PyPDFLoader

# Text splitter (same)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.chat_models import BedrockChat
from langchain_community.chat_models import BedrockChat


# ---------- Bedrock clients ----------
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
)

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_client,
)

folder_path = "/tmp/"


def get_unique_id():
    return str(uuid.uuid4())


# ---------- Download FAISS index from S3 ----------
def load_index():
    os.makedirs(folder_path, exist_ok=True)

    s3_client.download_file(
        Bucket=BUCKET_NAME,
        Key="my_faiss.faiss",
        Filename=f"{folder_path}my_faiss.faiss",
    )
    s3_client.download_file(
        Bucket=BUCKET_NAME,
        Key="my_faiss.pkl",
        Filename=f"{folder_path}my_faiss.pkl",
    )


# ---------- LLM wrapper ----------
def get_llm():
    llm = BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=bedrock_client,          # your boto3 bedrock-runtime client
        model_kwargs={
            "max_tokens": 512,          # response length
            "temperature": 0,           # deterministic, good for Q&A
        },
    )
    return llm


# ---------- Manual Retrieval + QA (no RetrievalQA chain needed) ----------
def get_response(llm, vectorstore, question: str) -> str:
    # 1. Retrieve top-k relevant chunks from FAISS
    docs = vectorstore.similarity_search(question, k=5)

    context = "\n\n".join(doc.page_content for doc in docs)

    # 2. Build prompt
    prompt_template = """
Human: Please use the given context to provide a concise answer to the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

<context>
{context}
</context>

Question: {question}

Assistant:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    ).format(context=context, question=question)

    # 3. Call the LLM
    # In the new LangChain style, LLMs support .invoke()
    answer = llm.invoke(prompt)

    # Some models return a string directly; others return an object with .content
    if hasattr(answer, "content"):
        return answer.content
    return str(answer)


# ---------- Streamlit app ----------
def main():
    st.header("This is Client Site for Chat with PDF demo using Bedrock + RAG")

    # 1. Download FAISS index from S3
    load_index()

    dir_list = os.listdir(folder_path)
    st.write(f"Files and directories in {folder_path}:")
    st.write(dir_list)

    # 2. Load FAISS index
    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True,
    )

    st.write("INDEX IS READY ✅")

    # 3. Q&A UI
    question = st.text_input("Please ask your question")
    if st.button("Ask Question") and question.strip():
        with st.spinner("Querying..."):
            llm = get_llm()
            answer = get_response(llm, faiss_index, question)
            st.write(answer)
            st.success("Done")


if __name__ == "__main__":
    main()
