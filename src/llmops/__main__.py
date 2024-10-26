import os
import warnings

import faiss
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_community.llms.fake import FakeListLLM
from langchain_core.embeddings import FakeEmbeddings
from langchain_core._api import LangChainDeprecationWarning
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
import mlflow
import mlflow.langchain

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["OPENAI_API_KEY"] = ""

mlflow.langchain.autolog(log_models=True)

MODEL_STORAGE = "./models/"
MODEL_NAME = "sample_retrieval_qa"
MODEL_VERSION = "latest"


LLM = FakeListLLM(responses=["hi", "hello"])

VECTORSTORE = FAISS(
    embedding_function=FakeEmbeddings(size=512),
    index=faiss.IndexFlatL2(512),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

def retrieval_qa():
    retriever = VectorStoreRetriever(vectorstore=VECTORSTORE)

    combine_docs_chain = create_stuff_documents_chain(
        llm=LLM,
        prompt=ChatPromptTemplate.from_messages(
            [("system", "What are everyone's favorite colors:\\n\\n{context}")]
        ),
    )

    qa = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain,
    )

    qa = RetrievalQA.from_llm(llm=LLM, retriever=retriever)

    fn_faiss = "index.faiss"
    VECTORSTORE.save_local(fn_faiss)

    def load_retriever(persist_directory: str):
        vectorstore = FAISS.load_local(
            persist_directory,
            embeddings=FakeEmbeddings(size=512),
            # you may need to add the line below
            # for langchain_community >= 0.0.27
            allow_dangerous_deserialization=True,
        )
        retriever = vectorstore.as_retriever(_type="retrieval_qa")
        return retriever

    mlflow.langchain.log_model(
        registered_model_name=MODEL_NAME,
        lc_model=qa,
        artifact_path=MODEL_NAME,
        loader_fn=load_retriever,
        persist_dir=fn_faiss,
        model_config=dict(chain_type="retrieval_qa"),
    )

    print("loading model...")
    mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}/{MODEL_VERSION}",
    )


def simple_chain():
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )

    chain = LLMChain(llm=LLM, prompt=prompt)

    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(chain, "langchain_model")

    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)

    print(loaded_model.predict([{"product": "colorful socks"}]))


if __name__ == "__main__":
    # simple_chain()
    retrieval_qa()
