import warnings

from langchain_core._api import LangChainDeprecationWarning
import mlflow
import mlflow.langchain

from llmops import (
    simple,
    simple_summary,
    recursive_summary,
    retrievalqa_v1,
)

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

mlflow.langchain.autolog(log_models=True)


if __name__ == "__main__":
    with mlflow.start_run():
        # simple_chain()
        # retrievalqa_v1.retrieval_qa()
        recursive_summary.recursive_summary()
