from langchain_community.llms.fake import FakeListLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import mlflow

MODEL_STORAGE = "./models/"
MODEL_NAME = "simple_chain"
MODEL_VERSION = "latest"


LLM = FakeListLLM(responses=["hi", "hello"])


def simple_chain():
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )

    chain = LLMChain(llm=LLM, prompt=prompt)

    logged_model = mlflow.langchain.log_model(chain, MODEL_NAME)
    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)

    print(loaded_model.predict([{"product": "colorful socks"}]))
