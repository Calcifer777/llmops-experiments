import operator
from pathlib import Path
import tempfile
from typing import Annotated, List, Literal, TypedDict

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.language_models import FakeListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph
import mlflow
import mlflow.langchain

MODEL_NAME = "recursive_summary"
MODEL_VERSION = "latest"

LLM = FakeListLLM(responses=["hi", "hello"])

TOKEN_MAX = 1000
REDUCE_TEMPLATE = """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary
of the main themes.
"""

map_prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following:\\n\\n{context}")]
)
map_chain = map_prompt | LLM | StrOutputParser()

reduce_prompt = ChatPromptTemplate([("human", REDUCE_TEMPLATE)])
reduce_chain = reduce_prompt | LLM | StrOutputParser()


def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents."""
    return sum(LLM.get_num_tokens(doc.page_content) for doc in documents)


# This will be the overall state of the main graph.
# It will contain the input document contents, corresponding
# summaries, and a final summary.
class OverallState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str


# This will be the state of the node that we will "map" all
# documents to in order to generate summaries
class SummaryState(TypedDict):
    content: str


# Here we generate a summary, given a document
async def generate_summary(state: SummaryState):
    response = await map_chain.ainvoke(state["content"])
    return {"summaries": [response]}


# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_summaries(state: OverallState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]


def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }


# Add node to collapse summaries
async def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, TOKEN_MAX
    )
    results = []
    for doc_list in doc_lists:
        results.append(await acollapse_docs(doc_list, reduce_chain.ainvoke))

    return {"collapsed_summaries": results}


# This represents a conditional edge in the graph that determines
# if we should collapse the summaries or not
def should_collapse(
    state: OverallState,
) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > TOKEN_MAX:
        return "collapse_summaries"
    else:
        return "generate_final_summary"


# Here we will generate the final summary
async def generate_final_summary(state: OverallState):
    response = await reduce_chain.ainvoke(state["collapsed_summaries"])
    return {"final_summary": response}


def load_graph():
    # Construct the graph
    # Nodes:
    graph_builder = StateGraph(OverallState)
    graph_builder.add_node("generate_summary", generate_summary)  # same as before
    graph_builder.add_node("collect_summaries", collect_summaries)
    graph_builder.add_node("collapse_summaries", collapse_summaries)
    graph_builder.add_node("generate_final_summary", generate_final_summary)

    # Edges:
    graph_builder.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph_builder.add_edge("generate_summary", "collect_summaries")
    graph_builder.add_conditional_edges("collect_summaries", should_collapse)
    graph_builder.add_conditional_edges("collapse_summaries", should_collapse)
    graph_builder.add_edge("generate_final_summary", END)

    graph = graph_builder.compile()
    return graph


def recursive_summary():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "graph.py"
        with tmp_file.open("w") as fp_dst:
            with Path(__file__).open() as fp_src:
                fp_dst.write(fp_src.read())

        mlflow.models.set_model(load_graph())  # type: ignore
        mlflow.langchain.log_model(
            registered_model_name=MODEL_NAME,
            lc_model=tmp_file.as_posix(),
            artifact_path=MODEL_NAME,
            loader_fn=None,
            persist_dir=None,
        )

        model = mlflow.langchain.load_model(
            f"models:/{MODEL_NAME}/{MODEL_VERSION}",
        )
