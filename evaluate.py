"""
evaluate.py - runs a test set of questions against the chatbot and scores
the answers, logging results to LangSmith as an experiment.

Run with: python3 evaluate.py
"""
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate

from rag import load_vectorstore
from tools import get_tools
from agent import build_agent, run_turn

vectorstore = load_vectorstore()
tools = get_tools(vectorstore)
agent_app = build_agent(tools)


def target(inputs: dict) -> dict:
    reply, _ = run_turn(agent_app, inputs["question"], [])
    return {"answer": reply}


def keyword_match(outputs: dict, reference_outputs: dict) -> bool:
    """Simple pass/fail: does the answer contain all expected keywords?"""
    expected = reference_outputs["expected_keywords"]
    answer = outputs["answer"].lower()
    return all(kw.lower() in answer for kw in expected)


examples = [
    # Calculator tool
    {"inputs": {"question": "What's 12 + 8?"},
     "outputs": {"expected_keywords": ["20"]}},
    {"inputs": {"question": "What's 234 * 17?"},
     "outputs": {"expected_keywords": ["3978"]}},
    {"inputs": {"question": "What's 12% of 500?"},
     "outputs": {"expected_keywords": ["60"]}},

    # Date tool (deterministic - should always pass)
    {"inputs": {"question": "What's today's date?"},
     "outputs": {"expected_keywords": [datetime_year_placeholder := __import__("datetime").datetime.now().strftime("%Y")]}},

    # Document search + reranking
    {"inputs": {"question": "What are the six criteria for techniques mentioned in the document?"},
     "outputs": {"expected_keywords": ["approved", "attack area", "awareness"]}},
    {"inputs": {"question": "Does turning away after a technique count as maintaining awareness?"},
     "outputs": {"expected_keywords": ["not maintaining awareness"]}},

    # Web search
    {"inputs": {"question": "What's the capital of France?"},
     "outputs": {"expected_keywords": ["paris"]}},
]

client = Client()
dataset_name = "deepseek-chatbot-eval-v2"

if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=dataset.id,
    )
    print(f"Created dataset '{dataset_name}' with {len(examples)} examples.")
else:
    print(f"Using existing dataset '{dataset_name}'.")

results = evaluate(
    target,
    data=dataset_name,
    evaluators=[keyword_match],
    experiment_prefix="deepseek-chatbot",
)

print("\nEvaluation complete. Check the LangSmith dashboard under")
print(f"'{dataset_name}' to see per-question results and pass rate.")
