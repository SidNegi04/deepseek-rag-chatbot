"""
agent.py - hosted-LLM agent (via OpenRouter's free tier) built as an
explicit LangGraph state graph instead of a black-box AgentExecutor.

OpenRouter is used instead of a local Ollama model so the bot no longer
depends on a tunnel (ngrok/Cloudflare) or a machine staying online, while
still using the real DeepSeek-R1 model (OpenRouter hosts a free variant).
Point this at any OpenRouter model via the OPENROUTER_MODEL env var - see
https://openrouter.ai/models?max_price=0 for the current free list.

DeepSeek-R1 does not support native tool-calling on OpenRouter's free
variant, so this graph replicates the ReAct text-parsing loop manually.
A custom, forgiving output parser handles messy output: if the model
gives both an Action and a Final Answer, it trusts the Final Answer; if
it gives neither cleanly (e.g. stops after just a "Thought"), it nudges
the model to retry instead of giving up and returning the incomplete
text as if it were an answer.
"""
import os
import re
from typing import TypedDict, List, Tuple, Union

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools.render import render_text_description
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langgraph.graph import StateGraph, END

# The free DeepSeek-R1 variant on OpenRouter. OpenRouter occasionally
# rotates which models are free - override with OPENROUTER_MODEL if this
# one disappears (check https://openrouter.ai/models?max_price=0).
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528:free")

REACT_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant with access to the user's internal documents
and external tools. Answer the user's question as accurately as possible.

Rules:
- If the question could relate to the user's own documents, check
  search_internal_documents first before using web_search.
- If the user asks anything that might be answered by an uploaded document
  (specific facts, rules, criteria, numbers, names), you MUST use
  search_internal_documents first. Never guess or answer from memory if a
  document search could contain the answer.
- For ANY question about today's date or the current day, you MUST use
  get_current_date. Do NOT use web_search for this, and do NOT answer from
  memory. Use web_search for current weather or current events instead.
- For ANY question about Formula 1 (drivers, constructors, races, results,
  standings, circuits, seasons from 1950-present), you MUST use the F1
  database tools instead of web_search or memory. Call f1_database_schema
  first if you don't already know the exact table/column names, then
  query_f1_database with a single SELECT statement.
- Only use tools when you actually need them; answer directly if you already know.
- Be concise and cite which source (document or web) your answer came from
  when relevant.
- Give EXACTLY ONE of: an Action, OR a Final Answer. Never both in the same response.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation:
{chat_history}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)


class AgentState(TypedDict):
    input: str
    chat_history: str
    intermediate_steps: List[Tuple[AgentAction, str]]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    retries: int


def _strip_think(message):
    text = message.content if hasattr(message, "content") else str(message)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if hasattr(message, "content"):
        message.content = cleaned
        return message
    return cleaned


_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(.*?)\s*\nAction Input:\s*(.*)", re.DOTALL)

MAX_RETRIES = 3


def _robust_parse(text: str, retries: int):
    """
    Forgiving parser with a safety net:
    - Both Final Answer and Action present -> trust the Final Answer.
    - Only Action present -> take the action.
    - Only Final Answer present -> finish.
    - Neither present (e.g. model stopped after just a Thought) -> retry,
      rather than treating the incomplete text as a real answer.
    """
    text = text.strip()

    final_match = _FINAL_RE.search(text)
    if final_match:
        answer = final_match.group(1).strip().split("\n\n")[0].strip()
        return AgentFinish(return_values={"output": answer}, log=text)

    action_match = _ACTION_RE.search(text)
    if action_match:
        tool = action_match.group(1).strip()
        tool_input = action_match.group(2).strip().split("\n")[0].strip().strip('"')
        return AgentAction(tool=tool, tool_input=tool_input, log=text)

    if retries >= MAX_RETRIES:
        # Give up gracefully after too many malformed attempts, rather than
        # looping forever or returning a half-finished internal thought.
        return AgentFinish(
            return_values={"output": "I wasn't able to complete that request. Could you try rephrasing it?"},
            log=text,
        )

    # Incomplete response (e.g. only a "Thought" with no Action/Final Answer).
    # Signal a retry by returning None; the caller loops back with a nudge.
    return None


def build_agent(tools):
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Get a free key at "
            "https://openrouter.ai/keys and set it as an environment variable."
        )
    llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        temperature=0.1,
        max_tokens=1536,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    tool_map = {t.name: t for t in tools}

    prompt = REACT_PROMPT.partial(
        tools=render_text_description(list(tools)),
        tool_names=", ".join(t.name for t in tools),
    )
    llm_with_stop = llm.bind(stop=["\nObservation"])

    def run_agent_node(state: AgentState):
        scratchpad = format_log_to_str(state["intermediate_steps"])
        chain = prompt | llm_with_stop
        message = chain.invoke({
            "input": state["input"],
            "chat_history": state.get("chat_history", ""),
            "agent_scratchpad": scratchpad,
        })
        cleaned = _strip_think(message)
        text = cleaned.content if hasattr(cleaned, "content") else cleaned
        retries = state.get("retries", 0)
        outcome = _robust_parse(text, retries)

        if outcome is None:
            # Malformed/incomplete output: add a nudging "observation" to the
            # scratchpad so the next attempt sees what went wrong, then retry.
            nudge_action = AgentAction(
                tool="_format_reminder_",
                tool_input="",
                log=text,
            )
            nudge_observation = (
                "Your last response was incomplete. You must respond with "
                "EITHER a complete 'Action:' + 'Action Input:' pair, OR a "
                "'Final Answer:' line. Try again now, following the format exactly."
            )
            return {
                "agent_outcome": None,
                "intermediate_steps": state["intermediate_steps"] + [(nudge_action, nudge_observation)],
                "retries": retries + 1,
            }

        return {"agent_outcome": outcome}

    def run_tool_node(state: AgentState):
        action = state["agent_outcome"]
        tool = tool_map.get(action.tool)
        observation = tool.run(action.tool_input) if tool else f"Tool '{action.tool}' not found."
        return {"intermediate_steps": state["intermediate_steps"] + [(action, str(observation))]}

    def should_continue(state: AgentState):
        outcome = state["agent_outcome"]
        if outcome is None:
            return "retry"
        return "end" if isinstance(outcome, AgentFinish) else "continue"

    graph = StateGraph(AgentState)
    graph.add_node("agent", run_agent_node)
    graph.add_node("tools", run_tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {
        "continue": "tools",
        "end": END,
        "retry": "agent",
    })
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_turn(app, user_input: str, history_pairs: list):
    history_text = "\n".join(f"{who}: {text}" for who, text in history_pairs) or "(none yet)"

    result = app.invoke({
        "input": user_input,
        "chat_history": history_text,
        "intermediate_steps": [],
        "agent_outcome": None,
        "retries": 0,
    }, config={"recursion_limit": 25})

    outcome = result["agent_outcome"]
    reply = outcome.return_values.get("output", "Sorry, I couldn't produce an answer.") \
        if isinstance(outcome, AgentFinish) else "Agent stopped without a final answer."

    updated_history = history_pairs + [("Human", user_input), ("AI", reply)]
    return reply, updated_history
