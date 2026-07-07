from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from state import AgentState
from tools import nutritional_tools
from rag_stores import rag_retrieve
import os

import os
from langchain_openai import ChatOpenAI

# Combine all tools (including RAG)
all_tools = nutritional_tools + [rag_retrieve]

planner_llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
)

writer_llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
)

planner_with_tools = planner_llm.bind_tools(all_tools)


# --- System Prompts ---
PLANNER_SYSTEM = """You are a precise nutritional planning agent.
You have access to the user's profile (age, weight, height, etc.) via the state.
**Rules:**
1. If the user asks for a meal plan or BMR, you MUST call the relevant tools.
2. If the user asks for generic facts (e.g., "Is tofu healthy?"), call 'rag_retrieve'.
3. If you already have the answer from the conversation history, do NOT call a tool.
4. Always respond with the appropriate tool calls (if any). 
User Profile: {profile}
"""

WRITER_SYSTEM = """You are a friendly, safe nutritional advisor.
Synthesize the tool outputs and RAG context below into a clear, concise answer for the user.
If the user asked for a plan, format it nicely with bullet points.
Context from RAG/DB:
{context}
"""

def planner_node(state: AgentState):
    """Decides which tools to call (or none)."""
    try:
        profile = state.get("user_profile", {})
        messages = state["messages"]
        sys_msg = ("system", PLANNER_SYSTEM.format(profile=profile))

        # Fix: Check if first message is already a system message
        if not messages or messages[0] != "system":
            messages = [sys_msg] + messages
            
        response = planner_with_tools.invoke(messages)
        # No need to bind tools again here since planner_llm is already configured
        # response = planner_llm.invoke(messages)
        return {"messages": [response]}
    except Exception as ex:
        print(f"❌ Planner error: {ex}")
        # Return a fallback message to prevent graph failure
        fallback_msg = {"role": "assistant", "content": "I encountered an error while planning. Please try again."}
        return {"messages": [fallback_msg]}

def responder_node(state: AgentState):
    """Generates the final natural language response."""
    try:
        context = state.get("retrieved_context", [])
        context_str = "\n".join(context) if context else "No specific context retrieved."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", WRITER_SYSTEM),
            MessagesPlaceholder(variable_name="messages")
        ])
        chain = prompt | writer_llm
        response = chain.invoke({"context": context_str, "messages": state["messages"]})
        return {"messages": [response]}
    except Exception as ex:
        print(f"❌ Responder error: {ex}")
        fallback_msg = {"role": "assistant", "content": "I encountered an error while generating the response. Please try again."}
        return {"messages": [fallback_msg]}

# --- Build Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("tools", ToolNode(all_tools))
workflow.add_node("responder", responder_node)

workflow.set_entry_point("planner")

# Conditional routing: if the planner called a tool -> go to tools, else directly to responder
def route_after_planner(state: AgentState):
    last_msg = state["messages"][-1]
    # Check if the last message has tool_calls attribute and it's not empty
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "responder"

workflow.add_conditional_edges("planner", route_after_planner)
workflow.add_edge("tools", "responder")
workflow.add_edge("responder", END)

# Compile
app = workflow.compile()