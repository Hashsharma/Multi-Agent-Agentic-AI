from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # User context (age, weight, restrictions)
    user_profile: dict
    # LangGraph automatically appends new messages using the reducer
    messages: Annotated[List[BaseMessage], operator.add]
    # Holds RAG context for the final responder
    retrieved_context: List[str]