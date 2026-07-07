Here:

agentState = the application's state model
Redis = the persistence layer

Redis becomes the database for the state.

Why not use Redis directly?

You can, but then you have to implement all the state-management logic yourself.

Instead of:

state.messages.append(msg)
return state

The answer depends on which framework you're using.

LangGraph provides the concept of agent state as a core part of the framework.
LangChain does not have a built-in AgentState abstraction in the same way. It focuses on models, prompts, tools, chains, and agents, while state management is generally left to memory components or external storage.

LangGraph is built on top of the LangChain ecosystem, so you'll often see them used together:

### LangChain provides:
LLM wrappers
Prompt templates
Tools
Model integrations
Output parsers

### LangGraph provides:
Stateful execution
Graph-based workflows
StateGraph
AgentState
Checkpointing
Human-in-the-loop support

## A common architecture looks like this:

LangGraph
    │
    ├── AgentState
    ├── Workflow
    ├── Checkpointer
    │
    └── Uses LangChain components
          ├── ChatOpenAI
          ├── Tools
          ├── Prompts
          └── Output parsers


`AgentState` and `StateGraph` are closely related, but they serve different purposes.

Think of it like this:

* **`AgentState` = the data**
* **`StateGraph` = the workflow that moves and updates that data**

### `AgentState`

`AgentState` defines **what information your agent carries around** while it runs.

For example:

```python
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    user_query: str
    search_results: str
    final_answer: str
```

This is just a schema. It doesn't execute anything.

Example state during execution:

```python
{
    "messages": [...],
    "user_query": "Best laptop under $1000",
    "search_results": "...",
    "final_answer": ""
}
```

---

### `StateGraph`

`StateGraph` defines **how the state flows through different nodes**.

For example:

```python
builder = StateGraph(AgentState)

builder.add_node("search", search_node)
builder.add_node("answer", answer_node)

builder.add_edge("search", "answer")
```

This tells LangGraph:

```
START
  │
  ▼
Search Node
  │
  ▼
Answer Node
  │
  ▼
 END
```

The graph passes the same state object from node to node.

---

### How they work together

Suppose the initial state is:

```python
{
    "user_query": "Weather in Bangalore",
    "search_results": "",
    "final_answer": ""
}
```

#### Search node

```python
def search_node(state):
    return {
        "search_results": "28°C and sunny"
    }
```

State becomes:

```python
{
    "user_query": "Weather in Bangalore",
    "search_results": "28°C and sunny",
    "final_answer": ""
}
```

#### Answer node

```python
def answer_node(state):
    return {
        "final_answer": f"Result: {state['search_results']}"
    }
```

Final state:

```python
{
    "user_query": "Weather in Bangalore",
    "search_results": "28°C and sunny",
    "final_answer": "Result: 28°C and sunny"
}
```

The graph orchestrates these steps; the state carries the data between them.

---

### An analogy

Imagine a package delivery system:

* **`AgentState`** is the **package** and everything inside it (address, contents, tracking number).
* **`StateGraph`** is the **delivery route** (warehouse → sorting center → delivery truck → customer).

The package doesn't decide where to go. The route doesn't hold the contents. They work together.

---

### Summary

| `AgentState`                                  | `StateGraph`                           |
| --------------------------------------------- | -------------------------------------- |
| Defines the data the agent carries            | Defines the workflow the agent follows |
| Usually a `TypedDict` or similar schema       | A graph of nodes and edges             |
| Holds messages, variables, tool outputs, etc. | Controls which node runs next          |
| Gets updated by nodes                         | Passes the state between nodes         |
| Doesn't execute logic                         | Executes and orchestrates the workflow |

In short, **`AgentState` is the "what" (the data), while `StateGraph` is the "how" (the execution flow).**


A **reducer** in LangGraph is a function that tells LangGraph **how to combine the old value of a state field with a new value returned by a node**.

Without a reducer, if multiple nodes update the same field, LangGraph wouldn't know whether to replace, append, merge, or otherwise combine the values.

## Example without a reducer

Suppose your state is:

```python
class AgentState(TypedDict):
    messages: list
```

Current state:

```python
{
    "messages": ["Hi"]
}
```

A node returns:

```python
{
    "messages": ["How are you?"]
}
```

Without a reducer, LangGraph would typically **replace** the old value:

```python
{
    "messages": ["How are you?"]
}
```

The original `"Hi"` is lost.

---

## Example with a reducer

For chat messages, you usually want to **append** instead of replace.

LangGraph provides a reducer like `add_messages`:

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

Current state:

```python
{
    "messages": ["Hi"]
}
```

Node returns:

```python
{
    "messages": ["How are you?"]
}
```

The reducer combines them:

```python
{
    "messages": [
        "Hi",
        "How are you?"
    ]
}
```

The node didn't append the message itself—the reducer did.

---

## Think of it like this

When a node returns:

```python
return {
    "messages": [ai_message]
}
```

LangGraph internally does something like:

```python
new_messages = reducer(
    old_messages,
    returned_messages
)
```

For `add_messages`, that's conceptually:

```python
new_messages = old_messages + returned_messages
```

---

## Why are reducers useful?

Imagine two parallel nodes both update the same field:

```
          START
         /     \
        A       B
         \     /
          Merge
```

Node A returns:

```python
{
    "messages": ["Search completed"]
}
```

Node B returns:

```python
{
    "messages": ["Database checked"]
}
```

The reducer determines how these updates are combined. An append-style reducer can produce:

```python
{
    "messages": [
        "Search completed",
        "Database checked"
    ]
}
```

instead of one update overwriting the other.

---

### In simple terms

A reducer answers the question:

> **"When a node returns a new value for this field, how should it be combined with the existing value?"**

Examples:

* **Replace** → `new_value`
* **Append** → `old_list + new_list`
* **Merge dictionaries** → `{**old, **new}`
* **Sum numbers** → `old + new`

The built-in `add_messages` reducer is specifically designed for conversation history, so new messages are appended (and intelligently merged by message ID when appropriate) instead of replacing the existing conversation.


In this case, the important part is actually **`List[BaseMessage]`**, not `operator.add`.

```python
messages: Annotated[List[BaseMessage], operator.add]
```

### Without `BaseMessage`

```python
messages: list[str]
```

Example:

```python
messages = [
    "Hi",
    "Hello!",
    "Search for Python"
]

print(messages)
```

**Output**

```text
['Hi', 'Hello!', 'Search for Python']
```

You only have strings. There's no information about who sent each message.

---

### With `BaseMessage`

```python
from langchain_core.messages import HumanMessage, AIMessage

messages: list[BaseMessage] = [
    HumanMessage(content="Hi"),
    AIMessage(content="Hello!")
]

print(messages)
```

**Output**

```text
[
    HumanMessage(content='Hi'),
    AIMessage(content='Hello!')
]
```

Now every element knows its type.

You can even inspect it:

```python
for msg in messages:
    print(type(msg).__name__, ":", msg.content)
```

**Output**

```text
HumanMessage : Hi
AIMessage : Hello!
```

### Why not `List[str]`?

Because later you can do things like:

```python
for msg in messages:
    if isinstance(msg, HumanMessage):
        print("User:", msg.content)
    elif isinstance(msg, AIMessage):
        print("AI:", msg.content)
```

With `List[str]`, every element is just:

```text
str
```

There's no way to know whether `"Hello!"` came from the user, the AI, or a system instruction unless you invent your own format.

So in:

```python
messages: Annotated[List[BaseMessage], operator.add]
```

* `List[BaseMessage]` means: "This state field holds a list of message objects (HumanMessage, AIMessage, SystemMessage, ToolMessage, etc.)."
* `operator.add` is the reducer that says: "When a node returns new messages, append them to the existing list."


## LangGraph mainly provides:

state management
node execution
conditional routing
looping
checkpointing
persistence
human-in-the-loop support
These features can be implemented manually, though it takes more effort.


Why do people use LangChain then?

## It saves development time by providing:

@tool decorator
Agent executors
Prompt templates
Message abstractions
Memory components
Output parsers
Integrations with many LLM providers
Vector store integrations
Retrieval pipelines

Instead of writing everything from scratch.

## Why do people use LangGraph?

Because agentic systems often need:

Multi-step reasoning
Loops
Conditional branching
Multiple agents
Persistent state
Human approval steps
Error recovery

Instead of manually writing complex control flow like: