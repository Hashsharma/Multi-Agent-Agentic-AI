[User Query + Profile]
        |
        v
+---------------------------------------+
|          ORCHESTRATOR                  |
|  (Session State, Context Prep)        |
+---------------------------------------+
        |
        v
+---------------------------------------+
|  PLANNER (Simple LLM API Call)        | <--- Retrieves RAG context to ground plan
|  Prompt: Tools Schema + Query         |
|  Output: JSON Tool-Call Plan          |
+---------------------------------------+
        |
        v
+---------------------------------------+
|  EXECUTOR (Tool Router)               |
|  +---------+   +---------+   +------+ |
|  | RAG     |   | BMR     |   | Food | |
|  | Retriever|   | Calc    |   | DB   | |
|  +---------+   +---------+   +------+ |
+---------------------------------------+
        |
        v
+---------------------------------------+
|  SYNTHESIZER (2nd LLM API Call)       |
|  Raw Results -> Natural Language      |
+---------------------------------------+
        |
        v (Parallel Async)
+---------------------------------------+
|  EVALUATOR (LLM-as-Judge)             |
|  Scores: Faithfulness, Relevance      |
+---------------------------------------+
        |
        v
[Final Response + In-Memory Trace Metrics]