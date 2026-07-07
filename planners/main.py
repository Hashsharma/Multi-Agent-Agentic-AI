import os
from dotenv import load_dotenv
from graph import app
from rag_stores import initialize_rag
from langsmith import traceable  # <-- ADD THIS IMPORT

load_dotenv()

# Pre-load RAG vector store
print("🧠 Initializing RAG knowledge base...")
initialize_rag()
print("✅ RAG ready.\n")

@traceable(run_type="chain", name="Nutritional Agent")  # <-- ADD THIS DECORATOR
def run_agent(user_query: str):
    """Run the nutritional agent with a default user profile."""
    profile = {
        "name": "Test User",
        "age": 30,
        "weight_kg": 80,
        "height_cm": 175,
        "gender": "male",
        "dietary_restrictions": ["vegetarian"]
    }
    
    initial_state = {
        "user_profile": profile,
        "messages": [("user", user_query)],
        "retrieved_context": []
    }
    
    print(f"🔍 Query: {user_query}\n")
    final_state = app.invoke(initial_state)
    return final_state["messages"][-1].content

if __name__ == "__main__":
    # --- Test Case 1: Complex (BMR + Meal Plan + RAG fact) ---
    query1 = "I'm male, 80kg, 175cm, 30yo. Give me a vegetarian dinner plan around 600 kcal. Also, is tofu actually good for me?"
    print("=" * 60)
    response1 = run_agent(query1)
    print("\n🤖 FINAL ANSWER:\n", response1)
    
    print("\n" + "=" * 60)
    
    # --- Test Case 2: Pure RAG (no tools needed) ---
    query2 = "What are the main health benefits of eating lentils?"
    response2 = run_agent(query2)
    print("\n🤖 FINAL ANSWER:\n", response2)