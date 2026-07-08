from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

# Mock Nutritional Knowledge Base
MOCK_DOCS = [
    "Tofu is a complete protein containing all essential amino acids. Rich in iron and calcium, it supports bone health.",
    "Lentils are an excellent source of plant-based protein and dietary fiber. They help stabilize blood sugar and improve digestion.",
    "Broccoli is loaded with vitamin C, vitamin K, and sulforaphane, which has powerful anti-inflammatory and antioxidant effects.",
    "Chicken breast is a lean protein source, low in saturated fat, and packed with B-vitamins, especially niacin and B6.",
    "The Mediterranean diet emphasizes whole grains, healthy fats like olive oil, and fresh produce. It is linked to heart health.",
    "Intermittent fasting cycles between eating and fasting periods. It may aid in weight management and metabolic regulation."
]

# Global retriever instance
_retriever = None

def initialize_rag(force_rebuild: bool = False):
    """
    Initialize the RAG retriever using LangChain's Chroma wrapper.
    
    Args:
        force_rebuild: If True, rebuild the vector store from scratch
    
    Returns:
        A LangChain retriever with .invoke() method
    """
    global _retriever
    
    # Return cached retriever if it exists and we don't want to rebuild
    if _retriever is not None and not force_rebuild:
        return _retriever
    
    # Create documents from mock data
    documents = [
        Document(page_content=text, metadata={"source": "mock_nutrition_guide"})
        for text in MOCK_DOCS
    ]
    
    # Use HuggingFace embeddings (free, runs locally)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create or load the vector store
    if force_rebuild:
        # Delete existing and create new
        import shutil
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
            print(f"🔄 Deleted existing ChromaDB at {CHROMA_DIR}")
    
    # LangChain's Chroma wrapper handles everything!
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    
    # Persist the database to disk
    vectorstore.persist()
    
    # Create retriever - this automatically has .invoke()!
    _retriever = vectorstore.as_retriever(
        search_type="similarity",  # Can also use "mmr" for diversity
        search_kwargs={"k": 3}     # Return top 3 most relevant documents
    )
    
    print(f"✅ RAG initialized with {len(MOCK_DOCS)} documents")
    return _retriever


@tool
def rag_retrieve(query: str) -> str:
    """
    Search the nutritional knowledge base (RAG) for general facts, 
    health benefits, or scientific context about foods or diets.
    
    Args:
        query: The search query (e.g., "health benefits of tofu")
    
    Returns:
        Relevant nutritional information as a formatted string
    """
    try:
        retriever = initialize_rag()
        
        # ✅ .invoke() works perfectly with LangChain's retriever!
        docs = retriever.invoke(query)
        
        if not docs:
            return "No relevant nutritional information found in the knowledge base."
        
        # Format the results
        result = "📚 **Nutritional Knowledge Found:**\n"
        for i, doc in enumerate(docs, 1):
            result += f"{i}. {doc.page_content}\n"
        
        return result
        
    except Exception as e:
        print(f"❌ RAG retrieval error: {e}")
        return f"Error retrieving information: {str(e)}"


# Optional: Add function to add new documents
def add_documents(new_docs: list):
    """
    Add new documents to the existing vector store.
    
    Args:
        new_docs: List of strings containing new nutritional information
    """
    retriever = initialize_rag()
    vectorstore = retriever.vectorstore  # Access the underlying vectorstore
    
    documents = [
        Document(page_content=text, metadata={"source": "user_added"})
        for text in new_docs
    ]
    
    vectorstore.add_documents(documents)
    vectorstore.persist()
    print(f"✅ Added {len(new_docs)} new documents to RAG")

