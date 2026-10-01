"""
Test script for the Advanced RAG System
Shows document ingestion and querying with self-correction
"""

import requests
import json
import time
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8000"

# Sample documents
DOCUMENTS = [
    {
        "title": "Machine Learning Basics",
        "content": """
Machine Learning is a subset of Artificial Intelligence that enables systems to learn and improve 
from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, 
and make decisions with minimal human intervention.

Types of Machine Learning:
1. Supervised Learning: Models learn from labeled data (inputs paired with outputs)
2. Unsupervised Learning: Models find patterns in unlabeled data
3. Reinforcement Learning: Agents learn by interacting with an environment and receiving rewards

Common Applications:
- Image Recognition
- Natural Language Processing
- Recommendation Systems
- Autonomous Vehicles
- Healthcare Diagnostics
"""
    },
    {
        "title": "Deep Learning Introduction",
        "content": """
Deep Learning is a branch of Machine Learning based on artificial neural networks with multiple layers.
These neural networks are inspired by biological neurons and can learn hierarchical representations of data.

Key Components:
- Neurons: Basic computational units that process inputs and produce outputs
- Layers: Organized groups of neurons (Input, Hidden, Output)
- Activation Functions: Non-linear functions that enable neural networks to learn complex patterns
- Backpropagation: Algorithm to update weights based on error gradients

Popular Deep Learning Frameworks:
- TensorFlow (developed by Google)
- PyTorch (developed by Meta)
- Keras (high-level API, often used with TensorFlow)

Applications:
- Computer Vision (CNN)
- Natural Language Processing (Transformers)
- Speech Recognition
- Game Playing (AlphaGo)
"""
    },
    {
        "title": "Large Language Models",
        "content": """
Large Language Models (LLMs) are deep learning models trained on vast amounts of text data.
They can generate human-like text, answer questions, and perform various language understanding tasks.

Architecture:
- Transformer Architecture: The foundation of modern LLMs
- Attention Mechanism: Allows the model to focus on relevant parts of input
- Token Embeddings: Converting text into numerical representations

Notable LLMs:
- GPT series (OpenAI): GPT-3, GPT-4
- BERT (Google): Bidirectional Encoder Representations
- LLaMA (Meta): Open-source large language model
- T5 (Google): Text-to-Text Transfer Transformer

Training Process:
1. Pre-training on large corpora
2. Fine-tuning on specific tasks
3. Instruction tuning for better alignment

Limitations:
- Hallucination: Generating false or nonsensical information
- Knowledge cutoff: Information only up to training date
- Computational requirements: Expensive to train and run
"""
    }
]

def test_health():
    """Test if server is running"""
    print("🔍 Testing server health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Server is running: {response.json()}\n")
        return True
    except Exception as e:
        print(f"❌ Server not running: {e}")
        return False

def add_documents():
    """Add sample documents to the knowledge base"""
    print("📚 Adding documents to knowledge base...")
    for i, doc in enumerate(DOCUMENTS, 1):
        try:
            response = requests.post(
                f"{BASE_URL}/add-documents",
                json={
                    "content": doc["content"],
                    "metadata": {"source": doc["title"], "doc_id": i}
                }
            )
            result = response.json()
            print(f"  ✅ {doc['title']}: {result['chunks_added']} chunks added")
        except Exception as e:
            print(f"  ❌ Error adding {doc['title']}: {e}")
    print()

def query_system(questions):
    """Query the RAG system with multiple questions"""
    print("🤖 Running queries with self-correction mechanism...\n")
    
    for question in questions:
        print(f"❓ Question: {question}")
        print("-" * 60)
        
        try:
            response = requests.post(
                f"{BASE_URL}/query",
                json={"question": question}
            )
            result = response.json()

            if response.status_code != 200:
                print(f"❌ Server returned {response.status_code}: {result.get('detail', result)}\n")
                print("=" * 60 + "\n")
                continue

            # Display answer
            print(f"📝 Answer:\n{result['answer']}\n")
            
            # Display metadata
            print(f"📊 Metadata:")
            print(f"   Iterations: {result['iterations']}")
            print(f"   Relevance Score: {result['relevance_score']*100:.1f}%")
            
            if result['rewrites']:
                print(f"   Query Rewrites: {len(result['rewrites'])}")
                for i, rewrite in enumerate(result['rewrites'], 1):
                    print(f"     {i}. {rewrite}")
            
            # Display sources
            if result['sources']:
                print(f"\n📚 Sources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"   [{i}] {source[:100]}...")
            
        except Exception as e:
            print(f"❌ Error querying: {e}")
        
        print("\n" + "="*60 + "\n")

def main():
    """Main test flow"""
    print("=" * 60)
    print("🧠 Advanced RAG System - Test Script")
    print("=" * 60)
    print()
    
    # Check if server is running
    if not test_health():
        print("⚠️  Make sure to run: python main.py")
        return
    
    # Add documents
    add_documents()
    
    # Give Qdrant time to index
    print("⏳ Waiting for indexing...")
    time.sleep(2)
    
    # Test queries
    test_queries = [
        "What is machine learning and what are its types?",
        "How do neural networks work in deep learning?",
        "What are the limitations of large language models?",
        "Explain the attention mechanism in transformers",  # Will trigger rewrite
    ]
    
    query_system(test_queries)
    
    print("✅ Test completed!")

if __name__ == "__main__":
    main()