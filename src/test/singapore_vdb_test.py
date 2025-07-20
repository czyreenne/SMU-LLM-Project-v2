import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from helper.vdb_manager import VectorDBManager
from dotenv import load_dotenv

load_dotenv()

def main():
    """
    Simple example of using the parallel processing Singapore JSON loader
    with chunk overlap functionality
    """
    
    print("🚀 Singapore Legal Cases - Parallel Processing with Overlap")
    print("=" * 60)
    
    # Check for OpenAI API key
    use_openai = True
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️  OpenAI API key not found. Falling back to local embeddings.")
        use_openai = False
    
    # Initialize the vector database manager
    vdb_manager = VectorDBManager(
        client_name="singapore",
        allowed_collections=["internal-collection", "external-collection"],
        embedding_model_name="all-MiniLM-L6-v2",  # Smaller, faster local model
        use_openai=use_openai
    )
    
    # Configuration
    singapore_json_path = "db/Internal/singapore.json"
    chunk_size = 1000          # Size of each text chunk
    chunk_overlap = 200        # 200 characters overlap between chunks
    batch_size = 100          # Process 100 chunks at a time
    max_workers = 4           # Use 4 parallel workers
    
    print(f"📁 File: {singapore_json_path}")
    print(f"📦 Chunk size: {chunk_size} characters")
    print(f"🔄 Chunk overlap: {chunk_overlap} characters ({chunk_overlap/chunk_size*100:.1f}%)")
    print(f"⚡ Workers: {max_workers}")
    print(f"🤖 Using OpenAI embeddings: {use_openai}")
    print("")
    
    # Check if file exists
    if not os.path.exists(singapore_json_path):
        print(f"❌ Error: {singapore_json_path} not found!")
        print("Please ensure the singapore.json file is in the correct location.")
        return
    
    try:
        # Load and process the Singapore legal cases with parallel processing
        print("🔄 Starting parallel processing...")
        
        vdb_manager.load_legal_cases_from_json(
            json_file_path=singapore_json_path,
            collection_name='internal-collection',
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            max_workers=max_workers,
            mode='parallel'
        )
        
        print("✅ Processing completed successfully!")
        
        # Debug collection contents first
        print("\n🔍 Debugging collection contents...")
        stats = vdb_manager.get_collection_stats('internal-collection')
        print(stats)
        
        # Test a simple query with debugging
        print("\n🔍 Testing query functionality...")
        results = vdb_manager.query_collection(
            collection_name='internal-collection',
            query_text="contract dispute resolution",
            n_results=3
        )
        
        if not results or not results['documents']:
            # Try alternative queries if the first one fails
            print("\n🔄 Trying alternative queries...")
            alternative_queries = [
                "legal case",
                "court decision", 
                "judgment",
                "singapore",
                ""  # Empty query to get any documents
            ]
            
            for query in alternative_queries:
                print(f"\nTrying query: '{query}'")
                results = vdb_manager.query_collection(
                    collection_name='internal-collection',
                    query_text=query,
                    n_results=3
                )
                if results and results['documents']:
                    print("Found results with alternative query.")
                    break
        
        if results and results['documents']:
            print("\nQuery results:")
            for doc, meta in zip(results['documents'], results['metadatas']):
                print(f"  - {meta.get('case_title', 'N/A')}: {doc[:100]}...")
        else:
            print("No results found, even with alternative queries.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
