import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from helper.vdb_manager import VectorDBManager
from dotenv import load_dotenv

load_dotenv()

def main():
    """
    Load and test Australian legal documents from JSONL file
    """
    
    print("🇦🇺 Australian Legal Documents - High Court of Australia Cases")
    print("=" * 70)
    
    # Check for OpenAI API key
    use_openai = True
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Error: OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        print("This script requires OpenAI embeddings to run.")
        return
    
    # Initialize the vector database manager for Australian documents
    vdb_manager = VectorDBManager(
        client_name="australia",  # This will create vdb/australia folder
        allowed_collections=["external-collection"],
        use_openai=use_openai
    )
    
    print("🤖 Using OpenAI embeddings (text-embedding-3-small)")
    
    # Configuration
    australia_jsonl_path = "db/External/australia.jsonl"
    chunk_size = 1000          # Size of each text chunk
    chunk_overlap = 200        # 200 characters overlap between chunks
    batch_size = 25 if use_openai else 50   # Smaller batches for OpenAI to avoid rate limits
    rate_limit_delay = 0.2 if use_openai else 0.1  # Longer delay for OpenAI
    
    print(f"📁 File: {australia_jsonl_path}")
    print(f"📦 Chunk size: {chunk_size} characters")
    print(f"🔄 Chunk overlap: {chunk_overlap} characters ({chunk_overlap/chunk_size*100:.1f}%)")
    print(f"📊 Batch size: {batch_size}")
    print(f"⏱️  Rate limit delay: {rate_limit_delay}s")
    print(f"🤖 Using OpenAI embeddings: {use_openai}")
    print(f"🔍 Filtering for: High Court of Australia cases only")
    print("")
    
    # Check if file exists
    if not os.path.exists(australia_jsonl_path):
        print(f"❌ Error: {australia_jsonl_path} not found!")
        print("Please ensure the australia.jsonl file is in the correct location.")
        return
    
    try:
        # Load and process the Australian legal documents
        print("🔄 Starting document processing...")
        
        vdb_manager.load_legal_cases_from_json(
            json_file_path=australia_jsonl_path,
            collection_name='external-collection',
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
        )
        
        print("✅ Processing completed successfully!")
        
        # Debug collection contents
        print("\n🔍 Debugging collection contents...")
        stats = vdb_manager.get_collection_stats('external-collection')
        print(stats)
        
        # Test queries with debugging
        print("\n🔍 Testing query functionality...")
        
        # Test with Australian legal terms
        test_queries = [
            "constitutional law",
            "high court decision",
            "commonwealth powers",
            "judicial review",
            "statutory interpretation"
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Testing query: '{query}'")
            
            # Test regular search
            results = vdb_manager.query_collection(
                query_text=query,
                collection_name='external-collection',
                n_results=3,
                similarity_threshold=0.6
            )
            
            if results and results['documents']:
                print(f"Found {len(results['documents'])} results:")
                for i, (doc, meta, dist) in enumerate(zip(results['documents'], results['metadatas'], results['distances'])):
                    print(f"  - Doc {i+1} (Distance: {dist:.4f}): {doc[:100]}...")
                    print(f"    Metadata: {meta}")
            else:
                print("  No results found.")

        # Test with filters
        print("\n" + "="*50)
        print("Testing with filters...")
        
        type_results = vdb_manager.query_collection(
            query_text="legislation",
            collection_name='external-collection',
            n_results=3,
            similarity_threshold=0.5,
            where_clause={"type": "primary_legislation"}
        )
        
        if type_results and type_results['documents']:
            print("\nFound primary legislation documents:")
            for doc, meta in zip(type_results['documents'], type_results['metadatas']):
                print(f"  - {meta.get('citation', 'N/A')}: {doc[:100]}...")
        else:
            print("\nNo primary legislation documents found.")
            
        # Get stats
        print("\n" + "="*50)
        print("Getting document stats...")
        stats = vdb_manager.get_collection_stats('external-collection')
        print(stats)

        # Get a specific document by ID if possible
        if stats.get('sample') and stats['sample']['ids']:
            sample_id = stats['sample']['ids'][0]
            print(f"\nGetting chunks for document: {sample_id}")
            # This part is tricky as we don't have a direct get_by_id in the new manager
            # We can simulate it with a query
            doc_chunks = vdb_manager.query_collection(
                collection_name='external-collection',
                query_text="", # No text, just filter
                n_results=100, # Get many chunks
                where_clause={"chunk_id": {"$like": f"{sample_id.split('_chunk_')[0]}%"}}
            )
            
            if doc_chunks and doc_chunks['documents']:
                print(f"Found {len(doc_chunks['documents'])} chunks for the document.")
            else:
                print("Could not retrieve chunks for the sample document.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
