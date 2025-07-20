import os
import sys
import unittest

# Add src directory to Python path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper.vdb_manager import VectorDBManager
from helper.configloader import load_agent_config

class TestRetrieval(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment by initializing the VectorDBManager.
        This setup mimics how agent clients are configured in the main workflow.
        """
        # Load agent configurations to get collection details
        agent_configs = load_agent_config()
        external_agent_config = agent_configs.get("external", {})
        
        self.client_name = "australia" # Use the correct vdb client name
        self.allowed_collections = external_agent_config.get("allowed_collections", ["australia"])
        
        # Initialize the manager, assuming local embeddings for testing to avoid API calls
        self.vdb_manager = VectorDBManager(
            client_name=self.client_name,
            allowed_collections=self.allowed_collections,
            use_openai=True 
        )

        # Load data into the collection for testing purposes

        # Only load if the collection is empty to avoid reloading on every test
        if self.vdb_manager.get_collection_stats("australia")['count'] == 0:
            print(f"Populating '{self.client_name}' collection for testing...")
            json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'australia', 'australia.jsonl'))
            self.assertTrue(os.path.exists(json_path), f"Test data file not found at {json_path}")
            
            self.vdb_manager.load_legal_cases_from_jsonl(
                jsonl_file_path=json_path,
                collection_name="australia",
                batch_size=20, # Use a smaller batch size to avoid token limits
                mode='batch', # Use batch mode for efficiency
                source_filter="high_court_of_australia"
            )
        else:
            print(f"Collection '{self.client_name}' is already populated.")

    def test_australia_case_retrieval(self):
        """
        Test the retrieval of australiaan legal cases based on a sample query.
        This test validates that the querying process works as expected and
        that the metadata (like the 'source') is correctly assigned.
        """
        legal_question = "What constitutes negligence in australiaan law?"
        # The collection name should align with the new, non-prefixed logic
        collection_name = "australia"
        
        # Check if the collection exists before querying
        self.assertIn(collection_name, self.vdb_manager.collections, 
                      f"Collection '{collection_name}' not found in VDB manager.")

        # Perform the query, similar to how agent_clients.py does it
        results = self.vdb_manager.query_collection(
            collection_name=collection_name,
            query_text=legal_question,
            n_results=5,
            distance_threshold=1.0 # Use a distance threshold; lower is more similar
        )

        print(f"\nQuery: '{legal_question}'")
        print(f"Found {len(results.get('documents', []))} results in '{collection_name}'.")

        # Basic assertions to ensure the results structure is correct
        self.assertIsNotNone(results)
        self.assertIn("documents", results)
        self.assertIn("metadatas", results)
        
        # Assuming the database is populated, we expect to get some results
        self.assertGreater(len(results["documents"]), 0, 
                           "Expected to retrieve at least one document from the populated database.")

        # Verify the metadata of each retrieved document
        for metadata in results["metadatas"]:
            self.assertIn("case_id", metadata)
            # This confirms the fix from the previous step is working correctly
            self.assertEqual(metadata.get("source"), "high_court_of_australia", 
                             f"Document source should be '{collection_name}'.")
        
        print("Successfully verified retrieved document metadata.")

if __name__ == '__main__':
    # This allows the test to be run directly from the command line
    unittest.main()
