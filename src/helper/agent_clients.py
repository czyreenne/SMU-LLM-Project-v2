from helper.vdb_manager import VectorDBManager
from helper.base_agent import BaseAgent

import uuid

class AgentClient:
    def __init__(self, name, config, agent_type, model_str="gpt-4o-mini", api_keys=None, allowed_collections=None):
        """
        initializes an agentclient with access to specific collections in the chromadb database
        """
        if allowed_collections is None:
            allowed_collections = []
        
        # Determine the client_name based on allowed_collections
        if "singapore" in allowed_collections:
            client_name = "singapore"
        elif "australia" in allowed_collections:
            client_name = "australia"
        else:
            client_name = name # Fallback to the agent's name

        self.name = name
        self.agent = BaseAgent(
            agent_type=agent_type,
            input_model=model_str,
            config=config,
            api_keys=api_keys
        )
        self.vdb_manager = VectorDBManager(
            client_name=client_name, 
            allowed_collections=allowed_collections,
            use_openai=True
        )
        self.phases = self.agent.phases

    def add_document(self, collection_name, document, metadata=None, id=None):
        """
        adds a document to a specific collection in the chromadb database
        """
        self.vdb_manager.add_to_collection(
            collection_name=collection_name,
            ids=[id] if id else [str(uuid.uuid4())],
            documents=[document],
            metadatas=[metadata] if metadata else [{}],
        )

    def query_vdb(self, collection_name, query_text, tags=None, distance_threshold=1.0):
        """
        queries a specific collection in the chromadb database    
        """
        try:
            stats = self.vdb_manager.get_collection_stats(collection_name)
            print("\n--- Collection Stats ---")
            print(f"Collection Name: {collection_name}")
            print(f"Document Count:   {stats.get('document_count', 'N/A')}")
            print(f"Total Chunks:    {stats.get('chunk_count', 'N/A')}")
            print("------------------------\n")

        except Exception as e:
            print(f"Failed to get stats: {e}")
            
        where_clause = {"tags": {"$in": tags}} if tags else None
        filtered_results = self.vdb_manager.query_collection(
            collection_name=collection_name,
            query_text=query_text,
            where_clause=where_clause,
            distance_threshold=distance_threshold
        )
        if not filtered_results:
            return ""

        context_pieces = []
        for doc, meta, dist in filtered_results:
            if collection_name == "australia":
                # Format for Australia: citation followed by document text
                citation = meta.get('case_title', 'No Citation Available')
                context_pieces.append(f"{citation}\n\n{doc}")
            elif collection_name == "singapore":
                # Format for Singapore: case_title followed by document text
                case_title = meta.get('case_title', 'No Title Available')
                context_pieces.append(f"{case_title}\n\n{doc}")
            else:
                # Default format for any other collection
                title = meta.get('case_title', 'No Title Available')
                context_pieces.append(f"{title}\n\n{doc}")

        return "\n\n---\n\n".join(context_pieces)

    def perform_phase_analysis(self, question: str, phase: str, step: int = 1, feedback: str = "", temp: float = None):
        """
        performs analysis for a given structured phase defined in legalagents
        """
        if phase not in self.phases:
            raise ValueError(f"Invalid phase '{phase}'. Valid phases are: {self.phases}")
        return self.agent.inference(
            question=question,
            phase=phase,
            step=step,
            feedback=feedback,
            temp=temp
        )

    def perform_full_structured_analysis(self, question: str, distance_threshold=1.0):
        """
        Performs all structured phases sequentially and returns aggregated results.
        Enhanced with relevant legal documents from vector database.
        """
        # Retrieve relevant legal documents from available collections
        collections = list(self.vdb_manager.collections.keys())
        relevant_contexts = []
        
        print("\nAttempting to retrieve relevant documents from vector database...")
        if not collections:
            print("No collections found in the vector database.")

        for collection in collections:
            try:
                # The collection name is now the definitive name, no prefixing.
                print(f"Querying collection: '{collection}'...")
                context_text = self.query_vdb(
                    collection_name=collection,
                    query_text=question,
                    distance_threshold=distance_threshold
                )

            except Exception as e:
                print(f"Error querying collection {collection}: {str(e)}")
        
        # Create enhanced question with retrieved context
        enhanced_question = question
        if context_text:
            print("\nDocument retrieval successful. Enhancing question with retrieved context.")


            enhanced_question = (
                f"Original Question: {question}\n\n"
                f"Relevant Cases and Statutes:\n{context_text}\n\n"
                f"Based on the above context and your legal knowledge, please analyze the original question."
            )
        else:
            print("\nNo documents retrieved. Proceeding with original question.")
        
        # Perform analysis through all phases
        results = {}
        for idx, phase in enumerate(self.phases, start=1):
            print(f"\nPerforming '{phase}' analysis (Step {idx}/{len(self.phases)})...")
            response = self.perform_phase_analysis(
                question=enhanced_question,
                phase=phase,
                step=idx
            )
            results[phase] = response
        
        return results

    def refine_analysis_with_feedback(self, initial_results: dict, feedback: str):
        """
        refines analysis results based on feedback using iterative methods
        """
        refined_results = {}
        for phase in initial_results.keys():
            print(f"\nRefining '{phase}' analysis based on feedback...")
            refined_results[phase] = self.perform_phase_analysis(
                question=initial_results[phase],
                phase=phase,
                feedback=feedback
            )
        return refined_results