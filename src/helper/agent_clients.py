from helper.vdb_manager import VectorDBManager
from helper.legalagents import Internal, External, LegalReviewPanel

import uuid

class AgentClient:
    def __init__(self, name, config, agent_type="internal", model_str="gpt-4o-mini", api_keys=None, allowed_collections=None):
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

        agent_class = Internal if agent_type.lower() == 'internal' else External
        self.name = name
        self.agent = agent_class(
            input_model=model_str,
            api_keys=api_keys,
            config=config,
        )
        self.vdb_manager = VectorDBManager(
            client_name=client_name, 
            allowed_collections=allowed_collections,
            use_openai=True
        )
        self.phases = self.agent.phases  
        # @zhiyi
        # if im not wrong the phases are currently hardcoded within legalagents right if 
        # we want can add it as an additional Aparam to the AgentClient constructor
        # ~ gong
        # Done
    def query(self, collection_name, query_text, **kwargs):
        """
        queries a specific collection in the chromadb database
        """
        return self.vdb_manager.query_collection(collection_name=collection_name, query_text=query_text, **kwargs)

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

    def query_documents(self, collection_name, query_text, tags=None, distance_threshold=1.0):
        """
        queries documents from a specific collection in the chromadb database
        """
        where_clause = {"tags": {"$in": tags}} if tags else None
        return self.vdb_manager.query_collection(
            collection_name=collection_name,
            query_text=query_text,
            where_clause=where_clause,
            distance_threshold=distance_threshold
        )["documents"]

    def query_metadatas(self, collection_name, query_text, tags=None, distance_threshold=1.0):
        """
        queries metadata from a specific collection in the chromadb database
        """
        where_clause = {"tags": {"$in": tags}} if tags else None
        return self.vdb_manager.query_collection(
            collection_name=collection_name,
            query_text=query_text,
            where_clause=where_clause,
            distance_threshold=distance_threshold
        )["metadatas"]

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
                documents = self.query_documents(
                    collection_name=collection,
                    query_text=question,
                    distance_threshold=distance_threshold
                )
                
                if documents:
                    print(f"Successfully retrieved {len(documents)} documents from '{collection}'.")
                    context = f"Documents from {collection}:\n" + "\n\n".join(documents)
                    relevant_contexts.append(context)
                else:
                    print(f"No relevant documents found in '{collection}' for the given question.")
            except Exception as e:
                print(f"Error querying collection {collection}: {str(e)}")
        
        # Create enhanced question with retrieved context
        enhanced_question = question
        if relevant_contexts:
            print("\nDocument retrieval successful. Enhancing question with retrieved context.")
            context_text = "\n\n".join(relevant_contexts)
            enhanced_question = (
                f"Original Question: {question}\n\n"
                f"Relevant Legal Context:\n{context_text}\n\n"
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