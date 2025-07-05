from helper.vdb_manager import db
from helper.legalagents import Internal, External, LegalReviewPanel

import os

class AgentClient:
    def __init__(self, name, config, agent_type="internal", model_str="gpt-4o-mini", api_keys=None, allowed_collections=None):
        """
        initializes an agentclient with access to specific collections in the chromadb database
        """
        if allowed_collections is None:
            allowed_collections = []
        agent_class = Internal if agent_type.lower() == 'internal' else External
        self.name = name
        self.agent = agent_class(
            input_model=model_str,
            api_keys=api_keys,
            config=config,
        )
        self.vdb_manager = db(client_name=name, allowed_collections=allowed_collections)
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
            id=id,
            document=document,
            metadata=metadata,
        )

    def query_documents(self, collection_name, query_text, tags=None, similarity_threshold=0.7):
        """
        queries documents from a specific collection in the chromadb database
        """
        return self.vdb_manager.query_collection(
            collection_name=collection_name,
            query_text=query_text,
            tags=tags,
            include=["documents"],
            similarity_threshold=similarity_threshold
        )["documents"]

    def query_metadatas(self, collection_name, query_text, tags=None, similarity_threshold=0.7):
        """
        queries metadata from a specific collection in the chromadb database
        """
        return self.vdb_manager.query_collection(
            collection_name=collection_name,
            query_text=query_text,
            tags=tags,
            include=["metadatas"],
            similarity_threshold=similarity_threshold
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

    def perform_full_structured_analysis(self, question: str, similarity_threshold=0.75):
        """
        Performs all structured phases sequentially and returns aggregated results.
        Enhanced with relevant legal documents from vector database.
        """
        # Retrieve relevant legal documents from available collections
        collections = list(self.vdb_manager.collections.keys())
        relevant_contexts = []
        
        for collection in collections:
            try:
                # Extract collection name without client prefix
                collection_name = collection.replace(f"{self.vdb_manager.client_name}_", "")
                documents = self.query_documents(
                    collection_name=collection_name,
                    query_text=question,
                    similarity_threshold=similarity_threshold
                )
                
                if documents:
                    context = f"Documents from {collection_name}:\n" + "\n\n".join(documents)
                    relevant_contexts.append(context)
            except Exception as e:
                print(f"Error querying collection {collection}: {str(e)}")
        
        # Create enhanced question with retrieved context
        enhanced_question = question
        if relevant_contexts:
            context_text = "\n\n".join(relevant_contexts)
            enhanced_question = (
                f"Original Question: {question}\n\n"
                f"Relevant Legal Context:\n{context_text}\n\n"
                f"Based on the above context and your legal knowledge, please analyze the original question."
            )
        
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