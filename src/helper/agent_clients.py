from helper.vdb_manager import VectorDBManager
from helper.legalagents import Internal, External, LegalReviewPanel
from typing import List
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
        Performs structured analysis for each sub-question.
        Each sub-question runs through all phases independently.
        """
        collections = list(self.vdb_manager.collections.keys())
        relevant_contexts = []

        print("\nAttempting to retrieve relevant documents from vector database...")
        for collection in collections:
            try:
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
            except Exception as e:
                print(f"Error querying collection {collection}: {str(e)}")

        # Enhance base question with context
        context_text = "\n\n".join(relevant_contexts)
        base_prompt_prefix = (
            f"Relevant Legal Context:\n{context_text}\n\n" if context_text else ""
        )

        # 🔥 NEW: Split into subquestions
        subquestions = self.split_into_subquestions(question)
        results = {}

        for idx, subq in enumerate(subquestions):
            subq_label = f"subquestion_{idx + 1}"
            print(f"\n➡️ Analyzing {subq_label}: {subq[:100]}...")

            enhanced_question = (
                f"{base_prompt_prefix}"
                f"Subquestion:\n{subq}\n\n"
                f"Based on the above context and your legal knowledge, please analyze the subquestion."
            )

            phase_outputs = {}
            for phase_idx, phase in enumerate(self.phases, start=1):
                print(f"  - Phase '{phase}' (Step {phase_idx}/{len(self.phases)})...")
                response = self.perform_phase_analysis(
                    question=enhanced_question,
                    phase=phase,
                    step=phase_idx
                )
                phase_outputs[phase] = response

            results[subq_label] = {
                "subquestion_text": subq,
                "phase_results": phase_outputs
            }
            print (results)

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
    
    def split_into_subquestions(self, full_text: str) -> List[str]:
        """
        Splits a full hypothetical into subquestions using common legal formatting:
        numbered (e.g. "1.") and lettered (e.g. "(a)").
        
        Returns a list of subquestion strings.
        """
        import re

        full_text = full_text.strip()

        # Match starts of numbered or lettered subquestions
        pattern = re.compile(r"(^\d+\.\s|^\([a-z]\)\s)", re.IGNORECASE | re.MULTILINE)

        # Find all starting positions
        matches = list(pattern.finditer(full_text))
        if not matches:
            return [full_text]

        subquestions = []
        for i, match in enumerate(matches):
            start_idx = match.start()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            subq = full_text[start_idx:end_idx].strip()
            subquestions.append(subq)

        return subquestions