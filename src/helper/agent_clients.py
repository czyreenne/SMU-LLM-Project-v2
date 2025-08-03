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

    def perform_full_structured_analysis(self, scenario: str, question: str, distance_threshold=1.0):
        """
        Performs all structured phases sequentially and returns aggregated results.
        Enhanced with relevant legal documents from vector database.
        """
        # Retrieve relevant legal documents from available collections
        collections = list(self.vdb_manager.collections.keys())
        # relevant_contexts = []
        
        scenario_with_qn = (
            f"<ProblemScenario>\n{scenario}\n</ProblemScenario>\n"
            f"<Questions>\n{question}\n</Questions>\n"
        )

        print("\nAttempting to retrieve relevant documents from vector database...")
        if not collections:
            print("No collections found in the vector database.")

        for collection in collections:
            try:
                # The collection name is now the definitive name, no prefixing.
                print(f"Querying collection: '{collection}'...")
                context_text = self.query_vdb(
                    collection_name=collection,
                    query_text=scenario_with_qn,
                    distance_threshold=distance_threshold
                )

            except Exception as e:
                print(f"Error querying collection {collection}: {str(e)}")
        
        # Create enhanced question with checks for any retrieved context
        if context_text:
            print("\nDocument retrieval successful. Enhancing question with retrieved context.")
        else:
            print("\nNo documents retrieved. Proceeding with original question.")
            context_text = "[No relevant legal documents found]"
                
        user_prompt = (
            f"{scenario_with_qn}\n"
            f"<RelevantLaw>\n{context_text}\n</RelevantLaw>\n"
            "<Instructions>\nWrite a legal analysis of the legal issues raised by the Problem Scenario. Focus on the specific Question(s) asked at the end. Ignore legal issues outside the scope of the Question(s). For each Question, start by providing a summary of all legal issues raised. Next, for each legal issue, break down your analysis into its basic elements and address each element separately in an Issue, Law, Application, and Conclusion (ILAC) block. Here is what to do within each ILAC block: \n\n1. Issue:\n  1.1. Each legal element forms an issue in the form of a question.\n  1.2. Before going through the next issue (legal element), you must first address the Law, Application and Conclusion for that issue.\n\n2. Law:\n  2.1. For each legal element, write all the legal materials relevant to that element and cite all the legislation and case law associated with that element. You may any relevant information provided under 'Relevant Law' (delimited with XML tags) but are not limited to the above. Cite any relevant statutues, regulations, guidance or case law that are authoritative in your jurisdiction.\n\n3. Application:\n  3.1. For every Application, provide as many arguments as possible from both the plaintiff and defendant's perspectives and assess the weaknesses and strengths of their case.\n  3.2. Your argument must be based on facts and events in the 'Problem Scenario'. Assume the reader does not have access to the Problem Scenario and therefore that you must spell out all the events you are referring to.\n\n4. Conclusion:\n  4.1. For each legal element, write a 'conclusion' based on your opinion.\n  4.2. It's important to consider whether the plaintiff or defendant's argument is stronger based on the available evidence and relevant burdens of proof.\n\nUse an academic tone, concise writing, postgraduate level.\n</Instructions>"
        )

        # Perform analysis through all phases
        results = {}
        for idx, phase in enumerate(self.phases, start=1):
            print(f"\nPerforming '{phase}' analysis (Step {idx}/{len(self.phases)})...")
            response = self.perform_phase_analysis(
                question=user_prompt,
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