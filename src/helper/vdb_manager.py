import os
import chromadb
from chromadb.utils import embedding_functions
import uuid

num_results = 10000000000000000  
char_length = 3000  

class db:
    def __init__(self, client_name, allowed_collections, EmbeddingModelName="BAAI/bge-m3", device="cpu"):
        """
        initialize a db instance
        """
        self.client_name = client_name
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EmbeddingModelName, device=device
        )
        self.client = self._create_client()
        self.collections = {
            name: self._create_collection(name) for name in allowed_collections
        }

    def _create_client(self):
        """
        create a chromadb client instance for the given client_name
        """
        client_path = os.path.join("vdb", self.client_name)
        os.makedirs(client_path, exist_ok=True)
        return chromadb.PersistentClient(path=client_path)

    def _create_collection(self, collection_name):
        """
        create a chromadb collection instance for the given collection_name
        """
        return self.client.get_or_create_collection(
            name=f"{self.client_name}_{collection_name}",
            embedding_function=self.embedding_fn,
        )

    def add_to_collection(self, collection_name, id=None, document="", metadata=None):
        """
        add a document to a specific collection
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not accessible.")
        collection = self.collections[collection_name]
        collection.add(
            documents=[document[:char_length]],
            metadatas=[metadata] if metadata else None,
            ids=[id] if id else [str(uuid.uuid4())]
        )

    def query_collection(self, collection_name, query_text, tags=None, n_results=num_results,
                         include=["documents", "metadatas", "distances"], similarity_threshold=0.7):
        """
        queries a specific collection in the chromadb database
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not accessible.")
        where_clause = {}
        if tags:
            where_clause["tags"] = {"$in": tags}
        result = self.collections[collection_name].query(
            query_texts=[query_text],
            n_results=n_results,
            include=include,
            where=where_clause,
        )
        return self.filter_results(result, similarity_threshold)

    @staticmethod
    def filter_results(result, similarity_threshold=0.7):
        """
        filters and sorts query results based on a similarity threshold and case date
        """
        filtered_results = []
        for doc, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            if 1 - distance >= similarity_threshold:
                filtered_results.append((doc, metadata, distance))
        sorted_results = sorted(filtered_results, key=lambda x: x[1].get("case_date", ""), reverse=True)
        return {
            "documents": [item[0] for item in sorted_results],
            "metadatas": [item[1] for item in sorted_results],
            "distances": [item[2] for item in sorted_results]
        }

    # OLD METHODS FOR QUERYING COLLECTIONS, DOCUMENTS and METADATA

    def query_internal_collection(self, query_text, tags=None, n_results=num_results,
                                   include=["documents", "metadatas", "distances"], similarity_threshold=0.7):
        return self.query_collection("internal-collection", query_text, tags, n_results, include, similarity_threshold)

    def query_external_collection(self, query_text, tags=None, n_results=num_results,
                                   include=["documents", "metadatas", "distances"], similarity_threshold=0.7):
        return self.query_collection("external-collection", query_text, tags, n_results, include, similarity_threshold)

    def query_internal_documents(self, query_text, tags=None, similarity_threshold=0.7):
        result = self.query_internal_collection(query_text=query_text,
                                                tags=tags,
                                                include=["documents"],
                                                similarity_threshold=similarity_threshold)
        return result["documents"]

    def query_external_documents(self, query_text, tags=None, similarity_threshold=0.7):
        result = self.query_external_collection(query_text=query_text,
                                                tags=tags,
                                                include=["documents"],
                                                similarity_threshold=similarity_threshold)
        return result["documents"]

    def query_internal_metadatas(self, query_text, tags=None, similarity_threshold=0.7):
        result = self.query_internal_collection(query_text=query_text,
                                                tags=tags,
                                                include=["metadatas"],
                                                similarity_threshold=similarity_threshold)
        return result["metadatas"]

    def query_external_metadatas(self, query_text, tags=None, similarity_threshold=0.7):
        result = self.query_external_collection(query_text=query_text,
                                                tags=tags,
                                                include=["metadatas"],
                                                similarity_threshold=similarity_threshold)
        return result["metadatas"]