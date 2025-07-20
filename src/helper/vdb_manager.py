"""
This module provides a high-level interface for managing and interacting with a ChromaDB vector database.
It is designed to handle the storage, chunking, and retrieval of legal documents.
"""

import os
import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import random
import threading
from typing import Any, Dict, List, Optional, Tuple, Literal

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Attempt to import LlamaIndex for advanced text splitting
try:
    import nltk
    from llama_index.core.text_splitter import SentenceSplitter

    # Ensure NLTK 'punkt' tokenizer is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt', quiet=True)
    
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    print("Warning: LlamaIndex not available. Using fallback text splitter.")
    SentenceSplitter = None
    LLAMA_INDEX_AVAILABLE = False


class SimpleSentenceSplitter:
    """
    A basic fallback sentence splitter that does not rely on NLTK or LlamaIndex.
    It splits text by sentences and then groups them into chunks.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Splits the given text into chunks.

        Args:
            text: The text to split.

        Returns:
            A list of text chunks.
        """
        if not text:
            return []

        # Split text into sentences using a simple regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Create overlap for the next chunk
                overlap = current_chunk[-self.chunk_overlap:]
                current_chunk = overlap + " " + sentence
            else:
                current_chunk += (" " if current_chunk else "") + sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks


class VectorDBManager:
    """
    Manages interactions with a ChromaDB vector database, including data loading,
    chunking, and querying, with special handling for legal documents.
    """
    DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
    DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

    def __init__(self, 
                 client_name: str, 
                 allowed_collections: List[str], 
                 embedding_model_name: str = DEFAULT_EMBEDDING_MODEL, 
                 device: str = "cpu", 
                 use_openai: bool = True):
        """
        Initializes the VectorDBManager.

        Args:
            client_name: Name for the ChromaDB client, used as a directory name.
            allowed_collections: A list of collection names to be managed.
            embedding_model_name: The name of the SentenceTransformer model to use.
            device: The device to run the embedding model on ('cpu' or 'cuda').
            use_openai: If True, use OpenAI's embedding API instead of a local model.
        """
        self.client_name = client_name
        self.use_openai = use_openai
        self.embedding_fn = self._get_embedding_function(embedding_model_name, device)
        
        self.client = self._create_client()
        self.collections = {
            name: self._get_or_create_collection(name) for name in allowed_collections
        }
        self._lock = threading.Lock()

    def _get_embedding_function(self, model_name: str, device: str) -> Any:
        """
        Initializes and returns the appropriate embedding function.
        """
        if self.use_openai:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in .env file.")
            
            # Set the environment variable for ChromaDB persistence if not already set
            if "CHROMA_OPENAI_API_KEY" not in os.environ:
                os.environ["CHROMA_OPENAI_API_KEY"] = openai_api_key

            print(f"✅ Using OpenAI embeddings ({self.DEFAULT_OPENAI_EMBEDDING_MODEL})")
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name=self.DEFAULT_OPENAI_EMBEDDING_MODEL
            )
        else:
            print(f"✅ Using local embeddings ({model_name} on {device})")
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name, device=device
            )

    def _create_client(self) -> chromadb.PersistentClient:
        """
        Creates and returns a persistent ChromaDB client.
        """
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        client_path = os.path.join(base_path, "vdb", self.client_name)
        os.makedirs(client_path, exist_ok=True)
        return chromadb.PersistentClient(path=client_path)

    def _get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """
        Gets an existing collection or creates a new one.
        """
        # The collection name is now the definitive name, no prefixing.
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add_to_collection(self, 
                          collection_name: str, 
                          documents: List[str], 
                          metadatas: List[Dict[str, Any]], 
                          ids: List[str],
                          skip_if_exists: bool = True):
        """
        Adds a batch of documents to a specified collection.

        Args:
            collection_name: The name of the collection.
            documents: A list of document texts.
            metadatas: A list of metadata dictionaries.
            ids: A list of unique IDs for the documents.
            skip_if_exists: If True, documents with existing IDs will be skipped.
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' is not allowed.")
        
        collection = self.collections[collection_name]

        if skip_if_exists:
            existing_ids = set(collection.get(ids=ids)['ids'])
            if existing_ids:
                new_docs, new_metas, new_ids = [], [], []
                for doc, meta, id_ in zip(documents, metadatas, ids):
                    if id_ not in existing_ids:
                        new_docs.append(doc)
                        new_metas.append(meta)
                        new_ids.append(id_)
                documents, metadatas, ids = new_docs, new_metas, new_ids
        
        if not documents:
            return

        self._add_with_retry(collection, documents, metadatas, ids)

    def _add_with_retry(self, collection: chromadb.Collection, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str], max_retries: int = 5):
        """
        Adds documents to a collection with a retry mechanism for handling rate limits.
        """
        attempt = 0
        while attempt <= max_retries:
            try:
                collection.add(documents=documents, metadatas=metadatas, ids=ids)
                return
            except Exception as e:
                if "rate_limit_exceeded" in str(e) and attempt < max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    attempt += 1
                else:
                    raise e

    def query_collection(self, 
                         collection_name: str, 
                         query_text: str, 
                         n_results: int = 10, 
                         where_clause: Optional[Dict[str, Any]] = None,
                         distance_threshold: float = 1.0) -> Dict[str, List[Any]]:
        """
        Queries a collection for documents similar to the query text.

        Args:
            collection_name: The name of the collection to query.
            query_text: The text to search for.
            n_results: The maximum number of results to return.
            where_clause: An optional filter to apply to the search.
            distance_threshold: The maximum distance for a result to be included.
                                Lower values mean higher similarity.

        Returns:
            A dictionary containing the filtered and sorted results.
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' is not allowed.")

        result = self.collections[collection_name].query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        # Filter results by distance threshold and sort by date
        filtered_results = []
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            if dist <= distance_threshold:
                filtered_results.append((doc, meta, dist))
        
        # Sort by case_date if available
        filtered_results.sort(key=lambda x: x[1].get("case_date", ""), reverse=True)

        return {
            "documents": [item[0] for item in filtered_results],
            "metadatas": [item[1] for item in filtered_results],
            "distances": [item[2] for item in filtered_results]
        }

    def load_legal_cases_from_jsonl(self,
                                    jsonl_file_path: str,
                                    collection_name: str,
                                    chunk_size: int = 1000,
                                    chunk_overlap: int = 200,
                                    batch_size: int = 50,
                                    max_workers: int = 4,
                                    mode: Literal['sequential', 'batch', 'parallel'] = 'batch',
                                    source_filter: Optional[str] = None):
        """
        Loads legal cases from a JSONL file, chunks them, and adds them to the database.

        Args:
            jsonl_file_path: Path to the JSONL file.
            collection_name: The collection to add the cases to.
            chunk_size: The size of each text chunk.
            chunk_overlap: The overlap between chunks.
            batch_size: The number of documents to process in a batch.
            max_workers: The number of parallel workers for 'parallel' mode.
            mode: The processing mode ('sequential', 'batch', 'parallel').
            source_filter: Optional source to filter by.
        """
        if not os.path.exists(jsonl_file_path):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file_path}")

        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data = [json.loads(line) for line in lines if line.strip()]

        if source_filter:
            data = [item for item in data if item.get('source') == source_filter]

        if not data:
            print(f"No cases found with source '{source_filter}' in {jsonl_file_path}")
            return

        text_splitter = (SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                         if LLAMA_INDEX_AVAILABLE else SimpleSentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap))

        print(f"Processing {len(data)} cases from {jsonl_file_path} in '{mode}' mode...")

        all_chunks = []
        for case_data in tqdm(data, desc="Preparing Chunks"):
            case_id = case_data.get('version_id')
            if not case_id:
                print(f"Skipping case due to missing 'version_id': {case_data.get('citation', 'N/A')}")
                continue
            all_chunks.extend(self._prepare_case_chunks(case_id, case_data, text_splitter, collection_name, is_jsonl=True))

        if mode == 'parallel':
            self._process_chunks_parallel(collection_name, all_chunks, batch_size, max_workers)
        else:  # batch and sequential
            self._process_chunks_sequentially(collection_name, all_chunks, batch_size)

    def load_legal_cases_from_json(self, 
                                   json_file_path: str, 
                                   collection_name: str,
                                   chunk_size: int = 1000, 
                                   chunk_overlap: int = 200, 
                                   batch_size: int = 50,
                                   max_workers: int = 4,
                                   mode: Literal['sequential', 'batch', 'parallel'] = 'batch'):
        """
        Loads legal cases from a JSON file, chunks them, and adds them to the database.

        Args:
            json_file_path: Path to the JSON file.
            collection_name: The collection to add the cases to.
            chunk_size: The size of each text chunk.
            chunk_overlap: The overlap between chunks.
            batch_size: The number of documents to process in a batch.
            max_workers: The number of parallel workers for 'parallel' mode.
            mode: The processing mode ('sequential', 'batch', 'parallel').
        """
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")

        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        text_splitter = (SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                         if LLAMA_INDEX_AVAILABLE else SimpleSentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap))

        print(f"Processing {len(data)} cases from {json_file_path} in '{mode}' mode...")
        
        all_chunks = []
        for case_id, case_data in tqdm(data.items(), desc="Preparing Chunks"):
            all_chunks.extend(self._prepare_case_chunks(case_id, case_data, text_splitter, collection_name))

        if mode == 'parallel':
            self._process_chunks_parallel(collection_name, all_chunks, batch_size, max_workers)
        else: # batch and sequential
            self._process_chunks_sequentially(collection_name, all_chunks, batch_size)

    def _prepare_case_chunks(self, case_id: str, case_data: Dict[str, Any], text_splitter, collection_name: str, is_jsonl: bool = False) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Prepares chunks for a single case."""
        raw_text = case_data.get('text' if is_jsonl else 'rawtext', '')
        if not raw_text:
            return []

        chunks = text_splitter.split_text(raw_text)
        base_metadata = self._create_base_metadata(case_id, case_data, collection_name, is_jsonl=is_jsonl)
        
        prepared_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{case_id}_chunk_{i}"
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_id': chunk_id,
                'chunk_index': i,
                'total_chunks': len(chunks),
            })
            prepared_chunks.append((chunk_id, chunk_text, chunk_metadata))
        return prepared_chunks

    def _create_base_metadata(self, case_id: str, case_data: Dict[str, Any], collection_name: str, is_jsonl: bool = False) -> Dict[str, Any]:
        """Creates a base metadata dictionary for a case."""
        if is_jsonl:
            return {
                'case_id': case_id,
                'case_title': self._safe_metadata_value(case_data.get('citation', '')),
                'court': self._safe_metadata_value(case_data.get('jurisdiction', '')),
                'decision_date': self._safe_metadata_value(case_data.get('date', '')),
                'source': self._safe_metadata_value(case_data.get('source', collection_name))
            }
        return {
            'case_id': case_id,
            'case_title': self._safe_metadata_value(case_data.get('case_title', '')),
            'court': self._safe_metadata_value(case_data.get('court', '')),
            'decision_date': self._safe_metadata_value(case_data.get('decision_date', '')),
            'source': collection_name
        }

    def _process_chunks_sequentially(self, collection_name: str, all_chunks: List[Tuple], batch_size: int):
        """Processes chunks sequentially in batches."""
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Adding to DB"):
            batch = all_chunks[i:i + batch_size]
            ids = [c[0] for c in batch]
            docs = [c[1] for c in batch]
            metas = [c[2] for c in batch]
            self.add_to_collection(collection_name, docs, metas, ids)

    def _process_chunks_parallel(self, collection_name: str, all_chunks: List[Tuple], batch_size: int, max_workers: int):
        """Processes chunks in parallel."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                ids = [c[0] for c in batch]
                docs = [c[1] for c in batch]
                metas = [c[2] for c in batch]
                futures.append(executor.submit(self.add_to_collection, collection_name, docs, metas, ids))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Adding to DB (Parallel)"):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred while adding a batch: {e}")

    @staticmethod
    def _safe_metadata_value(value: Any) -> str:
        """
        Ensures metadata values are in a format compatible with ChromaDB.
        """
        if value is None:
            return ""
        if isinstance(value, list):
            return ', '.join(map(str, value))
        return str(value)

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Retrieves statistics for a given collection.

        Args:
            collection_name: The name of the collection.

        Returns:
            A dictionary with collection statistics.
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' is not allowed.")
        
        collection = self.collections[collection_name]
        count = collection.count()
        
        stats = {'count': count}
        if count > 0:
            sample = collection.peek(limit=1)
            stats['sample'] = sample
            
        return stats

    def reset_collection(self, collection_name: str):
        """
        Deletes all data from a collection and recreates it.

        Args:
            collection_name: The name of the collection to reset.
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' is not allowed.")
        
        full_name = collection_name
        self.client.delete_collection(name=full_name)
        print(f"✅ Deleted collection '{full_name}'.")
        self.collections[collection_name] = self._get_or_create_collection(collection_name)
        print(f"✅ Recreated collection '{full_name}'.")