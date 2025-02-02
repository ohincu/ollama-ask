#############################################################################
# Loading libraries
#############################################################################
import pandas as pd

# LLM
from llama_index.readers.database import DatabaseReader
from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader, download_loader, Settings #, StorageContext, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SimpleNodeParser
from typing import List
from pydantic import BaseModel, Field
import requests

#############################################################################
# Setting models
#############################################################################
class Settings:
    models = {
        "llama_1b": Ollama(model="llama3.2:1b", request_timeout=120.0),
        "deepseek": Ollama(model="deepseek-r1:1.5b", request_timeout=120.0)
    }
deepseek_llm = Ollama(model="deepseek-r1:1.5b", request_timeout=120.0)
 # Set the model you want to use
Settings.current_model = Settings.models["deepseek"] 


#############################################################################
# Defining functions
#############################################################################
class OllamaEmbedding(BaseEmbedding, BaseModel):
    model: str = Field(default="deepseek-r1:1.5b", description="The model name for generating embeddings.")
    base_url: str = Field(default="http://localhost:11434", description="The base URL for the Ollama server.")


    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        endpoint = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "text": text
        }
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()  # Raise an error if the request failed
        return response.json().get("embedding", [])

    def get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text string.

        Args:
            text (str): The input text.

        Returns:
            List[float]: The embedding vector.
        """
        return self._get_embedding(text)

    def get_texts_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of text strings.

        Args:
            texts (List[str]): A list of text strings.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        return [self._get_embedding(text) for text in texts]

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        endpoint = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "text": text
        }
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()  # Raise an error if the request failed
        return response.json().get("embedding", [])

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a query string.

        Args:
            query (str): The input query.

        Returns:
            List[float]: The embedding vector.
        """
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Asynchronous method to get embedding for a query string.

        Args:
            query (str): The input query.

        Returns:
            List[float]: The embedding vector.
        """
        return self._get_query_embedding(query)

embedding_model = OllamaEmbedding(model = "deepseek-r1:1.5b", base_url = "http://localhost:11434")

#############################################################################
# Transforming and loading data
#############################################################################
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter

documents = SimpleDirectoryReader("ollama/src/data").load_data()
pipeline = IngestionPipeline(transformations=[TokenTextSplitter(), ...])
nodes = pipeline.run(documents=documents)

index = GPTVectorStoreIndex.from_documents( documents,llm = deepseek_llm, embed_model=embedding_model, 
                                            show_progress = True, 
                                            chunk_size=1024, 
                                            max_tokens=4096
                                            )
for i, doc in enumerate(documents[:10]):  # Print first 10 documents
    print(f"Document {i+1}: {doc}")

print(f"Number of documents processed: {len(documents)}")

print(f"Number of indexed documents: {len(index.docstore.docs.keys())}") # Lists all indexed doc IDs

sample_doc_id = list(index.docstore.docs.keys())[0]  # Get first document ID
print(index.docstore.docs[sample_doc_id].text) 

#############################################################################
# Asking questions
#############################################################################

response = query_engine.query("How many messages are in the text file?")
print(response)
