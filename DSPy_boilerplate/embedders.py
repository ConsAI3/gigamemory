import asyncio
import os
from typing import Dict, List
import pickle
import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm


class APIEmbedder:
    def __init__(
        self,
        model_path="multilingual-e5-large",
        base_url="https://paradigm-dev.lighton.ai/api/v2",
        sep=" ",
        **kwargs,
    ):
        self.client = AsyncOpenAI(
            api_key=os.getenv("PARADIGM_API_KEY_DEV"),
            base_url=base_url,
        )
        self.model_name = model_path
        self.sep = sep

    async def get_embedding(self, text: str) -> np.ndarray:
        response = await self.client.embeddings.create(
            model=self.model_name, input=text
        )
        return response.data[0].embedding
    
    def sync_get_embedding(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model_name, input=text
        )
        return response.data[0].embedding

    async def encode_query(self, query: str) -> np.ndarray:
        return await self.get_embedding("query: " + query)

    # Encoding query function (Returns: Query embeddings as numpy array)
    async def encode_queries(
        self, queries: List[str], batch_size: int, **kwargs
    ) -> np.ndarray:
        queries = ["query: " + query for query in queries]
        # TODO: can we get the embedding size from the model?
        embeddings = np.zeros((len(queries), 1024))
        pbar = tqdm(total=len(queries))
        index = 0
        while index < len(queries):
            end = min(index + batch_size, len(queries))
            embeddings[index:end] = await asyncio.gather(
                *[self.get_embedding(query) for query in queries[index:end]]
            )
            index += batch_size
            pbar.update(batch_size)
        return embeddings

    # Encoding corpus function (Returns: Document embeddings as numpy array)
    async def encode_corpus(
        self, corpus: List[Dict[str, str]], batch_size: int, **kwargs
    ) -> np.ndarray:
        if isinstance(corpus, dict):
            sentences = [
                (
                    "passage: " + corpus["title"][i] + self.sep + corpus["text"][i]
                ).strip()
                if "title" in corpus
                else "passage: " + corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                ("passage: " + doc["title"] + self.sep + doc["text"]).strip()
                if "title" in doc
                else "passage: " + doc["text"].strip()
                for doc in corpus
            ]
        embeddings = np.zeros((len(sentences), 1024))
        pbar = tqdm(total=len(sentences))
        index = 0
        while index < len(sentences):
            end = min(index + batch_size, len(sentences))
            embeddings[index:end] = await asyncio.gather(
                *[self.get_embedding(sentence) for sentence in sentences[index:end]]
            )
            index += batch_size
            pbar.update(batch_size)
        return embeddings
    

class STE5Embedder:
    def __init__(
        self,
        model_name_or_path="intfloat/multilingual-e5-large",
        sep: str = " ",
        **kwargs,
    ):
        self.model = SentenceTransformer(model_name_or_path)
        self.sep = sep

    async def encode_query(self, query: str) -> List[float]:
        print("St encoder")
        return self.model.encode("query: " + query).tolist()
    
    # Encoding query function (Returns: Query embeddings as numpy array)
    async def encode_queries(
        self, queries: List[str], batch_size: int, **kwargs
    ) -> np.ndarray:
        queries = ["query: " + query for query in queries]
        return self.model.encode(queries, batch_size=batch_size, show_progress_bar=True)

    # Encoding corpus function (Returns: Document embeddings as numpy array)
    async def encode_corpus(
        self, corpus: List[Dict[str, str]], batch_size: int, **kwargs
    ) -> np.ndarray:
        # print(corpus[0]["title"])
        if isinstance(corpus, dict):
            sentences = [
                (
                    "passage: " + corpus["title"][i] + self.sep + corpus["text"][i]
                ).strip()
                if "title" in corpus
                else "passage: " + corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                ("passage: " + doc["title"] + self.sep + doc["text"]).strip()
                if "title" in doc
                else "passage: " + doc["text"].strip()
                for doc in corpus
            ]
        return self.model.encode(
            sentences, batch_size=batch_size, show_progress_bar=True, **kwargs
        )