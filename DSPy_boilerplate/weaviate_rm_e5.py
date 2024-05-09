from typing import List, Optional, Union
from embedders import STE5Embedder
import asyncio

import dspy
from dsp.utils import dotdict
from dspy.retrieve.weaviate_rm import WeaviateRM

try:
    import weaviate
    import weaviate.classes as wvc
    from weaviate.collections.classes.grpc import HybridFusion
    from weaviate.classes.config import Property, DataType
except ImportError:
    raise ImportError(
        "The 'weaviate' extra is required to use WeaviateRM. Install it with `pip install dspy-ai[weaviate]`",
    )

import nltk
stemmer = nltk.stem.SnowballStemmer("english").stem
import re
import string

    

class WeaviateRME5(WeaviateRM):
    """
    A retrieval module that uses Weaviate to return the top passages for a given query.

    Assumes that a Weaviate collection has been created and populated with the following payload:
        - content: The text of the passage

    Args:
        weaviate_collection_name (str): The name of the Weaviate collection.
        weaviate_client (WeaviateClient): An instance of the Weaviate client.
        k (int, optional): The default number of top passages to retrieve. Defaults to 3.
        weaviate_collection_text_key (str, optional): The key in the collection with the content. Defaults to content.
        weaviate_alpha (float, optional): The alpha value for the hybrid query. Defaults to 0.5.
        weaviate_fusion_type (wvc.HybridFusion, optional): The fusion type for the query. Defaults to RELATIVE_SCORE.

    Examples:
        Below is a code snippet that shows how to use Weaviate as the default retriver:
        ```python
        import weaviate

        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        weaviate_client = weaviate.Client("your-path-here")
        retriever_model = WeaviateRM(weaviate_collection_name="my_collection_name",
                                     weaviate_collection_text_key="content", 
                                     weaviate_client=weaviate_client)
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use Weaviate in the forward() function of a module
        ```python
        self.retrieve = WeaviateRM("my_collection_name", weaviate_client=weaviate_client, k=num_passages)
        ```
    """

    def __init__(self, 
                weaviate_collection_name: str, 
                weaviate_client: weaviate.WeaviateClient,
                k: int = 3,
                embedder = STE5Embedder(),
                weaviate_collection_text_key: Optional[str] = "full_text",
                weaviate_alpha: Optional[float] = 0.5,
                weaviate_fusion_type: Optional[HybridFusion] = HybridFusion.RELATIVE_SCORE,

        ):
        self.embedder = embedder
        super().__init__(weaviate_collection_name,
                weaviate_client,
                k,
                weaviate_collection_text_key,
                weaviate_alpha,
                weaviate_fusion_type)
        
    async def encode_query(self, query: str) -> List[float]:
        return self.embedder.encode_query(query)
    
    def lexical_preprocessing(self, text: str) -> str:
        translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
        text = text.translate(translator)

        # Stemming
        text = text.split()
        text = [stemmer(word) for word in text]
        return " ".join(text)

    def create_collection(self, corpus: List[str]):
        async def create_collection_async(self, corpus: List[str]):
            if self._weaviate_client.collections.exists(self._weaviate_collection_name):
                self._weaviate_client.collections.delete(self._weaviate_collection_name) 
            collection = self._weaviate_client.collections.create(
                    name=self._weaviate_collection_name,
                    vector_index_config=wvc.config.Configure.VectorIndex.flat(
                        distance_metric=wvc.config.VectorDistances.COSINE
                    ),
                    properties=[
                        Property(name="processed_text", 
                                data_type=DataType.TEXT
                                ),
                        Property(name="title",
                                data_type=DataType.TEXT,
                                index_searchable=False,
                                index_filterable=False,
                                ),
                        Property(name="full_text",
                                data_type=DataType.TEXT,
                                index_searchable=False,
                                index_filterable=False,
                                )
                        
                    ],
                    inverted_index_config=wvc.config.Configure.inverted_index(
                        bm25_b=0.75,
                        bm25_k1=1.25
                    )
            
            )
            embeddings = await self.embedder.encode_corpus(corpus, batch_size=2)
            
            with collection.batch.dynamic() as batch:
                for document, embedding in zip(corpus, embeddings):
                    batch.add_object(
                        properties={"processed_text": self.lexical_preprocessing(document["text"]), "full_text": document["text"], "title": document["title"]},
                        vector=embedding
                    )
        return asyncio.run(create_collection_async(self, corpus))



    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None, **kwargs) -> dspy.Prediction:
        """Search with Weaviate for self.k top passages for query

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.
        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        async def forward_async(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None, **kwargs) -> dspy.Prediction:
            k = k if k is not None else self.k
            queries = (
                [query_or_queries]
                if isinstance(query_or_queries, str)
                else query_or_queries
            )
            queries = [q for q in queries if q]
            passages = []
            embeddings = await asyncio.gather(
                *[self.encode_query(query) for query in queries]
            )
            for query, vector in zip(queries, embeddings):
                collection = self._weaviate_client.collections.get(self._weaviate_collection_name)
                # vector = await self.encode_query(query)
                results = collection.query.hybrid(query=self.lexical_preprocessing(query),
                                                vector=vector,
                                                query_properties=["processed_text"],
                                                limit=k,
                                                alpha=self._weaviate_alpha,
                                                fusion_type=self._weaviate_fusion_type,
                                                return_metadata=wvc.query.MetadataQuery(
                                                    distance=True, score=True),
                                                **kwargs,
                                                )

                parsed_results = [result.properties[self._weaviate_collection_text_key] for result in results.objects]
                passages.extend(dotdict({"long_text": d}) for d in parsed_results)

            # Return type not changed, needs to be a Prediction object. But other code will break if we change it.
            return passages
        
        return asyncio.run(forward_async(self, query_or_queries, k, **kwargs))
