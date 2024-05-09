
import asyncio
import os
import logging
import tarfile
import bz2
import json

import weaviate
from embedders import APIEmbedder
from util import download_url, extract_archive
from weaviate_rm_e5 import WeaviateRME5


logger = logging.getLogger(__name__)

# data_path = download_and_extract("https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2", "/AE/", "datasets")
# print(data_path)
def main():
    out_dir = "datasets"
    url = "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2"
    chunk_size = 1024
    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split("/")[-1]
    zip_file = os.path.join(out_dir, dataset)

    if not os.path.isfile(zip_file):
        logger.info("Downloading {} ...".format(dataset))
        download_url(url, zip_file, chunk_size)

    total = 0
    corpus = []
    with tarfile.open(zip_file, "r:bz2") as tar:
        for member in tar:
            if member.name.endswith('.bz2'):
                bz2_file = bz2.BZ2File(tar.extractfile(member), 'rb')
                with bz2.BZ2File(tar.extractfile(member), 'rb') as file:
                    json_data = file.read().decode('utf-8')
                    for line in json_data.splitlines():
                        data = json.loads(line)
                        # print(data["text"])
                        if(len(data["text"]) > 1):
                            corpus.append({"title": data["title"], "text": "".join(data["text"][1])})
                            total += 1
                    if(total > 100):
                        break
    print(corpus)

    embedder = APIEmbedder()
    with weaviate.connect_to_local(port=8080) as weaviate_client:
        retriever_model = WeaviateRME5("wikipedia", embedder=embedder, weaviate_client=weaviate_client, weaviate_collection_text_key="full_text", k=10)
        retriever_model.create_collection(corpus)
        data = retriever_model("Who is Arnaldo David Cézar Coelho?", 10)
        print(data)

if __name__ ==  '__main__':
    main()