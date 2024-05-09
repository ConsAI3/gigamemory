import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.hotpotqa import HotPotQA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from openai import OpenAI
import os 
import asyncio

import weaviate

from embedders_sync import APIEmbedder
from weaviate_rm_e5 import WeaviateRME5

# Set up llama3 with a VLLM client, served on four GPUs. Please note that these URLs will not work for you; you'd need to refer to the documentation to set up your own VLLM/SGLANG server(s).
# llama3 = dspy.HFClientVLLM(model="meta-llama/Meta-Llama-3-8B-Instruct", port=None, url=["http://future-hgx-3:7411", "http://future-hgx-3:7412", "http://future-hgx-3:7413", "http://future-hgx-1:7414"], max_tokens=500, stop=('\n',))
def main():
    
    openai_api_key = os.getenv("PARADIGM_API_KEY_DEV")
    openai_api_base = "https://paradigm-dev.lighton.ai/api/v2/"

    llama3 = dspy.OpenAI(api_base = openai_api_base, 
                api_key=openai_api_key,
                model="llama-3-8b",
    )
    embedder = APIEmbedder()
    with weaviate.connect_to_local(port=8080) as weaviate_client:
        retriever_model = WeaviateRME5("wikipedia", embedder=embedder, weaviate_client=weaviate_client, weaviate_collection_text_key="full_text", k=10)
        dspy.configure(lm=llama3, rm=retriever_model)
        dataset = HotPotQA(train_seed=1, train_size=200, eval_seed=2023, dev_size=300, test_size=0)
        trainset = [x.with_inputs('question') for x in dataset.train[0:150]]
        valset = [x.with_inputs('question') for x in dataset.train[150:200]]
        devset = [x.with_inputs('question') for x in dataset.dev]
        agent = dspy.ReAct("question -> answer", tools=[dspy.Retrieve(k=1)])
        # Set up an evaluator on the first 300 examples of the devset.
        config = dict(num_threads=8, display_progress=True, display_table=5)
        evaluate = Evaluate(devset=devset, metric=dspy.evaluate.answer_exact_match, **config)
        config = dict(max_bootstrapped_demos=2, max_labeled_demos=0, num_candidate_programs=5, num_threads=32)
        tp = BootstrapFewShotWithRandomSearch(metric=dspy.evaluate.answer_exact_match, **config)
        optimized_react = tp.compile(agent, trainset=trainset, valset=valset)

        optimized_reactX = optimized_react.deepcopy()
        del optimized_reactX.candidate_programs

        config = dict(max_bootstrapped_demos=2, max_labeled_demos=0, num_candidate_programs=20, num_threads=32)
        tp = BootstrapFewShotWithRandomSearch(metric=dspy.evaluate.answer_exact_match, **config)
        optimized_react2 = tp.compile(agent, trainset=trainset, valset=valset, teacher=optimized_reactX)
        evaluate(optimized_react2)

if __name__ ==  '__main__':
    main()