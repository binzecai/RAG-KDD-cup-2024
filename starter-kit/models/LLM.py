import os
from typing import Dict, List

import numpy as np
import torch
from models.utils import trim_predictions_to_max_token_length
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

######################################################################################################
######################################################################################################
###
### IMPORTANT !!!
### Before submitting, please follow the instructions in the docs below to download and check in :
### the model weighs.
###
###  https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/download_baseline_model_weights.md
###
###
### DISCLAIMER: This baseline has NOT been tuned for performance
###             or efficiency, and is provided as is for demonstration.
######################################################################################################


# Load the environment variable that specifies the URL of the MockAPI. This URL is essential
# for accessing the correct API endpoint in Task 2 and Task 3. The value of this environment variable
# may vary across different evaluation settings, emphasizing the importance of dynamically obtaining
# the API URL to ensure accurate endpoint communication.

# Please refer to https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/crag-mock-api
# for more information on the MockAPI.
#
# **Note**: This environment variable will not be available for Task 1 evaluations.
CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")


class LLMPredictor:
    def __init__(self, model_path, device="cuda", **kwargs):

        self.prompt_template = \
        """
        You are given a quesition and references which may or may not help answer the question. Your goal is to answer the question in as few words as possible.\n
        Don't repeat the question in the answer. Give the answer directly!\n
        ### Question
        {query}

        ### References 
        {references}

        ### Answer
        """
        # Configuration for model quantization to improve performance, using 4-bit precision.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )

        # Specify the large language model to be used.
        # model_name = "models/meta-llama/Llama-2-7b-chat-hf"

        # Load the tokenizer for the specified model.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load the large language model with the specified quantization configuration.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )

        # Initialize a text generation pipeline with the loaded model and tokenizer.
        self.generation_pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=10,
        )

        self.max_token = 4096

        self.kwargs = kwargs
        # self.device = torch.device(device)
        # self.model.eval()
        # self.model.to(self.device)
        print('successful  load LLM', model_path)

        
    def predict(self, references, query):
        final_prompt = self.prompt_template.format(
            query=query, references=references
        )

        # Generate an answer using the formatted prompt.
        result = self.generation_pipe(final_prompt)
        result = result[0]["generated_text"]

        try:
            # Extract the answer from the generated text.
            answer = result.split("### Answer\n")[-1]
        except IndexError:
            # If the model fails to generate an answer, return a default response.
            answer = "I don't know"
        if len(answer.split('\n')) > 1:
            answer = answer.split('\n')[0]
        answer = answer.lower()
        # Trim the prediction to a maximum of 75 tokens (this function needs to be defined).
        trimmed_answer = trim_predictions_to_max_token_length(answer)

        return trimmed_answer