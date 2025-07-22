import logging
import numpy as np
import yaml
from hirag import HiRAG, QueryParam
from google import genai
from google.genai import types
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract configurations
GOOGLE_API_KEY = config["google"]["api_key"]
MODEL = config["google"]["model"]
EMBEDDING_MODEL = config["google"]["embedding_model"]

# Initialize the client with the new API
client = genai.Client(api_key=GOOGLE_API_KEY)


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


@wrap_embedding_func_with_attrs(
    embedding_dim=config["model_params"]["google_embedding_dim"],
    max_token_size=config["model_params"]["max_token_size"],
)
async def google_embedding(texts: list[str]) -> np.ndarray:
    """Generate embeddings using the new google-genai SDK."""
    response = await client.aio.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="retrieval_document"),
    )

    # Extract embeddings from the response
    embeddings = [embedding.values for embedding in response.embeddings]
    return np.array(embeddings)


async def google_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Generate content using the new google-genai SDK with caching support."""

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

    # Prepare contents in the new format
    contents = []
    if history_messages:
        for msg in history_messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, contents, system_prompt)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    try:
        # Prepare config for the new API
        config_params = {}
        if system_prompt:
            config_params["system_instruction"] = system_prompt

        if "response_format" in kwargs:
            if kwargs["response_format"]["type"] == "json_object":
                config_params["response_mime_type"] = "application/json"

        # Add any remaining kwargs to config
        for key, value in kwargs.items():
            if key not in ["hashing_kv", "response_format"]:
                if key == "max_tokens":
                    config_params["max_output_tokens"] = value
                else:
                    config_params[key] = value

        # Use the new async API structure
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(**config_params)
            if config_params
            else None,
        )

        result_text = response.text

    except Exception as e:
        logging.info(e)
        return "ERROR"

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result_text, "model": MODEL}})
        await hashing_kv.index_done_callback()
    # -----------------------------------------------------
    return result_text


graph_func = HiRAG(
    working_dir=config["hirag"]["working_dir"],
    enable_llm_cache=config["hirag"]["enable_llm_cache"],
    embedding_func=google_embedding,
    best_model_func=google_model_if_cache,
    cheap_model_func=google_model_if_cache,
    enable_hierachical_mode=config["hirag"]["enable_hierachical_mode"],
    embedding_func_max_async=config["hirag"]["embedding_func_max_async"],
    enable_naive_rag=config["hirag"]["enable_naive_rag"],
)

# comment this if the working directory has already been indexed
with open("./asd.txt") as f:
    graph_func.insert(f.read())

print("Perform hi search:")
print(graph_func.query("What does this text talk about?", param=QueryParam(mode="hi")))
