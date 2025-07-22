import numpy as np

from google import genai
from google.genai import types
from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError
from google.api_core import exceptions as google_exceptions

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage

global_openai_async_client = None
global_azure_openai_async_client = None
global_gemini_async_client = None


def get_gemini_async_client_instance(api_key=None):
    """Get or create a Gemini async client instance using the new google-genai SDK."""
    global global_gemini_async_client
    if global_gemini_async_client is None:
        if api_key:
            global_gemini_async_client = genai.Client(api_key=api_key)
        else:
            global_gemini_async_client = genai.Client()
    return global_gemini_async_client


def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (google_exceptions.ResourceExhausted, google_exceptions.GoogleAPICallError)
    ),
)
async def gemini_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    gemini_client = get_gemini_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

    # Prepare contents in the new format
    contents = []
    if history_messages:
        contents.extend(history_messages)
    contents.append(prompt)

    # Prepare config for the new API
    config_params = {}
    if system_prompt:
        config_params["system_instruction"] = system_prompt

    if "max_tokens" in kwargs:
        config_params["max_output_tokens"] = kwargs.pop("max_tokens")

    # Add any remaining kwargs to config
    for key, value in kwargs.items():
        if key not in ["hashing_kv"]:
            config_params[key] = value

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, contents, system_prompt)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    # Use the new async API structure
    response = await gemini_client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(**config_params) if config_params else None,
    )

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": response.text, "model": model}})
        await hashing_kv.index_done_callback()
    return response.text


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_35_turbo_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-3.5-turbo",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


async def gemini_pro_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await gemini_complete_if_cache(
        "gemini-2.5-pro",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gemini_flash_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await gemini_complete_if_cache(
        "gemini-2.5-flash",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=2048)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (google_exceptions.ResourceExhausted, google_exceptions.GoogleAPICallError)
    ),
)
async def gemini_embedding(texts: list[str]) -> np.ndarray:
    """Generate embeddings using the new google-genai SDK."""
    gemini_client = get_gemini_async_client_instance()

    # Use the new embedding API
    response = await gemini_client.aio.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=texts,
        config=types.EmbedContentConfig(task_type="retrieval_document"),
    )

    # Extract embeddings from the response
    embeddings = [embedding.values for embedding in response.embeddings]
    return np.array(embeddings)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    azure_openai_client = get_azure_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def azure_gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
