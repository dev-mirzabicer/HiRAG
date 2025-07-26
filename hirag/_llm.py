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

# =============================================================================
# LLM Usage Recording Integration for Token Estimation Learning
# =============================================================================

async def record_llm_usage_with_context(
    token_estimator,
    call_type,
    actual_response,
    estimated_input_tokens: int = 0,
    estimated_output_tokens: int = 0,
    chunk_content: str = "",
    chunk_size: int = None,
    document_type: str = "general",
    model_name: str = "",
    success: bool = True,
    metadata: dict = None
):
    """
    Record LLM usage with rich context for learning
    
    This function should be called after any LLM completion to record
    actual vs estimated token usage for the learning system.
    
    Args:
        token_estimator: TokenEstimator instance
        call_type: LLMCallType enum value
        actual_response: The actual LLM response (to extract token usage)
        estimated_input_tokens: Pre-call token estimate for input
        estimated_output_tokens: Pre-call token estimate for output
        chunk_content: The content being processed (for context)
        chunk_size: Size of chunk in tokens/words
        document_type: Type of document (academic, technical, general)
        model_name: Name of the model used
        success: Whether the call succeeded
        metadata: Additional metadata
    """
    if not token_estimator or not token_estimator.estimation_db:
        return
    
    try:
        # Import here to avoid circular imports
        from ._token_estimation import _extract_actual_usage
        
        # Extract actual token usage from the response
        context = {
            "chunk_size": chunk_size or (len(chunk_content.split()) if chunk_content else None),
            "document_type": document_type,
            "model_name": model_name
        }
        
        actual_input_tokens, actual_output_tokens = _extract_actual_usage(actual_response, context)
        
        # If we couldn't extract from response, try to estimate from content
        if actual_input_tokens == 0 and chunk_content:
            # Rough estimation from content length
            actual_input_tokens = len(chunk_content.split()) * 1.3  # Will be made learnable
        
        if actual_output_tokens == 0 and isinstance(actual_response, str):
            actual_output_tokens = len(actual_response.split()) * 1.3  # Will be made learnable
        elif actual_output_tokens == 0 and hasattr(actual_response, 'content'):
            actual_output_tokens = len(str(actual_response.content).split()) * 1.3
        
        # Record the usage
        await token_estimator.record_actual_usage(
            call_type=call_type,
            actual_input_tokens=int(actual_input_tokens),
            actual_output_tokens=int(actual_output_tokens),
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            model_name=model_name,
            chunk_size=context.get("chunk_size"),
            document_type=document_type,
            success=success,
            metadata=metadata or {}
        )
        
    except Exception as e:
        # Don't let recording errors break the main pipeline
        logger.debug(f"Failed to record LLM usage: {e}")


def create_instrumented_llm_caller(token_estimator, call_type, model_func):
    """
    Create an instrumented version of an LLM function that automatically records usage
    
    Args:
        token_estimator: TokenEstimator instance
        call_type: LLMCallType for this function
        model_func: The LLM function to instrument
    
    Returns:
        Instrumented function that records usage automatically
    """
    async def instrumented_caller(*args, **kwargs):
        # Extract context before the call
        chunk_content = ""
        chunk_size = None
        document_type = "general"
        
        # Try to extract context from arguments
        if args and isinstance(args[0], str):
            chunk_content = args[0]
            chunk_size = len(chunk_content.split())
            
            # Simple document type detection
            content_lower = chunk_content.lower()
            if any(word in content_lower for word in ["theorem", "proof", "lemma"]):
                document_type = "academic"
            elif any(word in content_lower for word in ["function", "class", "method"]):
                document_type = "technical"
        
        # Get model name
        model_name = model_func.__name__ if hasattr(model_func, '__name__') else str(model_func)
        
        # Make the actual LLM call
        start_time = time.time()
        success = True
        
        try:
            result = await model_func(*args, **kwargs)
            
            # Record the usage
            await record_llm_usage_with_context(
                token_estimator=token_estimator,
                call_type=call_type,
                actual_response=result,
                chunk_content=chunk_content,
                chunk_size=chunk_size,
                document_type=document_type,
                model_name=model_name,
                success=success,
                metadata={
                    "latency_ms": (time.time() - start_time) * 1000,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()) if kwargs else []
                }
            )
            
            return result
            
        except Exception as e:
            success = False
            # Record failed attempt
            await record_llm_usage_with_context(
                token_estimator=token_estimator,
                call_type=call_type,
                actual_response="",
                chunk_content=chunk_content,
                chunk_size=chunk_size,
                document_type=document_type,
                model_name=model_name,
                success=False,
                metadata={
                    "error": str(e),
                    "latency_ms": (time.time() - start_time) * 1000
                }
            )
            raise e
    
    return instrumented_caller
