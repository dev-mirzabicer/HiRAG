import asyncio
import html
import json
import logging
import os
import re
import numbers
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union

import numpy as np
import tiktoken

logger = logging.getLogger("HiRAG")
ENCODER = None

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        # If there is already an event loop, use it.
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If in a sub-thread, create a new event loop.
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def extract_first_complete_json(s: str):
    """Extract the first complete JSON object from the string using a stack to track braces."""
    stack = []
    first_json_start = None
    
    try:
        for i, char in enumerate(s):
            if char == '{':
                stack.append(i)
                if first_json_start is None:
                    first_json_start = i
            elif char == '}':
                if stack:
                    start = stack.pop()
                    if not stack:
                        first_json_str = s[first_json_start:i+1]
                        try:
                            # Attempt to parse the JSON string
                            parsed_json = json.loads(first_json_str.replace("\n", ""))
                            logger.debug(f"Successfully extracted JSON from position {first_json_start} to {i+1}")
                            return parsed_json
                        except json.JSONDecodeError as e:
                            logger.warning(f"[JSON_PARSING] Failed to decode JSON at position {first_json_start}-{i+1}: {e}")
                            logger.debug(f"Attempted JSON string: {first_json_str[:100]}...")
                            # Continue looking for other JSON objects
                            first_json_start = None
                            continue
                        except Exception as e:
                            logger.error(f"[JSON_PARSING] Unexpected error during JSON parsing: {e}")
                            return None
                        finally:
                            first_json_start = None
        
        logger.info(f"[JSON_PARSING] No complete JSON object found in input string of length {len(s)}")
        return None
        
    except Exception as e:
        logger.error(f"[JSON_PARSING] Critical error in extract_first_complete_json: {e}")
        return None

def parse_value(value: str):
    """Convert a string value to its appropriate type (int, float, bool, None, or keep as string). Work as a more broad 'eval()'"""
    value = value.strip()

    if value == "null":
        return None
    elif value == "true":
        return True
    elif value == "false":
        return False
    else:
        # Try to convert to int or float
        try:
            if '.' in value:  # If there's a dot, it might be a float
                return float(value)
            else:
                return int(value)
        except ValueError:
            # If conversion fails, return the value as-is (likely a string)
            return value.strip('"')  # Remove surrounding quotes if they exist

def extract_values_from_json(json_string, keys=["reasoning", "answer", "data"], allow_no_quotes=False):
    """Extract key values from a non-standard or malformed JSON string, handling nested objects."""
    extracted_values = {}
    
    try:
        # Enhanced pattern to match both quoted and unquoted values, as well as nested objects
        regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'
        
        for match in re.finditer(regex_pattern, json_string, re.DOTALL):
            try:
                key = match.group('key').strip('"')  # Strip quotes from key
                value = match.group('value').strip()

                # If the value is another nested JSON (starts with '{' and ends with '}'), recursively parse it
                if value.startswith('{') and value.endswith('}'):
                    extracted_values[key] = extract_values_from_json(value)
                else:
                    # Parse the value into the appropriate type (int, float, bool, etc.)
                    extracted_values[key] = parse_value(value)
            except Exception as e:
                logger.warning(f"[JSON_PARSING] Failed to parse key-value pair '{match.group()}': {e}")
                continue

        if not extracted_values:
            logger.info(f"[JSON_PARSING] No values could be extracted from non-standard JSON string of length {len(json_string)}")
        else:
            logger.debug(f"[JSON_PARSING] Successfully extracted {len(extracted_values)} key-value pairs from non-standard JSON")
    
    except Exception as e:
        logger.error(f"[JSON_PARSING] Critical error in extract_values_from_json: {e}")
    
    return extracted_values


def convert_response_to_json(response: str) -> dict:
    """Convert response string to JSON, with error handling and fallback to non-standard JSON extraction."""
    try:
        if not response or not response.strip():
            logger.warning("[JSON_PARSING] Empty or whitespace-only response provided")
            return {}
            
        prediction_json = extract_first_complete_json(response)
        
        if prediction_json is None:
            logger.info("[JSON_PARSING] Attempting to extract values from non-standard JSON string...")
            prediction_json = extract_values_from_json(response, allow_no_quotes=True)
        
        if not prediction_json:
            logger.error(f"[JSON_PARSING] Unable to extract meaningful data from response of length {len(response)}")
            logger.debug(f"[JSON_PARSING] Response preview: {response[:200]}...")
            return {}
        else:
            logger.debug(f"[JSON_PARSING] Successfully extracted JSON with {len(prediction_json)} fields")
        
        return prediction_json
        
    except Exception as e:
        logger.error(f"[JSON_PARSING] Critical error in convert_response_to_json: {e}")
        return {}




def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    """Encode string using tiktoken with error handling."""
    global ENCODER
    try:
        if ENCODER is None:
            ENCODER = tiktoken.encoding_for_model(model_name)
        tokens = ENCODER.encode(content)
        return tokens
    except Exception as e:
        logger.error(f"[ENCODING] Failed to encode string with tiktoken: {e}")
        logger.debug(f"[ENCODING] Content length: {len(content)}, Model: {model_name}")
        # Return empty list as fallback
        return []


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    """Decode tokens using tiktoken with error handling."""
    global ENCODER
    try:
        if ENCODER is None:
            ENCODER = tiktoken.encoding_for_model(model_name)
        content = ENCODER.decode(tokens)
        return content
    except Exception as e:
        logger.error(f"[ENCODING] Failed to decode tokens with tiktoken: {e}")
        logger.debug(f"[ENCODING] Token count: {len(tokens)}, Model: {model_name}")
        # Return empty string as fallback
        return ""


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size with error handling."""
    if max_token_size <= 0:
        logger.warning(f"[TRUNCATION] Invalid max_token_size: {max_token_size}, returning empty list")
        return []
        
    try:
        tokens = 0
        for i, data in enumerate(list_data):
            try:
                data_tokens = encode_string_by_tiktoken(key(data))
                tokens += len(data_tokens)
                if tokens > max_token_size:
                    logger.debug(f"[TRUNCATION] Truncated list at index {i} (tokens: {tokens}/{max_token_size})")
                    return list_data[:i]
            except Exception as e:
                logger.warning(f"[TRUNCATION] Error processing item {i}: {e}")
                # Skip this item and continue
                continue
                
        logger.debug(f"[TRUNCATION] Returned full list with {tokens} tokens (under limit of {max_token_size})")
        return list_data
        
    except Exception as e:
        logger.error(f"[TRUNCATION] Critical error in truncate_list_by_token_size: {e}")
        # Return original list as fallback
        return list_data


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def load_json(file_name):
    """Load JSON from file with error handling."""
    try:
        if not os.path.exists(file_name):
            logger.debug(f"[FILE_IO] JSON file does not exist: {file_name}")
            return None
            
        with open(file_name, encoding="utf-8") as f:
            data = json.load(f)
            logger.debug(f"[FILE_IO] Successfully loaded JSON from {file_name}")
            return data
            
    except json.JSONDecodeError as e:
        logger.error(f"[FILE_IO] Invalid JSON in file {file_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"[FILE_IO] Error reading JSON file {file_name}: {e}")
        return None


def write_json(json_obj, file_name):
    """Write JSON to file with error handling."""
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, indent=2, ensure_ascii=False)
            logger.debug(f"[FILE_IO] Successfully wrote JSON to {file_name}")
    except Exception as e:
        logger.error(f"[FILE_IO] Error writing JSON to file {file_name}: {e}")
        raise


# it's dirty to type, so it's a good way to have fun
def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def enclose_string_with_quotes(content: Any) -> str:
    """Enclose a string with quotes"""
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'


def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )


# -----------------------------------------------------------------------------------
# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


# Utils types -----------------------------------------------------------------------
@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


# Decorators ------------------------------------------------------------------------
def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro
