import requests
import json

# with open("../../resources/openai.key", 'r') as f:
key ="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"

def embedding_retriever_batch(terms):
    """
    Batch embedding retrieval - up to 2048 terms at once
    Args:
        terms: List of strings or single string
    Returns:
        List of embeddings in the same order as input
    """
    # Handle single term input
    if isinstance(terms, str):
        terms = [terms]

    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }

    # Using text-embedding-3-small: faster and 80% cheaper than ada-002
    payload = {
        "input": terms,
        "model": "text-embedding-3-small"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

    data = response.json()["data"]

    # Sort by index to maintain order (API may return out of order)
    embeddings = [item['embedding'] for item in sorted(data, key=lambda x: x['index'])]

    return embeddings

def embedding_retriever(term):
    """
    Single term embedding retrieval - for backward compatibility
    Args:
        term: String to embed
    Returns:
        Embedding as list of floats
    """
    return embedding_retriever_batch([term])[0]