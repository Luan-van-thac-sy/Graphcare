import pickle
import numpy as np
from tqdm import tqdm

def batch_cosine_similarity(target_embs, umls_ent_emb, threshold=0.7):
    """
    Vectorized cosine similarity - 100x faster than brute force!

    Args:
        target_embs: Dictionary of {code_id: embedding_vector}
        umls_ent_emb: Numpy array of UMLS embeddings (shape: [n_umls, embedding_dim])
        threshold: Minimum similarity threshold to accept a match

    Returns:
        Dictionary of {code_id: umls_index or None}
    """
    # Normalize UMLS embeddings once (avoid redundant calculations)
    umls_normalized = umls_ent_emb / np.linalg.norm(umls_ent_emb, axis=1, keepdims=True)

    results = {}
    for key, target_emb in tqdm(target_embs.items(), desc="Finding similarities"):
        # Normalize target embedding
        target_normalized = target_emb / np.linalg.norm(target_emb)

        # Vectorized dot product - computes ALL similarities at once!
        similarities = np.dot(umls_normalized, target_normalized)

        # Find best match
        max_idx = np.argmax(similarities)
        if similarities[max_idx] > threshold:
            results[key] = int(max_idx)
        else:
            results[key] = None

    return results

if __name__ == "__main__":
    print("=" * 60)
    print("UMLS Similarity Retriever (Optimized)")
    print("=" * 60)

    print("\nLoading UMLS concept names...")
    with open("../../KG_mapping/umls/concept_names.txt", 'r') as f:
        umls_ent = f.readlines()

    umls_ids = []
    umls_names = []
    for line in umls_ent:
        umls_id = line.split('\t')[0]
        umls_name = line.split('\t')[1][:-1]
        umls_names.append(umls_name)
        umls_ids.append(umls_id)
    print(f"✓ Loaded {len(umls_ids)} UMLS concepts")

    print("\nLoading embeddings...")
    with open('../../exp_data/umls_ent_emb_.pkl', 'rb') as f:
        umls_ent_emb = pickle.load(f)
        # Convert to numpy array if it isn't already
        if not isinstance(umls_ent_emb, np.ndarray):
            umls_ent_emb = np.array(umls_ent_emb)
    print(f"✓ UMLS embeddings shape: {umls_ent_emb.shape}")

    with open('../../exp_data/atc3_id2emb.pkl', 'rb') as f:
        atc3_id2emb = pickle.load(f)
        # Convert embeddings to numpy arrays
        atc3_id2emb = {k: np.array(v) for k, v in atc3_id2emb.items()}
    print(f"✓ ATC3 embeddings: {len(atc3_id2emb)} items")

    with open('../../exp_data/ccscm_id2emb.pkl', 'rb') as f:
        ccscm_id2emb = pickle.load(f)
        ccscm_id2emb = {k: np.array(v) for k, v in ccscm_id2emb.items()}
    print(f"✓ CCSCM embeddings: {len(ccscm_id2emb)} items")

    with open('../../exp_data/ccsproc_id2emb.pkl', 'rb') as f:
        ccsproc_id2emb = pickle.load(f)
        ccsproc_id2emb = {k: np.array(v) for k, v in ccsproc_id2emb.items()}
    print(f"✓ CCSPROC embeddings: {len(ccsproc_id2emb)} items")

    SIMILARITY_THRESHOLD = 0.7
    print(f"\nSimilarity threshold: {SIMILARITY_THRESHOLD}")
    print("=" * 60)

    print("\nMapping CCSCM to UMLS...")
    ccscm2umls = batch_cosine_similarity(ccscm_id2emb, umls_ent_emb, SIMILARITY_THRESHOLD)
    matched_ccscm = sum(1 for v in ccscm2umls.values() if v is not None)
    print(f"  Matched: {matched_ccscm}/{len(ccscm2umls)} codes")

    print("\nMapping CCSPROC to UMLS...")
    ccsproc2umls = batch_cosine_similarity(ccsproc_id2emb, umls_ent_emb, SIMILARITY_THRESHOLD)
    matched_ccsproc = sum(1 for v in ccsproc2umls.values() if v is not None)
    print(f"  Matched: {matched_ccsproc}/{len(ccsproc2umls)} codes")

    print("\nMapping ATC3 to UMLS...")
    atc32umls = batch_cosine_similarity(atc3_id2emb, umls_ent_emb, SIMILARITY_THRESHOLD)
    matched_atc3 = sum(1 for v in atc32umls.values() if v is not None)
    print(f"  Matched: {matched_atc3}/{len(atc32umls)} codes")

    print("\n" + "=" * 60)
    print("Saving results...")

    with open('../../exp_data/ccscm2umls.pkl', 'wb') as f:
        pickle.dump(ccscm2umls, f)
    print("✓ Saved ccscm2umls.pkl")

    with open('../../exp_data/ccsproc2umls.pkl', 'wb') as f:
        pickle.dump(ccsproc2umls, f)
    print("✓ Saved ccsproc2umls.pkl")

    with open('../../exp_data/atc32umls.pkl', 'wb') as f:
        pickle.dump(atc32umls, f)
    print("✓ Saved atc32umls.pkl")

    print("\n" + "=" * 60)
    print("✓ All mappings completed successfully!")
    print(f"Total matched: {matched_ccscm + matched_ccsproc + matched_atc3}/{len(ccscm2umls) + len(ccsproc2umls) + len(atc32umls)}")
    print("=" * 60)
