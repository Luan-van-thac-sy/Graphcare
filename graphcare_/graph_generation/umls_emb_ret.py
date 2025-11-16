from get_emb import embedding_retriever_batch
import pickle
from tqdm import tqdm
import time
import os

# Configuration
SAVE_INTERVAL = 10000  # Save checkpoint every 10,000 embeddings
BATCH_SIZE = 2000  # OpenAI max is 2048, use 2000 for safety
RATE_LIMIT_DELAY = 1.2  # Seconds between requests (adjust based on your tier)
MAX_RETRIES = 3  # Retry failed batches

# Paths (relative to this file's location)
EMBEDDING_SAVE_PATH = '../../exp_data/umls_ent_emb_.pkl'
CONCEPT_NAMES_PATH = '../../KG_mapping/umls/concept_names.txt'

# Create directory if it doesn't exist
os.makedirs('../../exp_data', exist_ok=True)

# Load previous embeddings if they exist
try:
    with open(EMBEDDING_SAVE_PATH, 'rb') as f:
        umls_ent_emb = pickle.load(f)
    print(f"✓ Resuming from {len(umls_ent_emb)} existing embeddings")
except FileNotFoundError:
    umls_ent_emb = []
    print("✓ Starting fresh - no previous embeddings found")

# Load concept names
print("✓ Loading concept names...")
with open(CONCEPT_NAMES_PATH, 'r') as f:
    umls_ent = f.readlines()

# Extract names from format: "CUI\tname"
umls_names = [line.split('\t')[1].strip() for line in umls_ent]
print(f"✓ Total concepts in file: {len(umls_names)}")

# Skip already processed
umls_names = umls_names[len(umls_ent_emb):]
print(f"✓ Concepts to process: {len(umls_names)}")

if len(umls_names) == 0:
    print("✅ All embeddings already generated!")
else:
    # Calculate batches
    num_batches = (len(umls_names) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"✓ Processing in {num_batches} batches of up to {BATCH_SIZE} terms")
    print(f"✓ Estimated time: ~{num_batches * RATE_LIMIT_DELAY / 60:.1f} minutes")
    print("-" * 60)

    # Process in batches
    for i in tqdm(range(0, len(umls_names), BATCH_SIZE), desc="Batch progress", unit="batch"):
        batch = umls_names[i:i+BATCH_SIZE]

        # Retry logic for failed batches
        for attempt in range(MAX_RETRIES):
            try:
                # Get embeddings for entire batch at once
                batch_embeddings = embedding_retriever_batch(batch)
                umls_ent_emb.extend(batch_embeddings)
                break  # Success, exit retry loop

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"\n⚠ Error on batch {i//BATCH_SIZE + 1} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                    print(f"  Retrying in {RATE_LIMIT_DELAY * 2} seconds...")
                    time.sleep(RATE_LIMIT_DELAY * 2)
                else:
                    print(f"\n❌ Failed batch {i//BATCH_SIZE + 1} after {MAX_RETRIES} attempts: {e}")
                    # Save progress before raising error
                    with open(EMBEDDING_SAVE_PATH, 'wb') as f:
                        pickle.dump(umls_ent_emb, f)
                    print(f"  Progress saved. Run again to resume from {len(umls_ent_emb)} embeddings")
                    raise

        # Periodic checkpoint save
        if len(umls_ent_emb) % SAVE_INTERVAL < BATCH_SIZE:
            with open(EMBEDDING_SAVE_PATH, 'wb') as f:
                pickle.dump(umls_ent_emb, f)
            print(f"\n💾 Checkpoint saved: {len(umls_ent_emb)} embeddings")

        # Rate limit delay to avoid hitting API limits
        time.sleep(RATE_LIMIT_DELAY)

    # Final save
    print("\n" + "=" * 60)
    with open(EMBEDDING_SAVE_PATH, 'wb') as f:
        pickle.dump(umls_ent_emb, f)

    print(f"✅ Complete! Total embeddings generated: {len(umls_ent_emb)}")
    print(f"✅ Saved to: {EMBEDDING_SAVE_PATH}")
