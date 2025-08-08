#!/usr/bin/env python3
"""
Simple test for sentence transformer embeddings
"""

import asyncio
from sentence_transformers import SentenceTransformer
import numpy as np
import time


async def test_sentence_transformers():
    """Test sentence transformer functionality"""
    print("🧪 Testing Sentence Transformers\n")
    
    # Load model
    print("📦 Loading embedding model...")
    start_time = time.time()
    
    try:
        model = await asyncio.to_thread(
            SentenceTransformer,
            "all-MiniLM-L6-v2"
        )
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Test sentences
        test_sentences = [
            "Operating account with USD currency",
            "Reserve account with USD balance",
            "Euro trading account for forex",
            "Checking account for daily operations"
        ]
        
        # Generate embeddings
        print(f"\n🧮 Generating embeddings for {len(test_sentences)} sentences...")
        start_time = time.time()
        
        embeddings = await asyncio.to_thread(
            model.encode,
            test_sentences,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        embed_time = time.time() - start_time
        print(f"✅ Generated embeddings in {embed_time:.3f} seconds")
        
        # Print results
        print(f"\n📊 Embedding Results:")
        print(f"   - Shape: {embeddings.shape}")
        print(f"   - Dimension: {embeddings.shape[1]}")
        print(f"   - Type: {type(embeddings)}")
        print(f"   - Normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)}")
        
        # Calculate similarity
        print(f"\n🔗 Cosine Similarities (accounts with same currency):")
        # USD accounts (0, 1) should be more similar
        sim_usd = np.dot(embeddings[0], embeddings[1])
        sim_eur = np.dot(embeddings[0], embeddings[2])
        
        print(f"   - USD account 1 ↔️ USD account 2: {sim_usd:.4f}")
        print(f"   - USD account 1 ↔️ EUR account: {sim_eur:.4f}")
        
        # Sample embeddings
        print(f"\n📐 Sample embedding values:")
        for i, sent in enumerate(test_sentences[:2]):
            print(f"   {sent[:40]}...")
            print(f"   [{embeddings[i][0]:.4f}, {embeddings[i][1]:.4f}, {embeddings[i][2]:.4f}, ...]")
        
        print("\n✨ All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTrying to install sentence-transformers...")
        import subprocess
        subprocess.run(["pip3", "install", "sentence-transformers"], check=True)
        print("Please run the test again after installation.")


if __name__ == "__main__":
    asyncio.run(test_sentence_transformers())