"""
Optimized data processing utilities for A2A platform
Handles large datasets efficiently with streaming and chunking
"""

import json
import csv
from typing import Iterator, Dict, Any, List
from pathlib import Path

class OptimizedDataProcessor:
    """Efficient data processing for large files"""
    
    @staticmethod
    def stream_json_lines(file_path: Path, chunk_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
        """Stream JSON lines in chunks for memory efficiency"""
        with open(file_path, 'r') as f:
            chunk = []
            for line in f:
                try:
                    chunk.append(json.loads(line.strip()))
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                except json.JSONDecodeError:
                    continue
            if chunk:
                yield chunk
    
    @staticmethod
    def stream_csv_chunks(file_path: Path, chunk_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
        """Stream CSV data in chunks for memory efficiency"""
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            chunk = []
            for row in reader:
                chunk.append(row)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
    
    @staticmethod
    def process_large_file_async(file_path: Path, processor_func, chunk_size: int = 1000):
        """Process large files asynchronously"""
        import asyncio
        
        async def process_chunk(chunk):
            return await asyncio.to_thread(processor_func, chunk)
        
        if file_path.suffix.lower() == '.csv':
            chunks = OptimizedDataProcessor.stream_csv_chunks(file_path, chunk_size)
        else:
            chunks = OptimizedDataProcessor.stream_json_lines(file_path, chunk_size)
        
        results = []
        for chunk in chunks:
            result = asyncio.run(process_chunk(chunk))
            results.extend(result)
        
        return results

# Cache for frequently accessed data
_data_cache = {}

def cache_data(key: str, data: Any, max_size: int = 100):
    """Simple LRU-style cache for data"""
    global _data_cache
    if len(_data_cache) >= max_size:
        # Remove oldest entry
        oldest_key = next(iter(_data_cache))
        del _data_cache[oldest_key]
    _data_cache[key] = data

def get_cached_data(key: str) -> Any:
    """Retrieve cached data"""
    return _data_cache.get(key)
