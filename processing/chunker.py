class Chunker:
    def __init__(self, max_tokens=1000, overlap_tokens=100):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text, metadata=None):
        # Token-based chunking with overlap
        words = text.split()
        total_tokens = len(words)
        chunks = []
        start = 0
        chunk_index = 0
        while start < total_tokens:
            end = min(start + self.max_tokens, total_tokens)
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            chunk_meta = dict(metadata or {})
            chunk_meta["chunk_index"] = chunk_index
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_meta
            })
            if end == total_tokens:
                break
            start = end - self.overlap_tokens if self.overlap_tokens > 0 else end
            chunk_index += 1
        return chunks