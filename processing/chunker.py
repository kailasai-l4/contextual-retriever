class Chunker:
    def __init__(self, max_tokens=1000, overlap_tokens=100):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text, metadata=None):
        # Simple chunking: split by paragraphs, fallback to lines if needed
        chunks = []
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        for i, para in enumerate(paragraphs):
            chunk_meta = dict(metadata or {})
            chunk_meta["chunk_index"] = i
            chunks.append({
                "text": para,
                "metadata": chunk_meta
            })
        if not chunks:
            # fallback: split by lines
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if line.strip():
                    chunk_meta = dict(metadata or {})
                    chunk_meta["chunk_index"] = i
                    chunks.append({
                        "text": line.strip(),
                        "metadata": chunk_meta
                    })
        return chunks