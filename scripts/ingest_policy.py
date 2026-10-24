"""Build or refresh the local policy index.

usage: python -m scripts.ingest_policy [--rebuild]

Safe to run again: nothing is embedded when the policy and the embedding model have not changed.
"""
import argparse
import sys

from app.rag.chunker import DEFAULT_POLICY_DIR, PolicyParseError, load_policy_chunks, policy_hash
from app.rag.embeddings import EmbeddingError, OpenAIEmbedder
from app.rag.index import DEFAULT_INDEX_DIR, PolicyIndexError, ingest, open_client


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or refresh the policy index.")
    parser.add_argument("--rebuild", action="store_true", help="delete the index and build it again")
    args = parser.parse_args()

    try:
        chunks = load_policy_chunks(DEFAULT_POLICY_DIR)
        version = policy_hash(DEFAULT_POLICY_DIR)
        report = ingest(open_client(DEFAULT_INDEX_DIR), chunks, OpenAIEmbedder(), version, rebuild=args.rebuild)
    except (PolicyParseError, PolicyIndexError, EmbeddingError) as e:
        print(f"Ingest failed: {e}", file=sys.stderr)
        return 1

    if report.skipped:
        print(f"Index already up to date ({report.total} clauses). Nothing was embedded.")
    else:
        print(f"Indexed {report.upserted} clauses, removed {report.removed}. Total in index: {report.total}.")
    print(f"Embedding model: {report.embedding_model}")
    print(f"Policy hash:     {report.policy_hash[:12]}")
    print(f"Index folder:    {DEFAULT_INDEX_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
