from __future__ import annotations

from pathlib import Path

from notes_lifelog_rag.embeddings.engines import get_embedding_backend
from notes_lifelog_rag.embeddings.repository import best_embedding_model_in_db, vector_search
from notes_lifelog_rag.rerank.engines import RerankCandidate, get_reranker
from notes_lifelog_rag.runtime.device import DeviceInfo
from notes_lifelog_rag.search.keyword import SearchResult, search_notes


def hybrid_search_notes(
    query: str,
    *,
    limit: int = 10,
    db_path: str | Path | None = None,
    embedding_backend: str = "auto",
    reranker_backend: str = "auto",
    rerank_top_k: int = 20,
    device_info: DeviceInfo | None = None,
    dtype: str = "auto",
    batch_size: int = 16,
) -> list[SearchResult]:
    keyword_results = search_notes(query, limit=max(limit * 3, 10), db_path=db_path)
    combined: dict[str, SearchResult] = {}
    for result in keyword_results:
        combined[result.note_id] = SearchResult(
            note_id=result.note_id,
            title=result.title,
            source_relative_path=result.source_relative_path,
            folder=result.folder,
            snippet=result.snippet,
            score=result.score * 0.60,
            source=result.source,
        )

    stored_model = best_embedding_model_in_db(db_path, include_mock=embedding_backend == "mock")
    if stored_model:
        backend = get_embedding_backend(
            "mock" if stored_model.startswith("mock-") else embedding_backend,
            model_name=None if stored_model.startswith("mock-") else stored_model,
            device_info=device_info,
            dtype=dtype,
            batch_size=batch_size,
        )
        for result in vector_search(
            query,
            backend,
            db_path=db_path,
            model_name=stored_model,
            limit=max(limit * 3, 10),
        ):
            previous = combined.get(result.note_id)
            vector_score = max(0.0, result.score) * 0.40
            if previous:
                combined[result.note_id] = SearchResult(
                    note_id=previous.note_id,
                    title=previous.title,
                    source_relative_path=previous.source_relative_path,
                    folder=previous.folder,
                    snippet=previous.snippet or result.snippet,
                    score=previous.score + vector_score,
                    source=f"{previous.source}+embedding",
                )
            else:
                combined[result.note_id] = SearchResult(
                    note_id=result.note_id,
                    title=result.title,
                    source_relative_path=result.source_relative_path,
                    folder=result.folder,
                    snippet=result.snippet,
                    score=vector_score,
                    source="embedding",
                )

    ranked = sorted(combined.values(), key=lambda item: item.score, reverse=True)
    reranker = get_reranker(
        reranker_backend,
        device_info=device_info,
        dtype=dtype,
        batch_size=batch_size,
    )
    if reranker.is_available() and ranked and reranker_backend not in {"none", "disabled"}:
        pool = ranked[: max(rerank_top_k, limit)]
        reranked = reranker.rerank(
            query,
            [
                RerankCandidate(
                    id=result.note_id,
                    text=f"{result.title}\n{result.snippet}",
                    original_score=result.score,
                )
                for result in pool
            ],
        )
        lookup = {result.note_id: result for result in pool}
        reordered: list[SearchResult] = []
        for rerank_result in reranked:
            original = lookup.get(rerank_result.id)
            if original is None:
                continue
            reordered.append(
                SearchResult(
                    note_id=original.note_id,
                    title=original.title,
                    source_relative_path=original.source_relative_path,
                    folder=original.folder,
                    snippet=original.snippet,
                    score=rerank_result.score,
                    source=f"{original.source}+rerank",
                )
            )
        remainder = [result for result in ranked if result.note_id not in {item.note_id for item in reordered}]
        ranked = reordered + remainder
    return ranked[:limit]
