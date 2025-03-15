import math

import ir_datasets
from ir_datasets.datasets.cord19 import Cord19Docs
from ir_datasets.formats.trec import TrecQrel

from indexing import DocId, Index, Posting, get_index  # noqa: F401
from ranking import RetrievalModel, ranking

DATASET: Cord19Docs = ir_datasets.load("cord19/trec-covid")

if __name__ == "__main__":
	query_relevant_docs: dict[int, dict[DocId, int]] = {}
	queries = [query for query in DATASET.queries_iter()]
	for qrel in DATASET.qrels_iter():
		qrel: TrecQrel
		query_id = int(qrel.query_id) - 1
		query = queries[query_id].title
		if query not in query_relevant_docs:
			query_relevant_docs[query] = {}

		if qrel.relevance > 0:
			query_relevant_docs[query][qrel.doc_id] = qrel.relevance

	index = get_index()
	map = 0
	for query, relevant_docs in query_relevant_docs.items():
		print(f"{repr(query)}:")
		p = 1000
		docs = ranking(
			query,
			p,
			index,
			do_stemming=False,
			model=RetrievalModel.BM25,
		)

		precision = sum(relevant_docs.get(doc_id) in [1, 2] for doc_id, _ in docs) / p
		map += precision
		print(f"- P      : {precision}")

		p10 = sum(relevant_docs.get(doc_id) in [1, 2] for doc_id, _ in docs[:10]) / 10
		print(f"- P@10   : {p10}")

		dcg = sum(
			(relevant_docs.get(doc_id) or 0) / math.log2(idx + 2)
			for idx, (doc_id, _) in enumerate(docs)
		)
		idcg = sum(
			rel / math.log2(idx + 2)
			for idx, rel in enumerate(
				sorted(list(relevant_docs.values()), reverse=True)[:p]
			)
		)
		ndcg = dcg / idcg
		print(f"- nDCG   : {ndcg:.2} ({dcg:.2} / {idcg:.2})")

		dcg_10 = sum(
			(relevant_docs.get(doc_id) or 0) / math.log2(idx + 2)
			for idx, (doc_id, _) in enumerate(docs[:10])
		)
		idcg_10 = sum(
			rel / math.log2(idx + 2)
			for idx, rel in enumerate(
				sorted(list(relevant_docs.values()), reverse=True)[:10]
			)
		)
		ndcg_10 = dcg_10 / idcg_10
		print(f"- nDCG@10: {ndcg_10:.2} ({dcg_10:.2} / {idcg_10:.2})")

	map /= len(query_relevant_docs)
	print(f"MAP: {map:.2}")
