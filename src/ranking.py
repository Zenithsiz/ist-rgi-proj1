import heapq
import math
from enum import Enum

import ir_datasets
from ir_datasets.datasets.cord19 import Cord19Docs
from ir_datasets.formats.trec import TrecQuery

import util
from indexing import DocId, Index, Posting, get_index

DATASET: Cord19Docs = ir_datasets.load("cord19/trec-covid")


class RetrievalModel(Enum):
	TF_IDF = 1
	BM25 = 2


def ranking(
	query: str,
	p: int,
	index: Index,
	do_stemming: bool,
	model=RetrievalModel.TF_IDF,
	k1=1.5,
	b=0.75,
) -> list[tuple[DocId, float]]:
	query_tokens = util.tokenize(query, do_stemming)

	doc_scores = {doc_id: 0 for doc_id in index.doc_word_count}

	doc_word_avg = sum(index.doc_word_count.values()) / len(index.doc_word_count)

	for token in query_tokens:
		if token not in index.inverted_index:
			continue

		idf = math.log10(len(index.inverted_index) / len(index.inverted_index[token]))

		for posting in index.inverted_index[token]:
			posting: Posting
			match model:
				case RetrievalModel.TF_IDF:
					tf = 1 + math.log10(posting.occurrences)
					score = tf * idf
				case RetrievalModel.BM25:
					score = (
						idf
						* (posting.occurrences * (k1 + 1))
						/ (
							posting.occurrences
							+ k1
							* (
								1
								- b
								+ b
								* index.doc_word_count[posting.doc_id]
								/ doc_word_avg
							)
						)
					)

			doc_scores[posting.doc_id] += score

	docs = [
		(doc_id, doc_scores[doc_id])
		for doc_id in heapq.nlargest(p, doc_scores, doc_scores.get)
	]
	return docs


if __name__ == "__main__":
	index = get_index(DATASET)
	for query in DATASET.queries_iter():
		query: TrecQuery
		docs = ranking(
			query.title, 10, index, do_stemming=False, model=RetrievalModel.BM25
		)

		print(f"{repr(query.title)}:")
		for doc_id, score in docs:
			print(f"- {doc_id}: {score}")
