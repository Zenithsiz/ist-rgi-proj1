import heapq
import math

import ir_datasets
from ir_datasets.datasets.cord19 import Cord19Docs
from ir_datasets.formats.trec import TrecQuery

import util
from indexing import Index, get_index

DATASET: Cord19Docs = ir_datasets.load("cord19/trec-covid")


def ranking(query: str, p: int, index: Index, do_stemming: bool, k1=1.5, b=0.75):
	query_tokens = util.tokenize(query, do_stemming)

	doc_scores = {doc_idx: 0 for doc_idx in index.doc_word_count}

	for token in query_tokens:
		if token not in index.inverted_index:
			continue

		idf = math.log10(len(index.inverted_index) / len(index.inverted_index[token]))

		for posting in index.inverted_index[token]:
			tf = 1 + math.log10(posting.occurrences)

			tf_idf = tf * idf

			doc_scores[posting.doc_idx] += tf_idf

	docs = heapq.nlargest(p, doc_scores, doc_scores.get)
	for doc_idx in docs:
		print(f"Doc score {doc_idx}: {doc_scores[doc_idx]}")

	return docs


if __name__ == "__main__":
	index = get_index()
	for query in DATASET.queries_iter():
		query: TrecQuery
		docs = ranking(query.title, 10, index, do_stemming=False)

		print(f"{repr(query.title)}: {docs}")
