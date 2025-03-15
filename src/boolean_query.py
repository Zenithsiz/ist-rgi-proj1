from dataclasses import dataclass

import ir_datasets
from ir_datasets.datasets.cord19 import Cord19Docs
from ir_datasets.formats.trec import TrecQuery

import util
from indexing import InvertedIndex, Posting, get_index

DATASET: Cord19Docs = ir_datasets.load("cord19/trec-covid")


def boolean_query(query: str, k: int, inverted_index: InvertedIndex, do_stemming: bool):
	query_tokens = util.tokenize(query, do_stemming)

	@dataclass
	class QueryPostings:
		postings: list[Posting]
		cur_idx: int

		def cur_posting(self) -> Posting:
			return self.postings[self.cur_idx]

		def cur_doc_idx(self) -> int:
			return self.cur_posting().doc_idx

	try:
		query_posting_idxs = [
			QueryPostings(inverted_index[token], 0) for token in query_tokens
		]
	except KeyError:
		return []

	# While we still have postings for each query
	docs = []
	while len(docs) < k and all(
		postings.cur_idx < len(postings.postings) for postings in query_posting_idxs
	):
		# If the first posting is the same for all, we found a common document, so add it
		if all(
			postings.postings[postings.cur_idx].doc_idx
			== query_posting_idxs[0].cur_doc_idx()
			for postings in query_posting_idxs[1:]
		):
			docs.append(query_posting_idxs[0].cur_doc_idx())
			for posting in query_posting_idxs:
				posting.cur_idx += 1

		# Otherwise, remove the one with the smallest doc index
		else:
			smallest_idx = None
			smallest_doc_idx = None
			for posting_idx, posting in enumerate(query_posting_idxs):
				posting_doc_idx = posting.cur_doc_idx()
				if smallest_doc_idx is None or posting_doc_idx < smallest_doc_idx:
					smallest_idx = posting_idx
					smallest_doc_idx = posting_doc_idx
			assert smallest_idx is not None

			query_posting_idxs[smallest_idx].cur_idx += 1

	return docs


if __name__ == "__main__":
	index = get_index()
	for query in DATASET.queries_iter():
		query: TrecQuery
		docs = boolean_query(query.title, 10, index.inverted_index, do_stemming=False)

		print(f"{repr(query.title)}: {docs}")
