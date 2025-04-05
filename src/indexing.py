import gzip
import itertools
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import ir_datasets
import nltk
from ir_datasets.datasets.cord19 import Cord19Doc, Cord19Docs

import util

for download in ["stopwords", "punkt_tab", "wordnet"]:
	nltk.download(download)

DocId = str


@dataclass
class Posting:
	doc_idx: int
	doc_id: DocId
	occurrences: int


InvertedIndex = dict[str, list[Posting]]
DocWordCount = dict[DocId, int]


@dataclass
class Index:
	inverted_index: InvertedIndex
	doc_word_count: DocWordCount


def size_of_index(index: Index) -> int:
	return (
		sys.getsizeof(index)
		+ sys.getsizeof(index.inverted_index)
		+ sum(
			sys.getsizeof(term)
			+ sys.getsizeof(postings)
			+ sum(
				sys.getsizeof(posting.doc_idx)
				+ sys.getsizeof(posting.doc_id)
				+ sys.getsizeof(posting.occurrences)
				for posting in postings
			)
			for term, postings in index.inverted_index.items()
		)
		+ sys.getsizeof(index.doc_word_count)
		+ sum(
			sys.getsizeof(doc_id) + sys.getsizeof(occurrences)
			for doc_id, occurrences in index.doc_word_count.items()
		)
	)


def indexing(d: Cord19Docs, do_stemming: bool) -> tuple[Index, float, int]:
	inverted_index = defaultdict(list)
	doc_word_count = {}

	start_time = time.time()
	total_docs = d.docs_count()
	for doc_idx, doc in enumerate(d.docs_iter()):
		doc: Cord19Doc
		print(f"\rProcessing document: {doc_idx:6}/{total_docs} ({doc.doc_id})", end="")

		if doc.doc_id in doc_word_count:
			print(f"\nIgnoring duplicate document {doc.doc_id}")
			continue

		token_counts = util.token_counts_nltk(doc, do_stemming)
		doc_word_count[doc.doc_id] = len(token_counts)
		for word, word_count in token_counts.items():
			inverted_index[word].append(Posting(doc_idx, doc.doc_id, word_count))
	print()

	index = Index(dict(inverted_index), doc_word_count)

	duration = time.time() - start_time
	space = size_of_index(index)

	return (index, duration, space)


def print_index(index: Index):
	print("Inverted index:")
	items_total = len(index.inverted_index)
	items_shown = 100
	for term, postings in itertools.islice(index.inverted_index.items(), items_shown):
		term_occurrences = sum(posting.occurrences for posting in postings)
		print(f"\tTerm: {repr(term)} → {term_occurrences} ({len(postings)})")
	print(f"\t... ({items_total - items_shown} more)")

	print("Document word count:")
	items_total = len(index.doc_word_count)
	items_shown = 10
	for doc_id, word_count in itertools.islice(
		index.doc_word_count.items(), items_shown
	):
		print(f"\tDocument: {repr(doc_id)} → {word_count}")
	print(f"\t... ({items_total - items_shown} more)")


def get_index(d: Cord19Docs) -> Index:
	path = Path("resources/index.pkl.gz")

	# If the cached inverted index exists, load it
	if path.exists():
		print("Loading cached index")
		with gzip.open(path, "rb") as f:
			return pickle.load(f)

		print("Index: (Cached)")

	# Otherwise, compute it and save it.
	else:
		print("Computing index")
		(
			index,
			index_duration,
			index_space,
		) = indexing(d, do_stemming=False)
		print(f"Index: {index_duration:.2f}s, {index_space / (1024 * 1024):.3} MiB")

		with gzip.open(path, "wb") as f:
			pickle.dump(index, f)

		return index


if __name__ == "__main__":
	DATASET: Cord19Docs = ir_datasets.load("cord19/trec-covid")
	index = get_index(DATASET)
	print_index(index)
