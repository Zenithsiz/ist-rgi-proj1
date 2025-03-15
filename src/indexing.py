import gzip
import itertools
import pickle
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import ir_datasets
import nltk
from ir_datasets.datasets.cord19 import Cord19Doc, Cord19Docs

import util

DATASET: Cord19Docs = ir_datasets.load("cord19/trec-covid")
for download in ["stopwords", "punkt_tab", "wordnet"]:
	nltk.download(download)


def token_counts_nltk(doc: Cord19Doc, do_stemming: bool) -> Counter[str]:
	counter = Counter()

	tokens_title = nltk.sent_tokenize(doc.title)
	tokens_abstract = nltk.sent_tokenize(doc.abstract)
	for sentence in [*tokens_title, *tokens_abstract]:
		tokens = util.tokenize(sentence, do_stemming)
		counter.update(tokens)

	return counter


DocId = int


@dataclass
class Posting:
	doc_idx: DocId
	occurrences: int


InvertedIndex = dict[str, list[Posting]]
DocWordCount = dict[DocId, int]


@dataclass
class Index:
	inverted_index: InvertedIndex
	doc_word_count: DocWordCount


def indexing(d: Cord19Docs, do_stemming: bool) -> tuple[Index, float, int]:
	inverted_index = defaultdict(list)
	doc_word_count = {}

	start_time = time.time()
	total_docs = d.docs_count()
	for doc_idx, doc in enumerate(d.docs_iter()):
		print(f"Processing document: {doc_idx:6}/{total_docs}", end="\r")

		token_counts = token_counts_nltk(doc, do_stemming)
		doc_word_count[doc_idx] = len(token_counts)
		for word in token_counts:
			inverted_index[word].append(Posting(doc_idx, token_counts[word]))
	print()
	duration = time.time() - start_time
	space = sys.getsizeof(inverted_index)

	return (Index(dict(inverted_index), doc_word_count), duration, space)


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
	for doc_idx, word_count in itertools.islice(
		index.doc_word_count.items(), items_shown
	):
		print(f"\tDocument: {doc_idx} → {word_count}")
	print(f"\t... ({items_total - items_shown} more)")


def get_index() -> Index:
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
		) = indexing(DATASET, do_stemming=False)
		print(f"Index: {index_duration:.2f}s, {index_space / (1024 * 1024):.3} MiB")

		with gzip.open(path, "wb") as f:
			pickle.dump(index, f)

		return index


if __name__ == "__main__":
	index = get_index()
	print_index(index)
