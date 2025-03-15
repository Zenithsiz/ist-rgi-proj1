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


@dataclass
class Posting:
	doc_id: int
	occurrences: int


InvertedIndex = dict[str, list[Posting]]


def indexing(d: Cord19Docs, do_stemming: bool) -> InvertedIndex:
	index = defaultdict(list)

	start_time = time.time()
	total_docs = d.docs_count()
	for doc_idx, doc in enumerate(d.docs_iter()):
		print(f"Processing document: {doc_idx:6}/{total_docs}", end="\r")

		token_counts = token_counts_nltk(doc, do_stemming)
		for word in token_counts:
			index[word].append(Posting(doc_idx, token_counts[word]))
	print()
	duration = time.time() - start_time
	space = sys.getsizeof(index)

	return (dict(index), duration, space)


def print_index(inverted_index: InvertedIndex):
	print("Inverted index:")
	items_total = len(inverted_index)
	items_shown = 100
	for term, postings in itertools.islice(inverted_index.items(), items_shown):
		term_occurrences = sum(posting.occurrences for posting in postings)
		print(f"\tTerm: {repr(term)} → {term_occurrences} ({len(postings)})")
	print(f"\t... ({items_total - items_shown} more)")


def get_inverted_index() -> InvertedIndex:
	path = Path("resources/inverted_index.pkl.gz")

	# If the cached inverted index exists, load it
	if path.exists():
		print("Loading cached reverse index")
		with gzip.open(path, "rb") as f:
			return pickle.load(f)

		print("Reverse index: (Cached)")

	# Otherwise, compute it and save it.
	else:
		print("Computing cached reverse index")
		inverted_index, inverted_index_duration, inverted_index_space = indexing(
			DATASET, do_stemming=False
		)
		print(
			f"Reverse index: {inverted_index_duration:.2f}s, {inverted_index_space / (1024 * 1024):.3} MiB"
		)

		with gzip.open(path, "wb") as f:
			pickle.dump(inverted_index, f)

		return inverted_index


if __name__ == "__main__":
	inverted_index = get_inverted_index()
	print_index(inverted_index)
