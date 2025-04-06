import heapq
import math

import ir_datasets
import numpy as np
import sklearn.metrics.pairwise
from ir_datasets.datasets.cord19 import Cord19Docs
from ir_datasets.formats.trec import TrecQuery
from scipy import sparse
from sklearn.preprocessing import normalize

from indexing import DocId, Index, Posting, get_index  # noqa: F401
from ranking import RetrievalModel, ranking

DATASET: Cord19Docs = ir_datasets.load("cord19/trec-covid/round1")

if __name__ == "__main__":
	index = get_index(DATASET)

	minimum_similarity_threshold = 1e-5
	for query in DATASET.queries_iter():
		# Perform initial IR
		query: TrecQuery
		docs = ranking(
			query.title, 1000, index, do_stemming=False, model=RetrievalModel.BM25
		)

		# Build the document matrix
		doc_ids = [doc_id for doc_id, _ in docs]
		doc_idxs = {doc_id: doc_idx for doc_idx, (doc_id, _) in enumerate(docs)}
		total_terms = len(index.inverted_index)
		doc_matrix = sparse.lil_matrix((len(docs), total_terms))
		for term_idx, (_, postings) in enumerate(index.inverted_index.items()):
			print(f"\rProcessing term: {term_idx:6}/{total_terms}", end="")
			idf = math.log10(total_terms / len(postings))

			for posting in postings:
				if posting.doc_id not in doc_idxs:
					continue

				doc_matrix_idx = doc_idxs[posting.doc_id]

				tf = 1 + math.log10(posting.occurrences)
				doc_matrix[doc_matrix_idx, term_idx] = tf * idf
		print()

		# Then the distance matrix
		doc_dist_matrix = sklearn.metrics.pairwise.cosine_distances(
			doc_matrix, doc_matrix
		)

		# Build the graph, an (1000x1000) matrix with each row representing the probability
		# to jump to another document, given by normalizing 1 minus the cosine distances
		doc_prob_matrix = 1.0 - doc_dist_matrix
		np.fill_diagonal(doc_prob_matrix, 0.0)
		doc_prob_matrix = normalize(doc_prob_matrix)

		# Add a random jump chance to the probability matrix
		random_jump_prob = 0.15
		random_prob_matrix = np.full(
			shape=doc_prob_matrix.shape, fill_value=1 / len(docs)
		)
		doc_prob_matrix = (
			1 - random_jump_prob
		) * doc_prob_matrix + random_jump_prob * random_prob_matrix

		# Now perform the power tower calculation
		num_iterations = 0
		doc_probs = np.full(shape=doc_prob_matrix.shape[1], fill_value=1 / len(docs))
		while num_iterations < 50:
			prev_doc_probs = doc_probs
			doc_probs = np.dot(doc_prob_matrix, doc_probs)
			doc_probs /= np.linalg.norm(doc_probs)

			similarity = np.linalg.norm(prev_doc_probs - doc_probs)
			if similarity <= minimum_similarity_threshold:
				break
			num_iterations += 1
		print(
			f"Took {num_iterations} iterations to reach similarity threshold ({similarity:.4})"
		)

		# Finally print out the best 10 results from page rank
		top10_doc_idxs = heapq.nlargest(
			10, range(len(doc_probs)), key=lambda idx: doc_probs[idx]
		)
		print(f"{repr(query.title)}:")
		for doc_idx in top10_doc_idxs:
			doc_id = doc_ids[doc_idx]
			doc_prob = doc_probs[doc_idx]
			print(f"- {doc_ids[doc_idx]}: {doc_prob:.06f}")
