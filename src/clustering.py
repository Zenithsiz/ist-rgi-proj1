import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import ir_datasets
import numpy as np
import sklearn.metrics.pairwise
import sklearn.preprocessing
from ir_datasets.datasets.cord19 import Cord19Docs
from scipy import sparse
from sklearn.cluster import DBSCAN

import indexing
import util

DATASET: Cord19Docs = ir_datasets.load("cord19/trec-covid/round1")


@dataclass
class Evaluation:
	cohesion: float
	separation: float
	silhouette: float


@dataclass
class Cluster:
	mean: list[float]
	medoid_doc_id: int

	doc_idxs: list[int]
	topics: list[str]

	evaluation: Evaluation


def clustering(d: Cord19Docs, do_stemming: bool) -> list[Cluster]:
	index = indexing.get_index(d)

	total_terms = len(index.inverted_index)
	total_docs = 20000  # len(index.doc_word_count)

	# Map from a document id to it's index on the matrix
	doc_matrix_idxs = {}

	# Vector of all document ids for each matrix entry
	doc_matrix_ids = []

	# Build the document matrix
	doc_matrix = sparse.lil_matrix((total_docs, total_terms))
	for term_idx, (_, postings) in enumerate(index.inverted_index.items()):
		print(f"\rProcessing term: {term_idx:6}/{total_terms}", end="")
		idf = math.log10(total_terms / len(postings))

		for posting in postings:
			if posting.doc_id in doc_matrix_idxs:
				doc_matrix_idx = doc_matrix_idxs[posting.doc_id]
			else:
				doc_matrix_idx = len(doc_matrix_idxs)
				if doc_matrix_idx >= total_docs:
					break

				doc_matrix_idxs[posting.doc_id] = doc_matrix_idx
				doc_matrix_ids.append(posting.doc_id)

			tf = 1 + math.log10(posting.occurrences)
			doc_matrix[doc_matrix_idx, term_idx] = tf * idf
	print()

	# Then calculate the distance matrix (using cosine)
	print(f"Calculating distance matrix ({total_docs}x{total_docs})")
	doc_dist_matrix = sklearn.metrics.pairwise.cosine_distances(doc_matrix, doc_matrix)

	# Then run dbscan
	# 0.7 , 3  good (100 clusters, 9000 outliers)
	# 0.8 , 3  good (200 clusters, 5000 outliers)
	# 0.85, 3  good (34  clusters, 2000 outliers)
	# 0.9 , 3  bad  (2   clusters, 500  outliers)
	# 0.82, 10 good (16  clusters, 6700 outliers)
	eps = 0.75
	min_samples = 5
	print(f"Running dbscan on distance matrix (eps={eps}, min_samples={min_samples})")
	clusters = defaultdict(list)
	ignored_docs = 0
	labels = DBSCAN(
		eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1
	).fit_predict(doc_dist_matrix)
	for doc_matrix_idx, label in enumerate(labels):
		if label == -1:
			ignored_docs += 1
			continue

		clusters[label].append(doc_matrix_idx)
	print(
		f"Found {len(clusters)} clusters, largest cluster is {max(map(len, clusters.values()))}..."
	)
	print(f"Ignoring {ignored_docs} outliers during clustering...")

	overall_doc_mean = (
		sum(
			(doc_matrix[doc_matrix_idx, :] for doc_matrix_idx in range(total_docs)),
			start=np.zeros(shape=(1, total_terms)),
		)
		/ total_docs
	)

	cluster_means = [
		sum(
			(
				doc_matrix[doc_matrix_idx, :]
				for doc_matrix_idx in cluster_doc_matrix_idxs
			),
			start=np.zeros(shape=(1, total_terms)),
		)
		/ len(cluster_doc_matrix_idxs)
		for cluster_doc_matrix_idxs in clusters.values()
	]

	def process_cluster(
		cluster_idx: int, cluster_doc_matrix_idxs: list[int]
	) -> Cluster:
		# Get the doc ids for all of the matrix indexes
		doc_ids = [
			doc_matrix_ids[doc_matrix_idx] for doc_matrix_idx in cluster_doc_matrix_idxs
		]

		# Then calculate the medoid and topics from the 10 largest medoids and top 10 words.
		medoids_dists = [
			sum(
				doc_dist_matrix[doc_matrix_idx, other_doc_idx]
				for other_doc_idx in cluster_doc_matrix_idxs
			)
			for doc_matrix_idx in cluster_doc_matrix_idxs
		]
		medoid_cluster_doc_idx = heapq.nsmallest(
			10,
			range(len(medoids_dists)),
			key=lambda idx: medoids_dists[idx],
		)
		medoid_doc_ids = [
			doc_ids[medoid_doc_matrix_idx]
			for medoid_doc_matrix_idx in medoid_cluster_doc_idx
		]

		topics = sum(
			(
				util.token_counts_nltk(d.docs_store().get(medoid_doc_id), do_stemming)
				for medoid_doc_id in medoid_doc_ids
			),
			Counter(),
		).most_common(10)
		topics = [topic for topic, _ in topics]

		mean = cluster_means[cluster_idx]
		cohesion = sum(
			np.linalg.norm(doc_matrix[doc_matrix_idx, :] - mean) ** 2
			for doc_matrix_idx in cluster_doc_matrix_idxs
		)

		separation = (
			len(cluster_doc_matrix_idxs) * np.linalg.norm(mean - overall_doc_mean) ** 2
		)

		def calc_silhouette(doc_matrix_idx: int) -> float:
			a = np.linalg.norm(doc_matrix[doc_matrix_idx, :] - mean)
			b = min(
				np.linalg.norm(
					doc_matrix[doc_matrix_idx, :] - cluster_means[other_cluster_idx]
				)
				for other_cluster_idx in range(len(clusters))
				if other_cluster_idx != cluster_idx
			)

			if a < b:
				return 1 - a / b
			else:
				return b / a - 1

		silhouette = sum(
			calc_silhouette(doc_matrix_idx)
			for doc_matrix_idx in cluster_doc_matrix_idxs
		) / len(cluster_doc_matrix_idxs)

		evaluation = Evaluation(cohesion, separation, silhouette)

		return Cluster(mean, medoid_doc_ids[0], doc_ids, topics, evaluation)

	return [
		process_cluster(cluster_idx, cluster_doc_matrix_idxs)
		for cluster_idx, cluster_doc_matrix_idxs in enumerate(clusters.values())
	]


if __name__ == "__main__":
	clusters = clustering(DATASET, do_stemming=False)

	for cluster_idx, cluster in enumerate(clusters):
		print(
			f"{cluster_idx}: {len(cluster.doc_idxs)} documents ({', '.join(cluster.topics)})"
		)
		print(f"\tMean: {cluster.mean}")
		print(f"\tMedoid document: {cluster.medoid_doc_id}")
		print(f"\tCohesion: {cluster.evaluation.cohesion}")
		print(f"\tSeparation: {cluster.evaluation.separation}")
		print(f"\tSilhouette: {cluster.evaluation.silhouette}")

	mean = sum((cluster.mean for cluster in clusters[1:]), clusters[0].mean) / len(
		clusters
	)
	print(f"Mean: {mean}")

	total_cohesion = sum(cluster.evaluation.cohesion for cluster in clusters)
	print(f"Total cohesion: {total_cohesion}")

	total_separation = sum(cluster.evaluation.separation for cluster in clusters)
	print(f"Total separation: {total_separation}")

	total_error = total_cohesion + total_separation
	print(f"Total error: {total_error}")

	total_silhouette = sum(cluster.evaluation.silhouette for cluster in clusters) / len(
		clusters
	)
	print(f"Total silhouette: {total_silhouette}")
