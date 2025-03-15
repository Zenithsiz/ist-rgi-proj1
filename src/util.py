import re
import string

import nltk

STOPWORDS = nltk.corpus.stopwords.words("english")
STOPWORD_REGEX = re.compile(r"([+\-]?\d+\.\d+)|([+\-]?\d+\/[+\-]?\d+)|([+\-]?\d+)")


def _is_stopword(token: str) -> bool:
	if token in STOPWORDS:
		return True

	if token in string.punctuation or token in ["‘", "’"]:
		return True

	if STOPWORD_REGEX.match(token) is not None:
		return True

	return False


LEMMATIZER = nltk.stem.WordNetLemmatizer()
STEMMER = nltk.stem.PorterStemmer()


def tokenize(sentence: str, do_stemming: bool) -> list[str]:
	tokens = nltk.word_tokenize(sentence)
	tokens = [token.lower() for token in tokens]
	tokens = [token for token in tokens if not _is_stopword(token)]
	if do_stemming:
		tokens: list[str] = [STEMMER.stem(tok) for tok in tokens]
	else:
		tokens = [LEMMATIZER.lemmatize(tok) for tok in tokens]

	return tokens
