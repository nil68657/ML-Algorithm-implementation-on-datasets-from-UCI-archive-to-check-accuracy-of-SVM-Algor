import pandas as pd
import re
import csv
import string
from collections import Counter
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords


class Utils:

    # Extracts optical recognition data from input filename
    @staticmethod
    def extract_optical_xls_file(file_name):
        try:
            file = open(file_name, 'r')
            data_set = pd.read_csv(file)
            features = list(data_set.columns[:-1])
            data = data_set[features]
            target = data_set.columns[-1]
            result = data_set[target]
            file.close()
            return {
                'data': data,
                'result': result
            }
        except IOError as error:
            print('IOError: ' + error.args[1])

    # Prepares grammar and linguistic components
    @staticmethod
    def __initialise_linguistic_components():
        stop_words = []
        emoticons_str = r"""
            (?:
                [:=;] # Eyes
                [oO\-]? # Nose (optional)
                [D\)\]\(\]/\\OpP] # Mouth
            )"""

        regex_str = [
            r'<[^>]+>',                     # HTML tags
            r"(?:[a-z][a-z\-_]+[a-z])",     # words with - and '
            r'(?:[\w_]+)',                  # other words
            r'(?:\S)'                       # anything else
        ]

        tokens = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
        emoticons = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
        punctuations = list(string.punctuation)
        stem_words = PorterStemmer()
        lem_words = WordNetLemmatizer()
        stop_word_rule = stopwords.words('english') + punctuations + ['rt', 'via', 'i\'m', 'us', 'it']
        for stop_word in stop_word_rule:
            stop_words.append(stem_words.stem(lem_words.lemmatize(stop_word, pos="v")))
        return {
            'tokens': tokens,
            'emoticons': emoticons,
            'stem_words': stem_words,
            'lem_words': lem_words,
            'stop_words': stop_words
        }

    # Converts the input string into a list of words
    @staticmethod
    def __pre_process(s, grammar):
        tokens = grammar['tokens'].findall(s)
        tokens = [token
                  if grammar['emoticons'].search(token)
                  else grammar['stem_words'].stem(grammar['lem_words'].lemmatize(token.lower(), pos="v"))
                  for token in tokens]
        return tokens

    # Calculates the decision attributes and returns the most frequent ones
    @staticmethod
    def __get_decision_attributes(digitised_review):
        counter = []
        decision_attributes = []
        for a in digitised_review:
            counter.append(" ".join(a))
        buffer = ""
        for a in counter:
            buffer += "".join(a)
        x = Counter(buffer.split(" "))
        for (k, v) in x.most_common(min(500, len(x))):
            decision_attributes.append(k)
        return decision_attributes

    # Generates a sparse matrix using the reviews
    @staticmethod
    def __sparse_matrix(digitised_review, decision_attributes):
        sparse_matrix = []
        for review in digitised_review:
            new_review = [0] * len(decision_attributes)
            for word in review:
                if word in decision_attributes:
                    index = decision_attributes.index(word)
                    new_review[index] += 1
                else:
                    pass
            sparse_matrix.append(new_review)
        return sparse_matrix

    # Converts textual reviews to their digital representation
    @staticmethod
    def __digitise_reviews(reviews):
        digitised_reviews = []
        grammar = Utils.__initialise_linguistic_components()
        for review in reviews:
            review_lover_case = str(review).lower()
            terms_stop = [term for term in Utils.__pre_process(review_lover_case, grammar)
                          if term not in grammar['stop_words'] and
                          not ~term.isdigit() and len(str(term)) > 1]
            digitised_reviews.append(terms_stop)
        return digitised_reviews

    # Extracts Amazon review data from input filename
    @staticmethod
    def extract_amazon_xls_file(file_name):
        try:
            data = []
            file = open(file_name, 'r')
            data_set = csv.reader(file, dialect='excel')
            for product in data_set:
                row = (product[1], product[-1])
                data.append(row)
            data_set = pd.DataFrame(data, columns=['review', 'rating'])
            digitised_review = Utils.__digitise_reviews(data_set['review'])
            decision_attributes = Utils().__get_decision_attributes(digitised_review)
            sparse_matrix = Utils().__sparse_matrix(digitised_review, decision_attributes)
            features = pd.DataFrame(sparse_matrix, columns=decision_attributes)
            file.close()
            return {
                'result': data_set['rating'],
                'data': features
            }
        except IOError as error:
            print('IOError: ' + error.args[1])
