import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set
   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """

    probabilities = []
    guesses = []

    # Implement the recognizer

    for test_word in test_set.get_all_sequences().keys():
        X, lengths = test_set.get_item_Xlengths(test_word)
        prob_dict = {}
        best_score = float("-inf")
        best_guess = ""

        for train_word, model in models.items():
            try:
                log_score = model.score(X, lengths)
            except:
                log_score = float("-inf")
            prob_dict[train_word] = log_score
            if log_score > best_score:
                best_score = log_score
                best_guess = train_word

        guesses.append(best_guess)
        probabilities.append(prob_dict)

    return probabilities, guesses
