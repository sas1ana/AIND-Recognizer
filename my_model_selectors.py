import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            components = range(self.min_n_components, self.max_n_components + 1)
            best_score, model = max([self.scr_model(i) for i in components])
            return model
        except:
            return self.base_model(self.n_constant)

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant
    """

    def select(self):
        """ select based on n_constant value
        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def scr_model(self, num_components):
        """
            Calculate the BIC score
        """
        hmm_model = self.base_model(num_components)
        logL = hmm_model.score(self.X, self.lengths)

        BIC = -2 *logL + num_components *(np.log(len(self.X)))
        return BIC, hmm_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def scr_model(self, num_components):
        hmm_model = self.base_model(num_components)
        logL = hmm_model.score(self.X, self.lengths)

        scr = [hmm_model.score(X, lengths) for i, (X, lengths) in self.hwords.items() if i != self.this_word]

        DIC = logL-sum(scr)/(len(scr)-1)
        return DIC, hmm_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''
    def scr_model(self, num_components):
        scr = []
        for cv_train_idx, cv_test_idx in KFold().split(self.sequences):
            self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)    
            X, lens = combine_sequences(cv_test_idx, self.sequences)
            
            hmm_model = self.base_model(num_components)
            scr.append(hmm_model.score(X, lens))
        return np.mean(scr), hmm_model
