import numpy as np
import sklearn.base as ba
import sklearn.multiclass as mc
import sklearn.neighbors as ne
import statsmodels.api as sm

class ParzenWindowB(ba.BaseEstimator, ba.ClassifierMixin):    
    """
    Parzen Window Bayes (ParzenWindowB)
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel        
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        # TODO trocar KernelDensity por sm.nonparametric.KDEMultivariate
        self.models_ = [ne.KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        # TODO remover probabilidade a priori
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self
            
    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        # TODO remover probabilidade a priori
        result = np.exp(logprobs + self.logpriors_)
        # somatório é o fator de normalização que garante probabilidades entre 0 e 1
        norm = result.sum(1, keepdims=True)
        norm[norm == 0.0] = 1e-10        
        return result / norm
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

class WindowSizeOptimizer():
    def bw(self, X):
        n_samples, n_vars = X.shape
        est = sm.nonparametric.KDEMultivariate(data=X, var_type='c'*n_vars, bw='normal_reference')
        return est.bw

        

'''        
class ParzenWindowEstimator(ba.BaseEstimator):
    def __init__(self):
        self.estimator = ne.KernelDensity()
        return
        
    def predict_proba(self, X):
        y = np.exp(self.estimator.score_samples(X))        
        y = y.reshape(len(y), 1)
        return np.concatenate(((1-y), y), axis=1)
        
    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        
    def partial_fit(self, X, y, classes=None):
        self.estimator.partial_fit(X, y, classes)
        
'''