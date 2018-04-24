import numpy as np
import sklearn.base as ba
import statsmodels.api as sm

class ParzenWindowB(ba.BaseEstimator, ba.ClassifierMixin):    
    """
    Parzen Window Bayes (ParzenWindowB)
    """
    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth
        
    def fit(self, X, y):
        n_samples, n_vars = X.shape
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [sm.nonparametric.KDEMultivariate(data=Xi, var_type='c'*n_vars, bw=[self.bandwidth]*n_vars)
                        for Xi in training_sets]
        return self
            
    def predict_proba(self, X):
        probs = np.array([model.pdf(X)
                             for model in self.models_]).T        
        # somatório é o fator de normalização que garante probabilidades entre 0 e 1
        norm = probs.sum(1, keepdims=True)
        # evita divisão por zero
        norm[norm == 0.0] = 1e-10        
        return probs / norm
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

class WindowSizeOptimizer():
    def bw(self, X):
        n_samples, n_vars = X.shape
        est = sm.nonparametric.KDEMultivariate(data=X, var_type='c'*n_vars, bw='normal_reference')
        return est.bw