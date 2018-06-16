import numpy as np
import statsmodels.api as sm
import sklearn.base as ba
from sklearn import model_selection as ms
from sklearn.externals.joblib import memory as me

class ParzenWindowB(ba.BaseEstimator, ba.ClassifierMixin):    
    """
    Parzen Window Bayes (ParzenWindowB)
    """
    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth        
        
    def fit(self, X, y):
        bandwidth = self.bandwidth
        if bandwidth is None:
            bandwidth = BandwidthOptimizer().best_bw(X, y)
        n_samples, n_vars = X.shape
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [sm.nonparametric.KDEMultivariate(data=Xi, var_type='c'*n_vars, bw=[bandwidth]*n_vars)
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

class BandwidthOptimizer():
    def __init__(self):
        self.memory = me.Memory(cachedir='.cache', verbose=0)
        
    def best_bw(self, X, y):
        best_bw_cached = self.memory.cache(_best_bw)
        return best_bw_cached(self, X, y)
    
    def clear_cache(self):
        self.memory.clear()
                                           
def _best_bw(bw_opt, X, y):        
    # o grid de parâmetros abaixo define os possíveis valores da janela
    param_grid = {'bandwidth': [3, 4, 5]}
    grid_search = ms.GridSearchCV(ParzenWindowB(), param_grid=param_grid, cv=ms.StratifiedKFold(n_splits=5),return_train_score=False)
    grid_search.fit(X, y)
    return grid_search.best_params_['bandwidth']  