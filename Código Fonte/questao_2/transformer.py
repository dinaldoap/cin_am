import pandas as pd
from sklearn import base

class BaseView(base.TransformerMixin):
    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s()' % (class_name)    

class CompleteView(BaseView):
    '''
    Mantém todas as variáveis.
    '''
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):                
        return X
    
    
class ShapeView(BaseView):
    '''
    Mantém as variáveis relacionadas à shape.
    '''
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        last_col = X.columns.get_loc('HEDGE-SD') + 1
        return X[X.columns[1:last_col]]
    
class RgbView(BaseView):
    '''
    Mantém as variáveis relacionadas à rgb.
    '''
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        first_col = X.columns.get_loc('INTENSITY-MEAN')
        return X[X.columns[first_col:]]