import re
import pandas as pd

class ResultFormatter(): 
    def __init__(self, names, n_splits):
        self.names = names
        self.n_splits = n_splits
    
    def param_to_col_name(self, params_dict):
        name = map(lambda x: self.names[x], params_dict.values())
        return '_'.join(name)

    def format(self, cv_results):
        result = pd.DataFrame(data=cv_results)
        score_columns = filter(lambda key: re.match(r'split\d+_test_score', key), cv_results.keys())
        score_columns = list(score_columns)
        result = result[['params'] + score_columns]
        result['params'] = result['params'].apply(self.param_to_col_name)
        result = result.T
        result.columns = result.iloc[0]
        result = result.drop(['params'])
        result.index = map(lambda idx: int(re.search(r'split(\d+)_test_score', idx).group(1)), score_columns)
        result.columns = result.columns.rename('repetition')
        result = result.infer_objects()
        result = result.groupby(result.index // self.n_splits).mean()
        return result  
        