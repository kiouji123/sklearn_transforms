from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.utils import resample

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
  

# All sklearn Transforms must have the `transform` and `fit` methods
class Render(BaseEstimator, TransformerMixin):
    def __init__(self, df):
        self.df = df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = self.df.copy()
        df_majority = data[data.OBJETIVO=='Aceptado']
        df_minority = data[data.OBJETIVO=='Sospechoso']
        df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=8873,    # to match majority class
                                 random_state=12)
        # Retornamos um novo dataframe sem as colunas indesejadas
        return  pd.concat([df_majority, df_minority_upsampled])
  
    
    
    
    
    
    

class DropRow(BaseEstimator, TransformerMixin):
    def __init__(self,numero):
        self.numero = numero

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.dropna(how='any',axis=self.numero)
