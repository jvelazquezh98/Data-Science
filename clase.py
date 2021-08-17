import pandas as pd
import nltk
import re
import math
from sklearn.preprocessing import OneHotEncoder


class Headers():
    def fit(self,X,y=None):
        self.columns = ['c_acousticness', 't_artists', 'c_danceability', 'c_duration_ms',
       'c_energy', 'v_explicit', 't_id', 'c_instrumentalness', 'v_key',
       'c_liveness', 'c_loudness', 'v_mode', 't_name', 'v_popularity',
       'd_release_date', 'c_speechiness', 'c_tempo', 'c_valence', 'v_year']
        return self

    def transform(self,X):
        X_1 = X.copy()
        X_1.columns = self.columns
        return X_1
    
class Engineering():
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X_1 = X.copy()
        X_1['t_artists'] = X_1['t_artists'].apply(lambda x: x.replace('[','').replace(']','').replace('"','').replace("'",''))
        X_1['t_artists_token'] = X_1['t_artists'].apply(lambda x: [i.strip() for i in x.split(',')])
        X_1['v_num_artists'] = X_1['t_artists_token'].apply(lambda x: len(x))
        X_1['t_artist'] = X_1['t_artists_token'].apply(lambda x: x[0])
        return X_1
    
class Drop_columns():
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X_1 = X.copy()
        X_1.drop(['v_year','c_loudness','c_valence','v_mode','v_key','t_artists_token','t_name','d_release_date','t_id','t_artists']
                ,axis=1, inplace=True)
        return X_1
    
class Get_popularity():
    def fit(self,X,y=None):
        df1 = pd.read_csv('data_by_artist.csv')
        df1.rename({'artists':'t_artist'},axis=1,inplace=True)
        df1.rename({'popularity':'c_mean_popularity'},axis=1,inplace=True)
        df1['t_artist'] = df1['t_artist'].apply(lambda x: x.replace('[','').replace(']','').replace('"','').replace("'",''))
        self.df_aux = df1.copy()
        return self
    
    def transform(self,X):
        X_1 = X.copy()
        X_2 = pd.merge(X_1,self.df_aux[['t_artist','c_mean_popularity']],on='t_artist',how='left')
        X_2['c_mean_popularity'] = X_2['c_mean_popularity'].fillna(X_2['c_mean_popularity'].mean())
        X_2['c_mean_popularity'] = X_2['c_mean_popularity'].apply(lambda x: math.floor(x/10)*10)
        X_2.drop('t_artist',axis=1,inplace=True)
        return X_2
    
class Encode_popularity():
    def fit(self,X,y=None):
        self.enc = OneHotEncoder(drop='first')
        self.column = pd.DataFrame([0,10,20,30,40,50,60,70,80,90],columns=['c_mean_popularity'])
        return self
    
    def transform(self,X):
        X_1 = X.copy()
        self.enc.fit(self.column)
        df_aux_1 = X_1[[x for x in X_1 if x not in ['c_mean_popularity']]].copy()
        df_aux_1.reset_index(drop=True,inplace=True)
        df_aux_2 = pd.DataFrame(self.enc.transform(X_1[['c_mean_popularity']]).toarray(),columns=list(self.enc.get_feature_names()))
        X_1 = pd.concat([df_aux_1,df_aux_2],axis=1).copy()
        return X_1
        
class Manage_tgt():
    def fit(self,X,y=None):
        self.column_name = 'popularity'
        return self
    
    def transform(self,X,y=None):
        X_1 = X.copy()
        X_1[self.column_name] = 0
        return X_1
    
class Encode_popularity_pred():
    def fit(self,X,y=None):
        self.enc = OneHotEncoder(drop='first')
        self.column = pd.DataFrame([0,10,20,30,40,50,60,70,80,90],columns=['c_mean_popularity'])
        return self
    
    def transform(self,X):
        X_1 = X.copy()
        self.enc.fit(self.column)
        df_aux_1 = X_1[[x for x in X_1 if x not in ['c_mean_popularity']]].copy()
        df_aux_1.reset_index(drop=True,inplace=True)
        df_aux_2 = pd.DataFrame(self.enc.transform(X_1[['c_mean_popularity']]).toarray(),columns=list(self.enc.get_feature_names()))
        X_1 = pd.concat([df_aux_1,df_aux_2],axis=1).copy()
        return X_1

class Drop_columns_pred():
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X_1 = X.copy()
        X_1.drop(['v_year','c_loudness','c_valence','v_mode','v_key','t_artists_token','t_name','d_release_date','t_id','t_artists','v_popularity']
                ,axis=1, inplace=True)
        return X_1