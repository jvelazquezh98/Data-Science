#FUNCION PARA LOS MODELOS
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from plotly.offline import plot,iplot
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import numpy as np
import plotly.graph_objects as go
from palettable.cmocean.diverging import Delta_4
import plotly.express as px


def plot_comparison(true_df,pred,tgt):
    #Function to plot the predicted values vs the real ones
    df_aux_test=pd.DataFrame()
    
    df_aux_test["True Value"] = true_df[tgt]
    df_aux_test["Predicted Value"] = pred
    
    df_aux_test.reset_index(drop=True)[["True Value","Predicted Value"]].iloc[:100,:].iplot(kind='line')

def modeling(df,tgt,modelos,nCV,param_grid_ridge,param_grid_lasso,param_grid_rf,param_grid_gb,param_grid_svr,param_grid_ab):
    #Function used for modeling a dataframe using distinct models
    #[df] : Pandas dataframe with the data, including the target variable
    #[tgt] : String with the name of the target variable
    #[modelos] : List with the models that the user wants to run using the entry 1 : Use model 0 : Don't use the model, for example [1,1,1,1,1,1] will run all the models
    #[nCV] : Number of Crossvalidation iterations that the models will run
    #[param_grid_ridge] : Param grid for Ridge regresion
    #[param_grid_lasso] : Param grid for Lasso regresion
    #[param_grid_rf] : Param grid for Random forest
    #[param_grid_gb] : Param grid for gradient boost
    #[param_grid_svr] : Param grid for SVR
    #[param_grid_ab] : Param grid for adaboost
    
    models=[]
    r2_train=[]
    r2_test=[]
    mse_train=[]
    mse_test=[]
    mae_train=[]
    mae_test=[]
    r2adj_train=[]
    r2adj_test=[]
    best_param=[]
    best_models = {}

    X = df.loc[:,df.columns != tgt]
    y = df.loc[:,df.columns == tgt]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25, random_state=0)
    n_obs_train = int(X_train.shape[0])
    n_obs_test = int(X_test.shape[0])
    n_x = int(X_train.shape[1])

    #REGRESION LINEAL RIDGE
    if modelos[0] == 1:
        print('MODELO: REGRESION RIDGE \n')

        #Escalar variables
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X)
        y = sc_y.fit_transform(y)

        X_train_sc,X_test_sc,y_train_sc,y_test_sc = train_test_split(X,y,test_size=0.25, random_state=0)

        #Grid
        alpha = param_grid_ridge
        param_grid = dict(alpha=alpha)

        estimator = Ridge()
        grid = GridSearchCV(cv=nCV,
              verbose=True,
              scoring='r2',
              estimator=estimator,
              n_jobs=-1,
              param_grid=param_grid)

        grid.fit(X_train_sc, y_train_sc)

        print(grid.best_params_)
        best_model_ridge = grid.best_estimator_
        y_pred_test = best_model_ridge.predict(X_test_sc)
        y_pred_train = best_model_ridge.predict(X_train_sc)
        
        y_test_sc_df = pd.DataFrame(y_test_sc,columns=[tgt])
        plot_comparison(y_test_sc_df,y_pred_test,tgt)

        models.append("Ridge")
        r2_train.append(r2_score(y_train_sc,y_pred_train))
        r2_test.append(r2_score(y_test_sc,y_pred_test))
        r2adj_train.append(1-(((n_obs_train-1)/(n_obs_train-n_x-1)))*(1-r2_score(y_train_sc,y_pred_train)))
        r2adj_test.append(1-(((n_obs_test-1)/(n_obs_test-n_x-1)))*(1-r2_score(y_test_sc,y_pred_test)))       
        mse_train.append(mean_squared_error(y_train_sc,y_pred_train))
        mse_test.append(mean_squared_error(y_test_sc,y_pred_test))
        mae_train.append(mean_absolute_error(y_train_sc,y_pred_train))
        mae_test.append(mean_absolute_error(y_test_sc,y_pred_test))
        
        best_models['RIDGE'] = best_model_ridge
        
    #REGRESION LINEAL LASSO
    if modelos[1] == 1:
        print('MODELO: REGRESION LASSO \n')

        #Escalar variables
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X)
        y = sc_y.fit_transform(y)

        X_train_sc,X_test_sc,y_train_sc,y_test_sc = train_test_split(X,y,test_size=0.25, random_state=0)

        #Grid
        alpha = param_grid_lasso
        param_grid = dict(alpha=alpha)

        estimator = Lasso()
        grid = GridSearchCV(cv=nCV,
              verbose=True,
              scoring='r2',
              estimator=estimator,
              n_jobs=-1,
              param_grid=param_grid)

        grid.fit(X_train_sc, y_train_sc)

        print(grid.best_params_)
        best_model_lasso = grid.best_estimator_
        y_pred_test = best_model_lasso.predict(X_test_sc)
        y_pred_train = best_model_lasso.predict(X_train_sc)
        
        y_test_sc_df = pd.DataFrame(y_test_sc,columns=[tgt])
        plot_comparison(y_test_sc_df,y_pred_test,tgt)

        models.append("Lasso")
        r2_train.append(r2_score(y_train_sc,y_pred_train))
        r2_test.append(r2_score(y_test_sc,y_pred_test))
        r2adj_train.append(1-(((n_obs_train-1)/(n_obs_train-n_x-1)))*(1-r2_score(y_train_sc,y_pred_train)))
        r2adj_test.append(1-(((n_obs_test-1)/(n_obs_test-n_x-1)))*(1-r2_score(y_test_sc,y_pred_test))) 
        mse_train.append(mean_squared_error(y_train_sc,y_pred_train))
        mse_test.append(mean_squared_error(y_test_sc,y_pred_test))
        mae_train.append(mean_absolute_error(y_train_sc,y_pred_train))
        mae_test.append(mean_absolute_error(y_test_sc,y_pred_test))
        
        best_models['LASSO'] = best_model_lasso
        
    #RANDOM FOREST
    if modelos[2] == 1:
        print('MODELO: RANDOM FOREST \n')

        #Grid
        param_grid = param_grid_rf
        
        estimator = RandomForestRegressor()
        grid = GridSearchCV(cv=nCV,
              verbose=True,
              scoring='r2',
              estimator=estimator,
              n_jobs=-1,
              param_grid=param_grid)

        grid.fit(X_train, y_train)

        print(grid.best_params_)
        best_model_rf = grid.best_estimator_
        y_pred_test = best_model_rf.predict(X_test)
        y_pred_train = best_model_rf.predict(X_train)
                
        plot_comparison(y_test,y_pred_test,tgt)

        models.append("Random Forest")
        r2_train.append(r2_score(y_train,y_pred_train))
        r2_test.append(r2_score(y_test,y_pred_test))
        r2adj_train.append(1-(((n_obs_train-1)/(n_obs_train-n_x-1)))*(1-r2_score(y_train,y_pred_train)))
        r2adj_test.append(1-(((n_obs_test-1)/(n_obs_test-n_x-1)))*(1-r2_score(y_test,y_pred_test))) 
        mse_train.append(mean_squared_error(y_train,y_pred_train))
        mse_test.append(mean_squared_error(y_test,y_pred_test))
        mae_train.append(mean_absolute_error(y_train,y_pred_train))
        mae_test.append(mean_absolute_error(y_test,y_pred_test))
        
        best_models['RF'] = best_model_rf
        
    #GRADIENT BOOSTING
    if modelos[3] == 1:
        print('MODELO: GRADIENT BOOSTING \n')

        #Grid
        param_grid = param_grid_gb
        
        estimator = GradientBoostingRegressor()
        grid = GridSearchCV(cv=nCV,
              verbose=True,
              scoring='r2',
              estimator=estimator,
              n_jobs=-1,
              param_grid=param_grid)

        grid.fit(X_train, y_train)

        print(grid.best_params_)
        best_model_gb = grid.best_estimator_
        y_pred_test = best_model_gb.predict(X_test)
        y_pred_train = best_model_gb.predict(X_train)
        
        plot_comparison(y_test,y_pred_test,tgt)

        models.append("Gradient Boosting")
        r2_train.append(r2_score(y_train,y_pred_train))
        r2_test.append(r2_score(y_test,y_pred_test))
        r2adj_train.append(1-(((n_obs_train-1)/(n_obs_train-n_x-1)))*(1-r2_score(y_train,y_pred_train)))
        r2adj_test.append(1-(((n_obs_test-1)/(n_obs_test-n_x-1)))*(1-r2_score(y_test,y_pred_test))) 
        mse_train.append(mean_squared_error(y_train,y_pred_train))
        mse_test.append(mean_squared_error(y_test,y_pred_test))
        mae_train.append(mean_absolute_error(y_train,y_pred_train))
        mae_test.append(mean_absolute_error(y_test,y_pred_test))
        
        best_models['GB'] = best_model_gb
        
    #SUPPORT VECTOR MACHINE
    if modelos[4] == 1:
        print('MODELO: SVR \n')

        #Escalar variables
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X)
        y = sc_y.fit_transform(y)

        X_train_sc,X_test_sc,y_train_sc,y_test_sc = train_test_split(X,y,test_size=0.25, random_state=0)

        #Grid
        param_grid = param_grid_svr

        estimator = SVR()
        grid = GridSearchCV(cv=nCV,
              verbose=True,
              scoring='r2',
              estimator=estimator,
              n_jobs=-1,
              param_grid=param_grid)

        grid.fit(X_train_sc, y_train_sc)

        print(grid.best_params_)
        best_model_svr = grid.best_estimator_
        y_pred_test = best_model_svr.predict(X_test_sc)
        y_pred_train = best_model_svr.predict(X_train_sc)
        
        y_test_sc_df = pd.DataFrame(y_test_sc,columns=[tgt])
        plot_comparison(y_test_sc_df,y_pred_test,tgt)

        models.append("SVR")
        r2_train.append(r2_score(y_train_sc,y_pred_train))
        r2_test.append(r2_score(y_test_sc,y_pred_test))
        r2adj_train.append(1-(((n_obs_train-1)/(n_obs_train-n_x-1)))*(1-r2_score(y_train_sc,y_pred_train)))
        r2adj_test.append(1-(((n_obs_test-1)/(n_obs_test-n_x-1)))*(1-r2_score(y_test_sc,y_pred_test))) 
        mse_train.append(mean_squared_error(y_train_sc,y_pred_train))
        mse_test.append(mean_squared_error(y_test_sc,y_pred_test))
        mae_train.append(mean_absolute_error(y_train_sc,y_pred_train))
        mae_test.append(mean_absolute_error(y_test_sc,y_pred_test))
        
        best_models['SVR'] = best_model_svr
        
    #ADABOOST
    if modelos[5] == 1:
        print('MODELO: ADABOOST \n')

        #Grid
        param_grid = param_grid_ab
        
        estimator = AdaBoostRegressor()
        grid = GridSearchCV(cv=nCV,
              verbose=True,
              scoring='r2',
              estimator=estimator,
              n_jobs=-1,
              param_grid=param_grid)

        grid.fit(X_train, list(y_train[tgt]))

        print(grid.best_params_)
        best_model_ab = grid.best_estimator_
        y_pred_test = best_model_ab.predict(X_test)
        y_pred_train = best_model_ab.predict(X_train)
        
        plot_comparison(y_test,y_pred_test,tgt)

        models.append("Adaboost")
        r2_train.append(r2_score(y_train,y_pred_train))
        r2_test.append(r2_score(y_test,y_pred_test))
        r2adj_train.append(1-(((n_obs_train-1)/(n_obs_train-n_x-1)))*(1-r2_score(y_train,y_pred_train)))
        r2adj_test.append(1-(((n_obs_test-1)/(n_obs_test-n_x-1)))*(1-r2_score(y_test,y_pred_test))) 
        mse_train.append(mean_squared_error(y_train,y_pred_train))
        mse_test.append(mean_squared_error(y_test,y_pred_test))
        mae_train.append(mean_absolute_error(y_train,y_pred_train))
        mae_test.append(mean_absolute_error(y_test,y_pred_test))
        
        best_models['AB'] = best_model_ab

    #Consolidacion
    models_df = pd.DataFrame(models,columns=["Model"])
    r2_train_df = pd.DataFrame(r2_train,columns=["R2_train"])
    r2_test_df = pd.DataFrame(r2_test,columns=["R2_test"])
    r2adj_train_df = pd.DataFrame(r2adj_train,columns=["R2adj_train"])
    r2adj_test_df = pd.DataFrame(r2adj_test,columns=["R2adj_test"])
    mse_train_df = pd.DataFrame(mse_train,columns=["MSE train"])
    mse_test_df = pd.DataFrame(mse_test,columns=["MSE test"])
    mae_train_df = pd.DataFrame(mae_train,columns=["MAE train"])
    mae_test_df = pd.DataFrame(mae_test,columns=["MAE test"])


    resumen = pd.concat([models_df,r2_train_df,r2_test_df,r2adj_train_df,r2adj_test_df,mse_train_df,mse_test_df
                      , mae_train_df, mae_test_df],axis=1)
    
    return (resumen, best_models)