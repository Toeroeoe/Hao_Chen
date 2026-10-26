# Tensforflow multiple-output-regression
import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam

print(tf.__version__)

import optuna
from optuna.integration import TFKerasPruningCallback

#Data visualization
from pydot import *
import matplotlib.pyplot as plt
from sklearn.metrics import *
from permetrics.regression import RegressionMetric

from scipy.stats import gaussian_kde
from scipy.stats import linregress

import warnings
warnings.filterwarnings('ignore') 

from matplotlib import rcParams
config={"font.size":16,"mathtext.fontset":'stix'}
rcParams.update(config)

import pickle

# extremely small value
eps = np.finfo(np.float64).eps


# the MLP model based on tensorflow + keras
def create_model(trial, optimizer, ntargets):
       
    model = keras.models.Sequential()
    activation = trial.suggest_categorical("activation", ['relu', 'sigmoid']) # 0~1
    n_layers = trial.suggest_int("n_layers", 1, 4)  
    for i in range(n_layers):
        
        num_hidden = trial.suggest_int("n_units_{}".format(i), 64, 1024, log=True)
        model.add(layers.Dense(num_hidden, activation=activation))
        # dropout = trial.suggest_float("dropout_{}".format(i), 0.1, 0.9, step=0.1)
        # model.add(layers.Dropout(rate=dropout))
        
    model.add(layers.Dense(ntargets, activation="linear"))

    model.compile(
        loss="mse",
        optimizer= optimizer,
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    return model

def create_optimizer(trial):
    
    kwargs = {}
    optimizer_options = ["Adam"] # , "SGD", it seems that Adam was always the best option
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    
    if  optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-3, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer

def mlp_objective(trial, x_train, y_train, x_valid, y_valid, outdir):
    
    tf.keras.backend.clear_session()
    
    ntargets = len(y_train.columns)
    
    optimizer = create_optimizer(trial)
    
    model = create_model(trial, optimizer, ntargets)
        
    epochs = 60    
    batch_size = int(x_train.shape[0] * 0.01)
    model.fit(x_train,
              y_train,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[TFKerasPruningCallback(trial, "val_loss")],
              validation_data=(x_valid, y_valid),
              verbose=1)
    
    # Save a trained model to a file.
    with open(outdir+os.sep+"{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(model, fout)
    
    score = model.evaluate(x_valid, y_valid, verbose=0)
    
    return score[0]


def dfSelCols(df, count):  # count: threshold of the number of unique value for each variable
    
    cols = df.columns.tolist()
    
    for col in cols:
        if len(list(df[col].unique())) < count:      
            print(col + ' is excluded')      
            df = df.drop([col], axis=1)  
                      
    return df


# data inputs
inDir = '/p/scratch/cjicg41/chenh/ML_spinup/samples4ai_PFT_final'
outDir = '/p/scratch/cjicg41/chenh/ML_spinup/mlp_optuna_pft_final_pco2_240'

try:
    os.mkdir(outDir)
    os.mkdir(outDir + os.sep + 'models')
    os.mkdir(outDir + os.sep + 'results')
    os.mkdir(outDir + os.sep + 'trials')
    os.mkdir(outDir + os.sep + 'inputs')    
    os.mkdir(outDir + os.sep + 'predictions')
    os.mkdir(outDir + os.sep + 'test_observations')
    os.mkdir(outDir + os.sep + 'test_predictions')
except FileExistsError:
    pass

for i in range(11, 12):
    if i < 10:
        pft = 'pft0' + str(i)
    else:
        pft = 'pft' + str(i)
    
    try:
        os.mkdir(outDir + os.sep + 'results' + os.sep + pft)
        os.mkdir(outDir + os.sep + 'trials' + os.sep + pft)
    except FileExistsError:
        pass


    if pft in ['pft06']: # 'pft04', 
        continue
    else:
        df0 = pd.read_csv(inDir + os.sep + pft + '.csv') 
        df1 = df0      
                
        # key point: inputs selection!
        df1 = df1.drop(['LATIXY', 'LONGXY'], axis=1)       
        
        # selThres = int(df0.shape[0] * 0.8 * 0.01)        
        selThres = 10
         
        df = dfSelCols(df1, selThres)  
        
        sel_vars = df.columns.tolist()

        ind = sel_vars.index("ZWT0")+1               

        Xdf = df[sel_vars[:ind]]            
        Ydf = df[['pco2_240']]
        
        inDf = Xdf
        inDf['LONGXY'] = np.array(df0['LONGXY'])
        inDf['LATIXY'] = np.array(df0['LATIXY'])
        
        inDf.to_csv(outDir + os.sep + 'inputs' + os.sep + pft + '.csv', index=False) 
        print('inputs saved')
        
        print('input cols: ',Xdf.columns)
        print('target  cols: ',Ydf.columns)
               
        
        #Train test split
        curDf = pd.concat([Xdf,Ydf],axis=1)
        
        df_train, df_test = train_test_split(curDf, test_size=0.2, random_state=100)
        
        x_train, x_test = df_train[Xdf.columns], df_test[Xdf.columns]
        y_train, y_test = df_train[Ydf.columns], df_test[Ydf.columns]        
        
        y_test.to_csv(outDir + os.sep + 'test_observations' + os.sep + pft + '.csv', index=False)
        print('test_observations saved')
        
         
        # Normalizing the data set: Z-Score
        inputs_norm = (Xdf - x_train.mean()) / x_train.std()              
        
        x_train_norm = (x_train - x_train.mean()) / x_train.std()
        x_test_norm = (x_test - x_train.mean()) / x_train.std()     
        
        # Normalizing the data set: Max-Min
        # x_train_norm = (x_train - x_train.min()) / (x_train.max() - x_train.min())
        # x_test_norm = (x_test - x_train.min()) / (x_train.max() - x_train.min())
        
        y_train_norm = (y_train - y_train.min()) / (y_train.max() - y_train.min())
        y_test_norm = (y_test - y_train.min()) / (y_train.max() - y_train.min())
        
        print('train X set shape: ', x_train_norm.shape)
        print('test Y set shape: ', y_test_norm.shape)
        
        # hyper-parameters tuning
        mlp_study = optuna.create_study(direction='minimize',
                                        study_name='MLP Regression')
        
        func = lambda trial: mlp_objective(trial, x_train_norm, y_train_norm, x_test_norm, y_test_norm, outDir+os.sep+'trials'+os.sep+pft)
        
        mlp_study.optimize(func, n_trials=90)
        
        print("Number of finished trials: ", len(mlp_study.trials))

        
        best_trial_number = mlp_study.best_trial.number        
        print("Best trial: " + str(best_trial_number))
        
        trial = mlp_study.best_trial
                
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
                
        # Load the best model        
        with open(outDir+os.sep+'trials'+os.sep+pft+os.sep+"{}.pickle".format(mlp_study.best_trial.number), "rb") as fin:
            model = pickle.load(fin)
        
        # # Load the best model      
        # mlp_study_best_trial_number = 45  
        # with open(outDir+os.sep+'trials'+os.sep+pft+os.sep+"{}.pickle".format(mlp_study_best_trial_number), "rb") as fin:
        #     model = pickle.load(fin)

        model.summary()
        
        model.save(outDir + os.sep + 'models' + os.sep + pft + os.sep,
                   save_format='tf')
        
        # Run predict with restored model
        all_predictions_norm = model.predict(inputs_norm)        
        test_predictions_norm = model.predict(x_test_norm)
        
        all_predictions_df = pd.DataFrame()   
        test_predictions_df = pd.DataFrame()  
        
        for t in range(len(Ydf.columns)):
            print(pft + ' ' + Ydf.columns[t])
            
            y_all_pred_norm = np.array(all_predictions_norm[:,t])
            y_test_pred_norm = np.array(test_predictions_norm[:,t])
            
            y_all_pred = y_all_pred_norm * (
                y_train[Ydf.columns[t]].max() - y_train[Ydf.columns[t]].min()) + y_train[Ydf.columns[t]].min()
            
            all_predictions_df[Ydf.columns[t]] = y_all_pred
            
            y_test_pred = y_test_pred_norm * (
                y_train[Ydf.columns[t]].max() - y_train[Ydf.columns[t]].min()) + y_train[Ydf.columns[t]].min()
            
            test_predictions_df[Ydf.columns[t]] = y_test_pred
        
        all_predictions_df['LONGXY'] = np.array(df0['LONGXY'])
        all_predictions_df['LATIXY'] = np.array(df0['LATIXY'])
        
        all_predictions_df.to_csv(outDir + os.sep + 'predictions' + os.sep + pft + '.csv', index = False)
        print('predictions saved')
        test_predictions_df.to_csv(outDir + os.sep + 'test_predictions' + os.sep + pft + '.csv', index = False)
        print('test_predictions saved')
        
         