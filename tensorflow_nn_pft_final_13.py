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
outDir = '/p/scratch/cjicg41/chenh/ML_spinup/mlp_optuna_pft_final'

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

shared_excluded_targets = ['availc', 'cpool' , 'excess_cflux', 
                           'gresp_storage', 'gresp_xfer',
                           'plant_calloc', 'plant_ndemand', 'storage_cdemand', 
                           'tempavg_agnpp', 'tempavg_bgnpp',
                           'pft_ctrunc', 'pft_ntrunc', 'npool', 'pco2_240']

excluded_targets_dict = {'pft01': [],
                         'pft02': [],
                         'pft03': ['deadcrootc_xfer', 'deadcrootn_xfer', 'deadstemc_xfer', 'deadstemn_xfer',
                                   'frootc', 'frootc_xfer', 'frootn', 'frootn_xfer', 
                                   'leafc', 'leafc_xfer', 'leafn', 'leafn_xfer',
                                   'livecrootc_xfer', 'livecrootn_xfer', 'livestemc_xfer', 'livestemn_xfer'],
                         'pft04': [],
                         'pft05': [],
                         'pft06': [],
                         'pft07': ['deadcrootc_xfer', 'deadcrootn_xfer', 'deadstemc_xfer', 'deadstemn_xfer',
                                   'frootc', 'frootc_xfer', 'frootn', 'frootn_xfer', 
                                   'leafc', 'leafc_xfer', 'leafn', 'leafn_xfer',
                                   'livecrootc_xfer', 'livecrootn_xfer', 'livestemc_xfer', 'livestemn_xfer'],
                         'pft08': ['deadcrootc_xfer', 'deadcrootn_xfer', 'deadstemc_xfer', 'deadstemn_xfer',
                                   'frootc', 'frootc_xfer', 'frootn', 'frootn_xfer', 
                                   'leafc', 'leafc_xfer', 'leafn', 'leafn_xfer',
                                   'livecrootc_xfer', 'livecrootn_xfer', 'livestemc_xfer', 'livestemn_xfer'],
                         'pft09': [],
                         'pft10': ['deadcrootc_xfer', 'deadcrootn_xfer', 'deadstemc_xfer', 'deadstemn_xfer',
                                   'frootc', 'frootc_xfer', 'frootn', 'frootn_xfer', 
                                   'leafc', 'leafc_xfer', 'leafn', 'leafn_xfer',
                                   'livecrootc_xfer', 'livecrootn_xfer', 'livestemc_xfer', 'livestemn_xfer'],
                         'pft11': ['deadcrootc_xfer', 'deadcrootn_xfer', 'deadstemc_xfer', 'deadstemn_xfer',
                                   'frootc', 'frootc_xfer', 'frootn', 'frootn_xfer', 
                                   'leafc', 'leafc_xfer', 'leafn', 'leafn_xfer',
                                   'livecrootc_xfer', 'livecrootn_xfer', 'livestemc_xfer', 'livestemn_xfer'],
                         'pft12': ['deadcrootc','deadcrootc_storage','deadcrootc_xfer', 
                                   'deadcrootn','deadcrootn_storage','deadcrootn_xfer', 
                                   'deadstemc','deadstemc_storage','deadstemc_xfer', 
                                   'deadstemn','deadstemn_storage','deadstemn_xfer',
                                   'frootc', 'frootc_xfer', 'frootn', 'frootn_xfer', 
                                   'leafc', 'leafc_xfer', 'leafn', 'leafn_xfer',
                                   'livecrootc','livecrootc_storage','livecrootc_xfer', 
                                   'livecrootn','livecrootn_storage','livecrootn_xfer', 
                                   'livestemc', 'livestemc_storage','livestemc_xfer', 
                                   'livestemn', 'livestemn_storage','livestemn_xfer'],
                         'pft13': ['deadcrootc','deadcrootc_storage','deadcrootc_xfer', 
                                   'deadcrootn','deadcrootn_storage','deadcrootn_xfer', 
                                   'deadstemc','deadstemc_storage','deadstemc_xfer', 
                                   'deadstemn','deadstemn_storage','deadstemn_xfer',
                                   'frootc_xfer','frootn_xfer', 'leafc_xfer','leafn_xfer',
                                   'livecrootc','livecrootc_storage','livecrootc_xfer', 
                                   'livecrootn','livecrootn_storage','livecrootn_xfer',
                                   'livestemc', 'livestemc_storage','livestemc_xfer', 
                                   'livestemn', 'livestemn_storage','livestemn_xfer'],
                         'pft14': ['deadcrootc','deadcrootc_storage','deadcrootc_xfer', 
                                   'deadcrootn','deadcrootn_storage','deadcrootn_xfer', 
                                   'deadstemc','deadstemc_storage','deadstemc_xfer', 
                                   'deadstemn','deadstemn_storage','deadstemn_xfer',
                                   'frootc_xfer','frootn_xfer', 'leafc_xfer','leafn_xfer',
                                   'livecrootc','livecrootc_storage','livecrootc_xfer', 
                                   'livecrootn','livecrootn_storage','livecrootn_xfer',
                                   'livestemc', 'livestemc_storage','livestemc_xfer', 
                                   'livestemn', 'livestemn_storage','livestemn_xfer'],
                         'pft15': ['deadcrootc','deadcrootc_storage','deadcrootc_xfer', 
                                   'deadcrootn','deadcrootn_storage','deadcrootn_xfer', 
                                   'deadstemc','deadstemc_storage','deadstemc_xfer', 
                                   'deadstemn','deadstemn_storage','deadstemn_xfer',
                                   'frootc_xfer','frootn_xfer', 'leafc_xfer','leafn_xfer',
                                   'livecrootc','livecrootc_storage','livecrootc_xfer', 
                                   'livecrootn','livecrootn_storage','livecrootn_xfer',
                                   'livestemc', 'livestemc_storage','livestemc_xfer', 
                                   'livestemn', 'livestemn_storage','livestemn_xfer'],
                         'pft16': ['deadcrootc','deadcrootc_storage','deadcrootc_xfer', 
                                   'deadcrootn','deadcrootn_storage','deadcrootn_xfer', 
                                   'deadstemc','deadstemc_storage','deadstemc_xfer', 
                                   'deadstemn','deadstemn_storage','deadstemn_xfer',
                                   'frootc_xfer','frootn_xfer', 'leafc_xfer','leafn_xfer',
                                   'livecrootc','livecrootc_storage','livecrootc_xfer', 
                                   'livecrootn','livecrootn_storage','livecrootn_xfer',
                                   'livestemc', 'livestemc_storage','livestemc_xfer', 
                                   'livestemn', 'livestemn_storage','livestemn_xfer']}

for i in range(13, 14):
    if i < 10:
        pft = 'pft0' + str(i)
    else:
        pft = 'pft' + str(i)
    
    try:
        os.mkdir(outDir + os.sep + 'results' + os.sep + pft)
        os.mkdir(outDir + os.sep + 'trials' + os.sep + pft)        
    except FileExistsError:
        pass

    if pft in ['pft06']:
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
        Ydf = df[sel_vars[ind:]]
        
        for shared_target in shared_excluded_targets:
            if shared_target in Ydf.columns:
                print(shared_target + ' is excluded for '+ pft)      
                Ydf = Ydf.drop([shared_target], axis=1)
                                
        if pft not in ['pft01', 'pft02', 'pft04', 'pft05', 'pft06', 'pft09']:
            for specific_target in excluded_targets_dict[pft]:
                if specific_target in Ydf.columns:
                    print(specific_target + ' is excluded for '+ pft)      
                    Ydf = Ydf.drop([specific_target], axis=1)
                  
        inDf = Xdf
        inDf['LONGXY'] = np.array(df0['LONGXY'])
        inDf['LATIXY'] = np.array(df0['LATIXY'])
        
        inDf.to_csv(outDir + os.sep + 'inputs' + os.sep + pft + '.csv', index=False) 
        print('inputs saved')
        
        print('input cols: ',Xdf.columns)
        print('target  cols: ',Ydf.columns)
        
        
        #Train test split
        curDf = pd.concat([Xdf,Ydf], axis=1)
              
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
        # best_trial_number = 34    
        # with open(outDir+os.sep+'trials'+os.sep+pft+os.sep+"{}.pickle".format(best_trial_number), "rb") as fin:
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
            
