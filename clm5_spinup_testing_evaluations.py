import numpy as np
import pandas as pd
import os

from sklearn.metrics import *
from permetrics.regression import RegressionMetric
from scipy.stats import gaussian_kde
from scipy.stats import linregress

import matplotlib as mpl
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import mpl_scatter_density
from matplotlib import cm
import copy

mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
params = {
        'lines.linewidth':3,
        'axes.facecolor':'white',
        'xtick.color':'k',
        'ytick.color':'k',
        'axes.labelsize': 22,
        'xtick.labelsize':22,
        'ytick.labelsize':22,
        'font.size':22,
        'text.usetex': False,
        "svg.fonttype": 'none'
}
plt.rcParams.update(params)

def newplot_density(outfile, true, predictions, title, ftext, unit):
    
    cmap = copy.copy(cm.get_cmap("Spectral_r"))
    cmap.set_under(alpha=0)
    
    fig = plt.figure(figsize=(13.5, 10.5))
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(true, predictions, vmin=0.5, vmax=np.nanmax,cmap=cmap, dpi=40)
    
    p1 = max(max(predictions), max(true))
    p2 = min(min(predictions), min(true))    
    ax.set_xlim(p2, p1)
    ax.set_ylim(p2, p1)
    
    slope, intercept, r_value, p_value, std_err = linregress(true, predictions)
    regress_values = slope * true + intercept
    plt.plot(true, regress_values, '--', color='red', label='_nolegend_')
    plt.plot([p1, p2], [p1, p2], color='dimgray', label='_nolegend_')  # 1:1 line
    plt.title(title, fontweight='bold', loc='left', pad=20)
    plt.xlabel('True Values'+' ('+unit+')')
    plt.ylabel('Predictions'+' ('+unit+')')
    
    fig.colorbar(density, label='Density')
    
    plt.text(0.03,
             0.97,
             ftext,
             ha='left',
             va='top',
             transform=ax.transAxes)
    
    fig.savefig(outfile, dpi=300, bbox_inches='tight')


pft_Vars = ['availc', 'cpool', 'npool', 
         'deadcrootc', 'deadcrootc_storage', 'deadcrootc_xfer',
         'deadcrootn', 'deadcrootn_storage', 'deadcrootn_xfer',
         'deadstemc', 'deadstemc_storage', 'deadstemc_xfer',
         'deadstemn', 'deadstemn_storage', 'deadstemn_xfer',
         'excess_cflux',
         'frootc', 'frootc_storage', 'frootc_xfer',
         'frootn', 'frootn_storage', 'frootn_xfer',
         'gresp_storage', 'gresp_xfer',         
         'leafc', 'leafc_storage', 'leafc_xfer',
         'leafn', 'leafn_storage', 'leafn_xfer',         
         'livecrootc', 'livecrootc_storage', 'livecrootc_xfer',
         'livecrootn', 'livecrootn_storage', 'livecrootn_xfer',         
         'livestemc', 'livestemc_storage', 'livestemc_xfer',
         'livestemn', 'livestemn_storage', 'livestemn_xfer',         
         'pft_ctrunc', 'pft_ntrunc',         
         'plant_calloc', 'plant_ndemand', 'storage_cdemand',         
         'totvegc',
         'pco2_240', 'LAI240_VALUE', 'tempavg_agnpp', 'tempavg_bgnpp'   
         ] # length: 52

pftC_Vars = ['cpool', 'deadcrootc', 'deadcrootc_storage', 'deadcrootc_xfer',
             'deadstemc', 'deadstemc_storage', 'deadstemc_xfer',
             'frootc', 'frootc_storage', 'frootc_xfer',   
             'leafc', 'leafc_storage', 'leafc_xfer',   
             'livecrootc', 'livecrootc_storage', 'livecrootc_xfer',      
             'livestemc', 'livestemc_storage', 'livestemc_xfer',  
             'totvegc']

pftN_Vars = ['deadcrootn', 'deadcrootn_storage', 'deadcrootn_xfer',
             'deadstemn', 'deadstemn_storage', 'deadstemn_xfer',
             'frootn', 'frootn_storage', 'frootn_xfer',
             'leafn', 'leafn_storage', 'leafn_xfer',         
             'livecrootn', 'livecrootn_storage', 'livecrootn_xfer',   
             'livestemn', 'livestemn_storage', 'livestemn_xfer']

soil_Vars = [
    'soil1c_vr', 'soil1n_vr', 'soil2c_vr', 'soil2n_vr', 'soil3c_vr', 'soil3n_vr'
]

inDir = '/p/scratch/cjicg41/chenh/ML_spinup'

pft_observation_dir = inDir+os.sep+'mlp_optuna_pft_final'+os.sep+'test_observations'
pft_pco2_240_observation_dir = inDir+os.sep+'mlp_optuna_pft_final_pco2_240'+os.sep+'test_observations'
soil_observation_dir = inDir+os.sep+'mlp_optuna_soil_final'+os.sep+'test_observations'


pft_prediction_dir = inDir+os.sep+'mlp_optuna_pft_final'+os.sep+'test_predictions'
pft_pco2_240_prediction_dir = inDir+os.sep+'mlp_optuna_pft_final_pco2_240'+os.sep+'test_predictions'
soil_prediction_dir = inDir+os.sep+'mlp_optuna_soil_final'+os.sep+'test_predictions'


# PFTs
pft_Vars = ['pco2_240'] #'pco2_240', 'others'
for pvar in pft_Vars:    
    if pvar == 'pco2_240':
        preDir = pft_pco2_240_prediction_dir
        obsDir = pft_pco2_240_observation_dir
        outDir = inDir+os.sep+'mlp_optuna_pft_final_pco2_240'+os.sep+'results'
        unit = '$Pa$'
    else:
        preDir = pft_prediction_dir
        obsDir = pft_observation_dir
        outDir = inDir+os.sep+'mlp_optuna_pft_final'+os.sep+'results'        
        
    print('processing '+pvar)
    
    for p in range(1,17):
        if p < 10:
            pft = 'pft0' + str(p)
        else:
            pft = 'pft' + str(p)
            
        print('processing '+ pft)
        
        if pft in ['pft06']:
            continue
        else:
            observations_df = pd.read_csv(obsDir+os.sep+pft+'.csv')  
            predictions_df = pd.read_csv(preDir+os.sep+pft+'.csv')
            
            var_cols = predictions_df.columns
            
            for var in var_cols:   
                if var in pftC_Vars:
                    unit = '$gC$'+' '+'$m^{-2}$'
                elif var in pftN_Vars:
                    unit = '$gN$'+' '+'$m^{-2}$'
                else:
                    unit = '$m^3$'+' '+'$m^{-3}$'
                outfile = outDir+os.sep+pft+os.sep+var+'_test.png'
                
                predictions_df.loc[predictions_df[var] < 0., var] = 0.      
                
                y_test_obs = np.array(observations_df[var])
                y_test_pred = np.array(predictions_df[var])
                
                n = y_test_obs.shape[0]
                mae = mean_absolute_error(y_test_obs, y_test_pred)
                mse = mean_squared_error(y_test_obs, y_test_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test_obs, y_test_pred)
                slope, intercept, r_value, p_value, std_err = linregress(y_test_obs, y_test_pred)

                evaluator = RegressionMetric(y_test_obs, y_test_pred, decimal=2)
                nrmse = evaluator.normalized_root_mean_square_error()

                title = var
                performance = r'$N=$' + str(format(n,',')) + '\n' 
                performance+= r'$MAE=$' + str(round(mae, 3)) + ' ' + unit + '\n' 
                performance+= r'$RMSE=$' + str(round(rmse, 3)) + ' ' + unit + '\n' 
                performance+= r'$nRMSE=$' + str(round(nrmse, 3)) + '\n' 
                performance+= r'$R^2=$' + str(round(r2, 3)) + '\n' 
                performance+= r'$Slope=$' + str(round(slope, 3))

                newplot_density(outfile, y_test_obs, y_test_pred, title, performance, unit)
                print(var)
            

# # Soils
# preDir = soil_prediction_dir
# obsDir = soil_observation_dir
# outDir = inDir+os.sep+'mlp_optuna_soil_final'+os.sep+'results'

# for col in range(1,4):        
#     scol = 'soilCol' + str(col)
    
#     observations_df = pd.read_csv(obsDir+os.sep+scol+'.csv')        
#     predictions_df = pd.read_csv(preDir+os.sep+scol+'.csv')
        
#     var_cols = predictions_df.columns    
    
#     for var in var_cols:      
#         if var[5] == 'c':
#             unit = '$gC$'+' '+'$m^{-2}$'   
#         else:
#             unit = '$gN$'+' '+'$m^{-2}$'   
                         
#         outfile = outDir+os.sep+scol+os.sep+var+'_test.png'
        
#         predictions_df.loc[predictions_df[var] < 0., var] = 0.      
        
#         y_test_obs = np.array(observations_df[var])
#         y_test_pred = np.array(predictions_df[var])
        
#         n = y_test_obs.shape[0]
#         mae = mean_absolute_error(y_test_obs, y_test_pred)
#         mse = mean_squared_error(y_test_obs, y_test_pred)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(y_test_obs, y_test_pred)
#         slope, intercept, r_value, p_value, std_err = linregress(y_test_obs, y_test_pred)

#         evaluator = RegressionMetric(y_test_obs, y_test_pred, decimal=2)
#         nrmse = evaluator.normalized_root_mean_square_error()

#         title = var               
#         performance = r'$N=$' + str(format(n,',')) + '\n' 
#         performance+= r'$MAE=$' + str(round(mae, 3)) + ' '  + unit + '\n' 
#         performance+= r'$RMSE=$' + str(round(rmse, 3)) + ' '  + unit + '\n' 
#         performance+= r'$nRMSE=$' + str(round(nrmse, 3)) + ' '  + '\n' 
#         performance+= r'$R^2=$' + str(round(r2, 3)) + ' '  + '\n' 
#         performance+= r'$Slope=$' + str(round(slope, 3))

#         newplot_density(outfile, y_test_obs, y_test_pred, title, performance, unit)
#         print(var)
    

