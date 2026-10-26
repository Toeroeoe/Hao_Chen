from netCDF4 import Dataset
import xarray as xr
import numpy as np
import pandas as pd
import os


def getNCback_df(array, df): 
    indices = np.argwhere(~np.isnan(array.flatten()))
    outArray = array.copy()
    outArray.flat[indices] = df.values

    return array

def getNCback_value(array, value): 
    indices = np.argwhere(~np.isnan(array.flatten()))
    outArray = array.copy()
    outArray.flat[indices] = value

    return outArray

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

soil_Vars = [
    'soil1c_vr', 'soil1n_vr', 'soil2c_vr', 'soil2n_vr', 'soil3c_vr', 'soil3n_vr'
]

inDir = '/p/scratch/cjicg41/chenh/ML_spinup'
outDir = '/p/largedata/jicg41/ninjala/ml_spinup_predictions_20231203'
try:
    os.mkdir(outDir)
except:
    pass

pft_iniData = xr.open_dataset(inDir+os.sep+'initinal_sates'+os.sep+'CLM5_rst_2d_joined.nc')
soil_iniData = xr.open_dataset(inDir+os.sep+'initinal_sates'+os.sep+'CLM5_rst_2d_joined_soil.nc')

pft_prediction_dir = inDir+os.sep+'mlp_optuna_pft_final'+os.sep+'predictions'
pft_pco2_240_prediction_dir = inDir+os.sep+'mlp_optuna_pft_final_pco2_240'+os.sep+'predictions'
soil_prediction_dir = inDir+os.sep+'mlp_optuna_soil_final'+os.sep+'predictions'


out_pft_iniData = pft_iniData
out_soil_iniData = soil_iniData


# leafc, frootc 20; leafc_storage, frootc_storage 0
evergreenPFTs = [1,2,4,5,9]
# leafc, frootc 0; leafc_storage, frootc_storage 0
cropPFTs = [15,16]

# leafc, frootc 0; leafc_storage, frootc_storage 20
otherPFTs = [3,6,7,8,10,11,12,13]

# deadstemc 0.1
woodyPFTs = [1,2,3,4,5,6,7,8,9,10,11]

# PFTs
for pvar in pft_Vars:
    if pvar == 'pco2_240':
        preDir = pft_pco2_240_prediction_dir
    else:
        preDir = pft_prediction_dir
        
    print('processing '+pvar)
    
    for p in range(0,16):
        if p < 9:
            pft = 'pft'+'0'+str(p+1)
        else:
            pft = 'pft'+str(p+1)
            
        print('processing '+ pft)
        
        if pft in ['pft06']:
            continue
        else:        
            predictions_df = pd.read_csv(preDir+os.sep+pft+'.csv')
            predictions_df = predictions_df.drop(['LATIXY', 'LONGXY'], axis=1)     
            
            var_cols = predictions_df.columns
            
            if pvar in var_cols:
                predictions_df.loc[predictions_df[pvar] < 0., pvar] = 0.           
                predictions = predictions_df[pvar]                      
                out_pft_iniData[pvar][p,:,:] = getNCback_df(np.array(pft_iniData[pvar][p,:,:]), predictions)
                
            else:
                out_pft_iniData[pvar][p,:,:] = getNCback_value(np.array(pft_iniData[pvar][p,:,:]), 0.)

out_pft_iniData.to_netcdf(outDir+os.sep+"CLM5_rst_2d_joined_pft_ml.nc")

# Soils
for svar in soil_Vars:            
    preDir = soil_prediction_dir
    
    for col in range(3):        
        scol = 'soilCol' + str(col + 1)
        
        predictions_df = pd.read_csv(preDir+os.sep+scol+'.csv')
        predictions_df = predictions_df.drop(['LATIXY', 'LONGXY'], axis=1) 
          
        var_cols = predictions_df.columns    
    
        for l in range(25):
            curVar = svar + '_' + str(l + 1)
            print('processing '+curVar)
            if curVar in var_cols:
                predictions_df.loc[predictions_df[curVar] < 0., curVar] = 0.
                predictions = predictions_df[curVar]                       
                out_soil_iniData[svar][col,l,:,:] = getNCback_df(np.array(soil_iniData[svar][col,l,:,:]), predictions)
            else:
                out_soil_iniData[svar][col,l,:,:] = getNCback_value(np.array(soil_iniData[svar][col,l,:,:]), 0.)



out_soil_iniData.to_netcdf(outDir+os.sep+"CLM5_rst_2d_joined_soil_ml.nc")

