# Tensforflow multiple-output-regression
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def dfSelCols(df, count, colName):  # count: threshold of the number of unique value for each variable
    
    cols = df.columns.tolist()
    
    outDf = pd.DataFrame(columns=[colName])
    
    for col in cols:
        if len(list(df[col].unique())) < count:  
            outDf.loc[col] = 0
        else:            
            outDf.loc[col] = 1                   
    return outDf

# data inputs
inDir = '/p/scratch/cjicg41/chenh/ML_spinup/samples4ai_Soil_final'
outDir = '/p/scratch/cjicg41/chenh/ML_spinup/selVars_soil_threshods'

try:
    os.mkdir(outDir)
except FileExistsError:
    pass


outDf = pd.DataFrame()
outfile = outDir+os.sep+'selYcols_0.9.csv'

for i in range(1, 4):
    scol = 'soilCol' + str(i)     
    
    print(scol)  
    
    df = pd.read_csv(inDir + os.sep + scol + '.csv')       
        
    # key point: inputs selection!
    df = df.drop(['LATIXY', 'LONGXY'], axis=1)      
    
    sel_vars = df.columns.tolist()

    ind = sel_vars.index("soil1c_vr_1")

    Xdf = df[sel_vars[:ind]]
    Ydf = df[sel_vars[ind:]]
    
    df_train, df_test = train_test_split(df, test_size=0.2)
    
    x_train, x_test = df_train[Xdf.columns], df_test[Xdf.columns]
    y_train, y_test = df_train[Ydf.columns], df_test[Ydf.columns]
    
    # count = int(df.shape[0] * 0.8 * 0.01) # batch_size
    count = int(df.shape[0] * 0.9)
   
    curDf = dfSelCols(Ydf, count, scol)  
            
    outDf = pd.concat([outDf, curDf], axis=1)

outDf.to_csv(outfile, index = True)
    
    

# for count in [10, 50, 100]:    
#     outDf = pd.DataFrame()
    
#     outfile = outDir+os.sep+'selYcols_'+str(count)+'.csv'

#     for i in range(1, 4):
#         scol = 'soilCol' + str(i)     
        
#         print(scol)  
        
#         df = pd.read_csv(inDir + os.sep + scol + '.csv')       
            
#         # key point: inputs selection!
#         df = df.drop(['LATIXY', 'LONGXY'], axis=1)      
        
#         sel_vars = df.columns.tolist()

#         ind = sel_vars.index("soil1c_vr_1")

#         Xdf = df[sel_vars[:ind]]
#         Ydf = df[sel_vars[ind:]]
        
#         df_train, df_test = train_test_split(df, test_size=0.2)
        
#         x_train, x_test = df_train[Xdf.columns], df_test[Xdf.columns]
#         y_train, y_test = df_train[Ydf.columns], df_test[Ydf.columns]
        
        
#         curDf = dfSelCols(Ydf, count, scol)  
              
#         outDf = pd.concat([outDf, curDf], axis=1)
    
#     outDf.to_csv(outfile, index = True)


