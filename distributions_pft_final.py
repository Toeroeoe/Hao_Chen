# Tensforflow multiple-output-regression
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

import matplotlib as mpl
import matplotlib.pyplot as plt

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

def plot_hist(outFile, title, inData):
    plt.figure(figsize=(12, 9))
    plt.hist(inData, color='lightgreen', ec='black', bins = 100)
    plt.title(title, fontweight='bold', loc='left', pad=20)
    plt.savefig(outFile, bbox_inches='tight')
    plt.close()

# data inputs
inDir = '/p/scratch/cjicg41/chenh/ML_spinup/samples4ai_PFT_final'
outDir = '/p/scratch/cjicg41/chenh/ML_spinup/distribution_pft_final1'

try:
    os.mkdir(outDir)
except:
    pass

for i in range(1, 17):
    if i < 10:
        pft = 'pft0' + str(i)
    else:
        pft = 'pft' + str(i)
    try:
        os.mkdir(outDir + os.sep + pft)
        os.mkdir(outDir + os.sep + pft + os.sep + 'X')
        os.mkdir(outDir + os.sep + pft + os.sep + 'Y')
    except:
        pass
    
    if pft == 'pft06':
        continue
    else:

        df = pd.read_csv(inDir + os.sep + pft + '.csv')
        
        sel_vars = df.columns.tolist()

        ind = sel_vars.index("availc")

        Xdf = df[sel_vars[:ind]]
        Ydf = df[sel_vars[ind:]]

        for xcol in Xdf.columns:
            print(xcol)
            outFile = outDir+os.sep+pft+os.sep+'X'+os.sep+xcol+'.png'
            inData = Xdf[xcol]
            plot_hist(outFile, xcol, inData)
                
        for ycol in Ydf.columns:
            print(ycol)
            outFile = outDir+os.sep+pft+os.sep+'Y'+os.sep+ycol+'.png'
            inData = Ydf[ycol]
            plot_hist(outFile, ycol, inData)
            