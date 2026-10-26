import os
import numpy as np
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

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
        'axes.labelsize': 18,
        'xtick.labelsize':18,
        'ytick.labelsize':18,
        'font.size':18,
        'text.usetex': False,
        "svg.fonttype": 'none'
}
plt.rcParams.update(params)




country_borders = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_boundary_lines_land',
    scale='50m',
    facecolor='none')


def pool_plot(outfile, lon, lat, var, title, unit = '%'):
    extent=[lon.min(), lon.max(), lat.min(), lat.max()]
    # Set the figure size, projection, and extent
    fig = plt.figure(figsize=(12,9))
    ax = plt.axes(projection=ccrs.PlateCarree()) # ccrs.Robinson()
    # ax.set_global()
    ax.set_extent(extent)
    

    # Set contour levels, then draw the plot and a colorbar
    clevels = np.linspace(-30,30,31)
    plt.contourf(lon, lat, var, levels = clevels, cmap='RdBu_r') # transform=ccrs.PlateCarree(), cmap=plt.cm.jet
    
    # ax.coastlines(resolution="110m",linewidth=0.5)
    ax.add_feature(country_borders, edgecolor='gray')
    # ax.gridlines(linestyle='--',color='black')
    
    plt.title(title, size=18)
    cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    cb.set_label(unit,size=18,rotation=0,labelpad=15)
    cb.ax.tick_params(labelsize=18)
    
    fig.savefig(outfile, format='png', dpi=300)
    plt.close(fig)

# LATIXY, LONGXY

# Initials
## Variables 16*1544*1592
iVars = ['availc', 'cpool', 'npool', 
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
         ] # 52
fluxC_Vars = ['availc', 'excess_cflux', 'plant_calloc', 'tempavg_agnpp', 'tempavg_bgnpp']
fluxN_Vars = ['plant_ndemand']

pftC_Vars = ['cpool', 'gresp_storage', 'gresp_xfer', 'pft_ctrunc', 'storage_cdemand',     
             'deadcrootc', 'deadcrootc_storage', 'deadcrootc_xfer',
             'deadstemc', 'deadstemc_storage', 'deadstemc_xfer',
             'frootc', 'frootc_storage', 'frootc_xfer',   
             'leafc', 'leafc_storage', 'leafc_xfer',   
             'livecrootc', 'livecrootc_storage', 'livecrootc_xfer',      
             'livestemc', 'livestemc_storage', 'livestemc_xfer',  
             'totvegc']

pftN_Vars = ['npool', 'pft_ntrunc',   
             'deadcrootn', 'deadcrootn_storage', 'deadcrootn_xfer',
             'deadstemn', 'deadstemn_storage', 'deadstemn_xfer',
             'frootn', 'frootn_storage', 'frootn_xfer',
             'leafn', 'leafn_storage', 'leafn_xfer',         
             'livecrootn', 'livecrootn_storage', 'livecrootn_xfer',   
             'livestemn', 'livestemn_storage', 'livestemn_xfer']

inDir = "/p/scratch/cjicg41/chenh/ML_spinup"
outDir = "/p/scratch/cjicg41/chenh/ML_spinup/cnPools_plots/"

try:
    os.mkdir(outDir)
except:
    pass

surfData = Dataset(inDir+os.sep+'surface_files'+os.sep+'surfdata_joined.nc')
oriIniData = Dataset(inDir+os.sep+'initinal_sates'+os.sep+'CLM5_rst_2d_joined.nc')
mlpIniData = Dataset('/p/largedata/jicg41/ninjala/ml_spinup_predictions_20231203/CLM5_rst_2d_joined_pft_ml.nc')

lons = np.unique(surfData.variables['LONGXY'][0,:])
lats = np.unique(surfData.variables['LATIXY'][:,0])

# pools
for var in iVars:
    oriOutfile = outDir+os.sep+var+'_ori.png'
    mlpOutfile = outDir+os.sep+var+'_mlp.png'
    difOutfile = outDir+os.sep+var+'_dif.png'
    oriData = oriIniData.variables[var][:,:,:]
    mlpData = mlpIniData.variables[var][:,:,:]
    
    oriVegTot = np.nansum(oriData, axis=0)
    mlpVegTot = np.nansum(mlpData, axis=0)
    
    if oriVegTot.max() == oriVegTot.min():
        print(var+' 0')        
    else:
        difVegTot = (mlpVegTot - oriVegTot) / oriVegTot * 100.
        
        if var in pftC_Vars:
            unit = '$gC$'+' '+'$m^{-2}$'
        elif var in pftN_Vars:
            unit = '$gN$'+' '+'$m^{-2}$'
        elif var in fluxC_Vars:
            unit = '$gC$'+' '+'$m^{-2}$'+' s'
        elif var in fluxN_Vars:
            unit = '$gN$'+' '+'$m^{-2}$'+' s'
        elif var == 'LAI240_VALUE':
            unit = '$m^3$'+' '+'$m^{-3}$'
        elif var == 'pco2_240':
            unit = '$Pa$'
        else:
            print(var+' not exist')
        
        # pool_plot(oriOutfile, lons, lats, oriVegTot, var, unit = unit)
        # pool_plot(mlpOutfile, lons, lats, mlpVegTot, var, unit = unit)
        pool_plot(difOutfile, lons, lats, difVegTot, var, unit = '%')
        print(var+' 1')
    
    
    
