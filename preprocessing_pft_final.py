from netCDF4 import MFDataset, Dataset
from cftime import num2date, date2num
from glob import glob
import xarray as xr
import numpy as np
import pandas as pd
import calendar
import os


# Forcings
# # Variables:
#     FLDS - Downward_longwave_radiation, Wm-2, 12*1544*1592
#     FSDS - Downward_shortwave_radiation, Wm-2, 12*1544*1592
#     PRECTmms - Instantaneous_precipitation, mm/s, 12*1544*1592
#     PSRF - Surface_pressure, Pa, 12*1544*1592
#     RH - Relative_humidity, %, 12*1544*1592
#     TBOT - Temperature, K, 12*1544*1592
#     WIND - Wind_speed, m/s, 12*1544*1592

# # maximum, minimum, mean and standard deviation of monthly Vars
# fVars = ['FLDS', 'FSDS', 'PRECTmms', 'PSRF', 'RH', 'TBOT', 'WIND']
# Units = ['Wm-2', 'Wm-2', 'mm', 'Pa', '%', 'K', 'm/s']
# Names = ['downward longwave radiation', 'downward shortwave radiation', 
#          'precipitation', 'surface pressure', 'relative humidity', 'temperature', 'wind speed']

# diri = "/p/largedata/jicg41/chenh/ML_spinUp/forcings/monthly/"
# diro = "/p/largedata/jicg41/chenh/ML_spinUp/forcings/"

# paths = sorted(glob(diri+'*.nc'))

# referNC = Dataset(diri+'1996.nc')
# print(referNC)
# lat2d = referNC['LATIXY'][:]
# lon2d = referNC['LONGXY'][:]

# for var in fVars:
#     print(var)
#     index = fVars.index(var)
#     unit = Units[index]
#     name = Names[index]    
    
    
#     if var != 'PRECTmms':   
#         allData = []     
#         for file in paths:
#             data = Dataset(file)
#             allData.append(data[var])
#             # dates = num2date(data['time'][:], units=data['time'].units, calendar=data['time'].calendar)
#             # print(dates)
#         allData = np.array(allData)        
        
#     else:
#         allData = []
#         # secs of the months
#         year0 = 1995
#         year1 = 2018
        
#         allSecs = []
#         for year in range(year0, year1+1):
#             for month in range(1, 13):
#                 days = calendar.monthrange(year, month)[1]
#                 secs = days*24.*60.*60.
#                 allSecs.append(secs)
#         # print(allSecs)
#         allSecs = np.array(allSecs)   
        
#         for file in paths:
#             data = Dataset(file)
#             allData.append(data[var])
#             # dates = num2date(data['time'][:], units=data['time'].units, calendar=data['time'].calendar)
#             # print(dates)           
#         allData = np.array(allData)         
        
#         mi=0
#         for y in range(allData.shape[0]):
#             for tm in range(allData.shape[1]):
#                 allData[y,tm,:,:] = allData[y,tm,:,:] * allSecs[mi]
#                 mi+=1                
        
#     for m in range(0,12):
#         if m+1 < 10:
#             tmonth = '0'+str(m+1)
#         else:
#             tmonth = str(m+1)
        
#         selData = allData[:,m,:,:]
#         mean = np.mean(selData, axis=0)
#         std = np.std(selData, axis=0)
#         max = np.max(selData, axis=0)
#         min = np.min(selData, axis=0)                
        
#         try: ncfile.close()
#         except: pass

#         ncfile =  Dataset(diro+var+tmonth+'.nc',mode='w',format='NETCDF4_CLASSIC')
        
#         lat_dim = ncfile.createDimension('lat', 1544)     # latitude axis
#         lon_dim = ncfile.createDimension('lon', 1592) 
        
#         meanD = ncfile.createVariable('mean',np.float32,('lat','lon')) # note: unlimited dimension is leftmost
#         meanD.units = unit
#         meanD.standard_name = 'mean monthly' + name
        
#         stdD = ncfile.createVariable('std',np.float32,('lat','lon')) # note: unlimited dimension is leftmost
#         stdD.units = unit
#         stdD.standard_name = 'std monthly' + name
        
#         maxD = ncfile.createVariable('max',np.float32,('lat','lon')) # note: unlimited dimension is leftmost
#         maxD.units = unit
#         maxD.standard_name = 'max monthly' + name
        
#         minD = ncfile.createVariable('min',np.float32,('lat','lon')) # note: unlimited dimension is leftmost
#         minD.units = unit
#         minD.standard_name = 'min monthly' + name


#         meanD[:,:] = mean
#         stdD[:,:] = std
#         maxD[:,:] = max
#         minD[:,:] = min
        
#         print(ncfile)
#         ncfile.close()
#         print('Dataset is closed!')

    
# Surfaces
## Variables
    # F0 - maximum gridcell fractional inundated area, -, 1544*1592
    # FMAX - maximum fractional saturated area, -, 1544*1592
    # MONTHLY_HEIGHT_BOT - monthly height bottom, m, 12*17*1544*1592
    # MONTHLY_HEIGHT_TOP - monthly height top, m, 12*17*1544*1592
    # MONTHLY_LAI -monthly leaf area index, -, 12*17*1544*1592
    # MONTHLY_SAI - monthly stem area index, -, 12*17*1544*1592
    # ORGANIC - organic matter density at soil layers, kg/m3 (assumed carbon content 0.58 gC per gOM), 10*1544*1592
    # PCT_CLAY - percent clay, -, 10*1544*1592
    # PCT_CROP - total percent crop landunit, -, 1544*1592
    # PCT_NAT_PFT -  percent plant functional type on the natural veg landunit (% of landunit), 15*1544*1592
    # PCT_NATVEG - total percent natural vegetation landunit, 1544*1592
    # PCT_SAND - percent sand, -, 10*1544*1592
    # SLOPE - mean topographic slope, degrees, 1544*1592
    # SOIL_COLOR - soil color, -, 1544*1592
    # STD_ELEV - standard deviation of elevation, m, 1544*1592
    # zbedrock - soil depth, m, 1544*1592


# Initials
## Variables 16*1544*1592
    # availc, C flux available for allocation, gC/m^2/s (optional)

    # cpool, temporary photosynthate C pool, gC/m^2
    # npool, temporary plant N pool, gN/m^2

    # deadcrootc, dead coarse root C, gC/m^2
    # deadcrootc_storage, dead coarse root C storage, gC/m^2
    # deadcrootc_xfer, dead coarse root C transfer, gC/m^2

    # deadcrootn, dead coarse root N, gN/m^2
    # deadcrootn_storage, dead coarse root N storage, gN/m^2
    # deadcrootn_xfer, dead coarse root N transfer, gN/m^2

    # deadstemc, dead stem C, gC/m^2
    # deadstemc_storage, dead stem C storage, gC/m^2
    # deadstemc_xfer, dead stem C transfer, gC/m^2

    # deadstemn, dead stem N, gN/m^2
    # deadstemn_storage, dead stem N storage, gN/m^2
    # deadstemn_xfer, dead stem N transfer, gN/m^2

    # excess_cflux, C flux not allocated due to downregulation, gC/m^2/s

    # frootc, fine root C, gC/m^2
    # frootc_storage, fine root C storage, gC/m^2
    # frootc_xfer, fine root C transfer, gC/m^2

    # frootn, fine root N, gN/m^2
    # frootn_storage, fine root N storage, gN/m^2
    # frootn_xfer, fine root N transfer, gN/m^2

    # gresp_storage, growth respiration storage, gC/m^2
    # gresp_xfer, growth respiration transfer, gC/m^2

    # leafc, leaf C, gC/m^2
    # leafc_storage, leaf C storage, gC/m^2
    # leafc_xfer, leaf C transfer, gC/m^2

    # leafn, leaf N, gN/m^2
    # leafn_storage, 	leaf N storage, gN/m^2
    # leafn_xfer, leaf N transfer, gN/m^2

    # livecrootc, live coarse root C, gC/m^2
    # livecrootc_storage, live coarse root C storage, gC/m^2
    # livecrootc_xfer, live coarse root C transfer, gC/m^2

    # livecrootn, live coarse root C transfer, gC/m^2
    # livecrootn_storage, live coarse root N storage, gN/m^2
    # livecrootn_xfer, live coarse root N transfer, gN/m^2

    # livestemc, live stem C, gC/m^2
    # livestemc_storage, live stem C storage, gC/m^2
    # livestemc_xfer, live stem C transfer, gC/m^2

    # livestemn, live stem N, gN/m^2
    # livestemn_storage, live stem N storage, gN/m^2
    # livestemn_xfer, live stem N transfer, gN/m^2

    # pft_ctrunc, pft-level sink for C truncation, gC/m^2
    # pft_ntrunc, pft-level sink for N truncation, gN/m^2

    # plant_calloc, total allocated C flux, gC/m^2/s
    # plant_nalloc, total allocated N flux, gN/m^2/s (missing)

    # plant_ndemand, N flux required to support initial GPP, gN/m^2/s
    # storage_cdemand, C use from the C storage pool, gC/m^2
    
    # totvegc, total vegetation carbon, excluding cpool, 	gC/m^2    
    
    # pco2_240, 10-day mean CO2 partial pressure, Pa (input?)    
    # LAI240_VALUE, 240hr average of leaf area index (input?)
    
    # tempavg_agnpp, Temp. Average AGNPP (input?)
    # tempavg_bgnpp, Temp. Average BGNPP (input?)


fVars = ['FLDS', 'FSDS', 'PRECTmms', 'PSRF', 'RH', 'TBOT', 'WIND']
staVars = ['mean', 'std', 'max', 'min']

sVars = ['abm', # cat
         'ALB_IMPROAD_DIF', 'ALB_IMPROAD_DIR', 'ALB_PERROAD_DIF', 'ALB_PERROAD_DIR', # 2*3*1544*1592
         'ALB_ROOF_DIF', 'ALB_ROOF_DIR', 'ALB_WALL_DIF', 'ALB_WALL_DIR', # 2*3*1544*1592
         'AREA', 'binfl', 
         'CANYON_HWR', # 3*1544*1592
        #  'cft', # 1D
         'CONST_FERTNITRO_CFT', # 2*1544*1592 
         'CONST_GRAZING', 
         'CONST_HARVEST_SH1', 'CONST_HARVEST_SH2', 'CONST_HARVEST_SH3', 'CONST_HARVEST_VH1', 'CONST_HARVEST_VH2', 
         'WTROAD_PERV', 'WTLUNIT_ROOF', # 3*1544*1592
         'CV_IMPROAD', 'CV_ROOF', 'CV_WALL',# 5*3*1544*1592
         'Ds', 'Dsmax', 
         'EF1_BTR', 'EF1_CRP', 'EF1_FDT', 'EF1_FET', 'EF1_GRS', 'EF1_SHR', 
         'EM_IMPROAD', 'EM_PERROAD', 'EM_ROOF', 'EM_WALL', # 3*1544*1592
         'F0', 'FMAX', 
         'gdp',
         # 'GLACIER_REGION', # cat
         # 'GLC_MEC', # 1D
         'HT_ROOF',  # 3*1544*1592
         'LAKEDEPTH', 'LANDFRAC_PFT',
         'LATIXY', 'LONGXY', # 
         'MONTHLY_HEIGHT_BOT', 'MONTHLY_HEIGHT_TOP', 'MONTHLY_LAI', 'MONTHLY_SAI', # 12*17*1544*1592
        #  'mxsoil_color', # -D 
        # 'natpft', # 1D
         'NLEV_IMPROAD',  # 3*1544*1592
         'P3',
         'PCT_CFT', # 2*1544*1592
         'PCT_CROP', 'PCT_NATVEG', 'PCT_GLACIER', 'PCT_GLC_GIC', 'PCT_GLC_ICESHEET', 'PCT_LAKE', 'PCT_WETLAND', 'peatf',
         'PCT_URBAN', # 3*1544*1592   
        #  'PFTDATA_MASK', # bool          
         'PCT_NAT_PFT', # 15*1544*1592
         'PCT_GLC_MEC', 'PCT_GLC_MEC_GIC', 'PCT_GLC_MEC_ICESHEET', # 10*1544*1592         
         'SLOPE', 'SOIL_COLOR', 'STD_ELEV', 'zbedrock',          
         'T_BUILDING_MIN', 'THICK_ROOF', 'THICK_WALL',# 3*1544*1592
         'TK_IMPROAD', 'TK_ROOF', 'TK_WALL',# 5*3*1544*1592
         'TOPO_GLC_MEC', # 10*1544*1592
         'UNREPRESENTED_CFT_LULCC', # 2*1544*1592
         'UNREPRESENTED_PFT_LULCC', # 15*1544*1592
        #  'URBAN_REGION_ID',  # cat
        'Ws',
        'WIND_HGT_CANYON', 'WTROAD_PERV', 'WTLUNIT_ROOF', # 3*1544*1592
         'ORGANIC', 'PCT_CLAY', 'PCT_SAND', # 10*1544*1592
         'ZWT0'
         ]

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
         'plant_calloc', #'plant_nalloc',         
         'plant_ndemand', 'storage_cdemand',         
         'totvegc',
         'pco2_240', 'LAI240_VALUE', 'tempavg_agnpp', 'tempavg_bgnpp'   
         ] # 52

inDir = "/p/largedata/jicg41/chenh/ML_spinUp"
outDir = "/p/largedata/jicg41/chenh/ML_spinUp/samples4ai_PFT_final/"

os.mkdir(outDir)

iniData = Dataset(inDir+os.sep+'initinal_sates'+os.sep+'CLM5_rst_2d_joined.nc')

# PFTs

for p in range(0,16):
    print(p)
    allVarsList = []
    allVarsData = []
    
    refData = iniData['leafc'][p,:,:]
    
    for fvar0 in fVars:
        fData0 = Dataset(inDir+os.sep+'forcings'+os.sep+fvar0+'.nc')
        for sta_var in staVars:
            var_name = fvar0+'_'+sta_var
            data = fData0[sta_var][:]
            data = np.where(np.isnan(refData), np.nan, data)
            allVarsList.append(var_name)
            allVarsData.append(data)
    
    for fvar in fVars:
        for m in range(1,13):
            if m<10:
                tm = '0'+str(m)
            else:
                tm = str(m)
            fData = Dataset(inDir+os.sep+'forcings'+os.sep+fvar+tm+'.nc')
            for sta_var in staVars:
                var_name = fvar+tm+'_'+sta_var
                data = fData[sta_var][:]
                data = np.where(np.isnan(refData), np.nan, data)
                allVarsList.append(var_name)
                allVarsData.append(data)
    
    surfData = Dataset(inDir+os.sep+'surface_files'+os.sep+'surfdata_joined.nc')
    for svar in sVars:
        if svar in ['MONTHLY_HEIGHT_BOT', 'MONTHLY_HEIGHT_TOP']:
            # 12*17*1544*1592
            data = surfData[svar][:,p+1,:,:]                        
            mean = np.mean(data, axis=0)            
            allVarsList.append(svar)            
            allVarsData.append(mean)                    
           
        elif svar in ['MONTHLY_LAI', 'MONTHLY_SAI']:
            # 12*17*1544*1592
            data = surfData[svar][:,p+1,:,:]
                        
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            max = np.amax(data, axis=0)
            min = np.amin(data, axis=0)
            
            allVarsList.append(svar+'_'+'mean')
            allVarsList.append(svar+'_'+'std')
            allVarsList.append(svar+'_'+'max')
            allVarsList.append(svar+'_'+'min')
            
            allVarsData.append(mean)
            allVarsData.append(std)  
            allVarsData.append(max)  
            allVarsData.append(min)
                     
            for pm in range(12):                
                allVarsList.append(svar+str(pm+1))
                allVarsData.append(data[pm,:,:])    
        
        elif svar in ['ALB_IMPROAD_DIF', 'ALB_IMPROAD_DIR', 'ALB_PERROAD_DIF', 'ALB_PERROAD_DIR', 'ALB_ROOF_DIF', 'ALB_ROOF_DIR', 'ALB_WALL_DIF', 'ALB_WALL_DIR']:
            # 2*3*1544*1592
            for ii in range(2):
                for jj in range(3):
                    data = surfData[svar][ii,jj,:,:]
                    data = np.where(np.isnan(refData), np.nan, data)
                    allVarsList.append(svar+str(ii)+'_'+str(jj))
                    allVarsData.append(data)    
        
        elif svar in ['CV_IMPROAD', 'CV_ROOF', 'CV_WALL', 'TK_IMPROAD', 'TK_ROOF', 'TK_WALL']:
            # 5*3*1544*1592
            for iii in range(5):
                for jjj in range(3):
                    data = surfData[svar][iii,jjj,:,:]
                    data = np.where(np.isnan(refData), np.nan, data)
                    allVarsList.append(svar+str(iii)+'_'+str(jjj))
                    allVarsData.append(data)    
        
        elif svar in ['PCT_NAT_PFT', 'UNREPRESENTED_PFT_LULCC']:
            # 15*1544*1592
            for a in range(15):
                data = surfData[svar][a,:,:]
                data = np.where(np.isnan(refData), np.nan, data)
                allVarsList.append(svar+'_'+str(a+1))
                allVarsData.append(data)              
        
        elif svar in ['CANYON_HWR', 'EM_IMPROAD', 'EM_PERROAD', 'EM_ROOF', 'EM_WALL', 'HT_ROOF', 'NLEV_IMPROAD', 'PCT_URBAN', 'T_BUILDING_MIN', 'THICK_ROOF', 'THICK_WALL', 'WIND_HGT_CANYON', 'WTROAD_PERV', 'WTLUNIT_ROOF']:
            # 3*1544*1592
            for b in range(3):
                data = surfData[svar][b,:,:]
                data = np.where(np.isnan(refData), np.nan, data)
                allVarsList.append(svar+'_'+str(b+1))
                allVarsData.append(data)     
                     
        elif svar in ['CONST_FERTNITRO_CFT', 'PCT_CFT', 'UNREPRESENTED_CFT_LULCC']:
            # 2*1544*1592
            for c in range(2):
                data = surfData[svar][c,:,:]
                data = np.where(np.isnan(refData), np.nan, data)
                allVarsList.append(svar+'_'+str(c+1))
                allVarsData.append(data)
        
        elif svar in ['ORGANIC', 'PCT_CLAY', 'PCT_SAND', 'PCT_GLC_MEC', 'PCT_GLC_MEC_GIC', 'PCT_GLC_MEC_ICESHEET', 'TOPO_GLC_MEC']:
            # 10*1544*1592
            for d in range(10):
                data = surfData[svar][d,:,:]
                data = np.where(np.isnan(refData), np.nan, data)
                allVarsList.append(svar+'_'+str(d+1))
                allVarsData.append(data)                
        
        else:
            # 1544*1592
            data = surfData[svar][:]
            data = np.where(np.isnan(refData), np.nan, data)
            allVarsList.append(svar)
            allVarsData.append(data)
    
    for ivar in iVars:
        data = iniData[ivar][p,:,:]
        data = np.where(np.isnan(refData), np.nan, data)
        allVarsList.append(ivar)
        allVarsData.append(data)
        
    
    allVarsData = np.array(allVarsData)
    print(allVarsData.shape)
    
    fAllVarsData = allVarsData.reshape(allVarsData.shape[0], -1)
    print(fAllVarsData.shape)
    
    finalData = pd.DataFrame(fAllVarsData.T, columns = allVarsList).dropna()

    print(finalData.shape)
    
    if p < 9:
        pft = '0'+str(p+1)
    else:
        pft = str(p+1)
            
    finalData.to_csv(outDir+os.sep+'pft'+pft+'.csv',index=False)
    