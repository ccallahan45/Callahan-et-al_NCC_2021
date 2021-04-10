#!/usr/bin/env python
# coding: utf-8

# # Nino3 tropical longitudinal profile
# #### Christopher Callahan
# #### Christopher.W.Callahan.GR@Dartmouth.edu


# #### Mechanics
# Import dependencies

# In[1]:


import xarray as xr
import numpy as np
import sys
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.io import loadmat
from matplotlib.patches import Polygon
from scipy import stats
import xesmf as xe
import dask

# Model names

# In[2]:


modelnames_fig = ['CCSM3 abrupt 2x','CCSM3 abrupt 4x','CCSM3 abrupt 8x',     'CESM1.0.4 abrupt 2x','CESM1.0.4 abrupt 4x','CESM1.0.4 abrupt 8x', 'CNRM-CM6.1 abrupt4x',     'GFDL-CM3 1pct 2x','GFDL-ESM2M 1pct 2x','GISS-E2-R 1pct 4x',     'GISS-E2-R abrupt 4x','HadCM3L abrupt 2x','HadCM3L abrupt 4x',     'HadCM3L abrupt 6x','HadCM3L abrupt 8x','IPSL-CM5A-LR abrupt 4x',     'MIROC3.2 1pct 2x','MIROC3.2 1pct 4x','MPIESM-1.2 abrupt 2x',     'MPIESM-1.2 abrupt 4x','MPIESM-1.2 abrupt 8x']

modelnames_file = ['CCSM3_abrupt2x','CCSM3_abrupt4x','CCSM3_abrupt8x',     'CESM104_abrupt2x','CESM104_abrupt4x','CESM104_abrupt8x',     'CNRMCM61_abrupt4x','GFDLCM3_1pct2x','GFDLESM2M_1pct2x','GISSE2R_1pct4x',     'GISSE2R_abrupt4x','HadCM3L_abrupt2x','HadCM3L_abrupt4x',     'HadCM3L_abrupt6x','HadCM3L_abrupt8x','IPSLCM5A_abrupt4x',     'MIROC32_1pct2x','MIROC32_1pct4x','MPIESM12_abrupt2x',     'MPIESM12_abrupt4x','MPIESM12_abrupt8x']



#modelnames_file = ['CNRMCM61_abrupt4x','MPIESM12_abrupt2x','MPIESM12_abrupt4x','MPIESM12_abrupt8x']


# Data location

# In[3]:

loc_thetao = "~/" # not using raw data here
loc_out = "../Data/Depth_Profile/"


# Latitude and longitude bounds

# In[4]:


depth_levels = [10,20] #,50,100,200,500,1000]

# Nino3 region

lat_min = -5
lat_max = 5

lon_min = 210
lon_max = 270

# If the lon arrays have -180-180
lon_min_180 = -150
lon_max_180 = -90

# nino4 region
lon_min_nino4 = 160
lon_max_nino4 = 210

lon_min_nino4_180 = 160
lon_max_nino4_180 = -150


# #### Analysis

# In[ ]:


for i in np.arange(0,len(modelnames_file),1): #len(modelnames_file),1):

    model, exp = modelnames_file[i].split("_")
    print(modelnames_file[i])

    fname_exp_start = "thetao_ann_"+modelnames_file[i]
    fname_control_start = "thetao_ann_"+model+"_control"

    fname_exp = [f for f in os.listdir(loc_thetao) if f.startswith(fname_exp_start)][0]
    fname_control = [f for f in os.listdir(loc_thetao) if f.startswith(fname_control_start)][0]

    n1, n2, n3, n4, year_exp_nc = fname_exp.split("_")
    n1, n2, n3, n4, year_control_nc = fname_control.split("_")
    year_exp, nc = year_exp_nc.split(".")
    year_control, nc = year_control_nc.split(".")

    # In this script we're repeatedly doing the calculations for the control, for each experiment
    # That is, this loop for CESM abrupt2x and abrupt4x does the exact same calculation on the
    # CESM control and re-writes the file
    # but I'm too lazy to be more precise about it

    if (model == "MPIESM12"):
        thetao_exp_global = xr.DataArray(xr.open_dataset(loc_thetao+fname_exp,decode_times=False).data_vars["thetao"])
        thetao_control_global = xr.DataArray(xr.open_dataset(loc_thetao+fname_control,decode_times=False).data_vars["thetao"])

    elif ((model == "CCSM3") | (model == "CESM104")):
        #thetao_exp_global = xr.DataArray(xr.open_dataset(loc_thetao+fname_exp, chunks={'z_t': 10}).data_vars["thetao"])
        #thetao_exp_global = xr.open_dataarray(loc_thetao+fname_exp)[0:1000,0:35,:,:]
        #thetao_control_global = xr.open_dataarray(loc_thetao+fname_control)[:,0:35,:,:]
        #thetao_exp_global = xr.open_dataset(loc_thetao+fname_exp, chunks={'z_t': 10}).thetao.persist()


        thetao_exp_global = xr.DataArray(xr.open_dataset(loc_thetao+fname_exp).data_vars["thetao"][0:1000,0:35,:,:])
        thetao_control_global = xr.DataArray(xr.open_dataset(loc_thetao+fname_control).data_vars["thetao"][:,0:35,:,:])
        #print(thetao_exp_global)
        #sys.exit()

    elif ((model == "GFDLCM3") | (model == "GFDLESM2M")):
        thetao_exp_global = xr.DataArray(xr.open_dataset(loc_thetao+fname_exp).data_vars["thetao"][0:1000,0:35,:,:])
        thetao_control_global = xr.DataArray(xr.open_dataset(loc_thetao+fname_control).data_vars["thetao"][0:1000,0:35,:,:])

    elif (model == "GISSE2R"):
        thetao_exp_global = xr.DataArray(xr.open_dataset(loc_thetao+fname_exp).data_vars["thetao"][0:1000,:,:,:])
        thetao_control_global = xr.DataArray(xr.open_dataset(loc_thetao+fname_control).data_vars["thetao"][0:1000,:,:,:])

    elif (model == "MIROC32"):
        thetao_exp_global = xr.DataArray(xr.open_dataset(loc_thetao+fname_exp,decode_times=False).data_vars["thetao"][0:1000,0:35,:,:])
        thetao_control_global = xr.DataArray(xr.open_dataset(loc_thetao+fname_control,decode_times=False).data_vars["thetao"][:,0:35,:,:])

    else:
        thetao_exp_global = xr.DataArray(xr.open_dataset(loc_thetao+fname_exp).data_vars["thetao"])
        thetao_control_global = xr.DataArray(xr.open_dataset(loc_thetao+fname_control).data_vars["thetao"])

    if ((model == "CCSM3") | (model == "CESM104")):
        lat = thetao_exp_global.coords["TLAT"][:,0].values
        lon = thetao_exp_global.coords["TLONG"][0,:].values
        lat2d = thetao_exp_global.coords["TLAT"]
        lon2d = thetao_exp_global.coords["TLONG"]
        depth = thetao_exp_global.coords["z_t"].values/100.
        lat_array_name = "TLAT"
        lon_array_name = "TLONG"

    if ((model == "CNRMCM61") | (model == "IPSLCM5A") | (model == "MPIESM12")):
        lat = thetao_exp_global.coords["lat"][:,0].values
        lon = thetao_exp_global.coords["lon"][0,:].values
        lat2d = thetao_exp_global.coords["lat"]
        lon2d = thetao_exp_global.coords["lon"]
        lat_array_name = "lat"
        lon_array_name = "lon"

        if ((model == "CNRMCM61") | (model == "IPSLCM5A")):
            depth = thetao_exp_global.coords["lev"].values
        else:
            depth = thetao_exp_global.coords["depth"].values

    if ((model == "GFDLCM3") | (model == "GFDLESM2M")):
        lat = thetao_exp_global.coords["yt_ocean"].values
        lon = thetao_exp_global.coords["xt_ocean"].values
        lat2d = None
        lon2d = None
        depth = thetao_exp_global.coords["st_ocean"].values
        lat_array_name = "yt_ocean"
        lon_array_name = "xt_ocean"

    if ((model == "GISSE2R") | (model == "MIROC32")):
        lat = thetao_exp_global.coords["lat"].values
        lon = thetao_exp_global.coords["lon"].values
        lat2d = None
        lon2d = None
        if model == "GISSE2R":
            depth = thetao_exp_global.coords["lev"].values
        else:
            depth = thetao_exp_global.coords["depth"].values
        lat_array_name = "lat"
        lon_array_name = "lon"

    if model == "HadCM3L":
        lat = thetao_exp_global.coords["latitude_2"].values
        lon = thetao_exp_global.coords["longitude_1"].values
        lat2d = None
        lon2d = None
        depth = thetao_exp_global.coords["depth"].values
        lat_array_name = "latitude_2"
        lon_array_name = "longitude_1"


    # regrid to 2.5 degree grid using xesmf
    # https://xesmf.readthedocs.io/en/latest/notebooks/Curvilinear_grid.html
    # https://xesmf.readthedocs.io/en/latest/notebooks/Rectilinear_grid.html
    # note that despite the input grid being 0-360 and the xesmf routine using
    # a -180-180 grid, it's smart and knows how to regrid between the two
    # so the output data will be on the -180-180 lon grid but the data is correct
    # (See Calculate_Depth_Profile_Nino3.ipynb)

    thetao_exp_global_rename = thetao_exp_global.rename({lat_array_name:"lat",lon_array_name:"lon"})

    #print(thetao_exp_global_rename)
    #ds_out = xr.Dataset({'lat': (['lat'], np.arange(-80,52,2)),'lon': (['lon'], np.arange(-180,182, 2)),})

    del(thetao_exp_global)
    grid_out_2d = xe.util.grid_global(2.5,2.5)
    grid_out_1d = xr.Dataset({'lat': (['lat'], grid_out_2d.lat[:,0].values),'lon': (['lon'],grid_out_2d.lon[0,:].values),})

    if lat2d is None:
        grid_out = grid_out_1d
    else:
        grid_out = grid_out_2d

    if ((model == "CNRMCM61") | (model == "IPSLCM5A")):
        regridder = xe.Regridder(thetao_exp_global_rename,grid_out,"bilinear",ignore_degenerate=True)
    else:
        regridder = xe.Regridder(thetao_exp_global_rename,grid_out,"bilinear")
    #regridder = xe.Regridder(thetao_exp_global_rename,grid_out,"bilinear")
    thetao_exp_global_regrid = regridder(thetao_exp_global_rename)
    del(thetao_exp_global_rename)

    thetao_control_global_rename = thetao_control_global.rename({lat_array_name:"lat",lon_array_name:"lon"})
    del(thetao_control_global)
    if ((model == "CNRMCM61") | (model == "IPSLCM5A")):
        regridder = xe.Regridder(thetao_control_global_rename,grid_out,"bilinear",ignore_degenerate=True,reuse_weights=True)
    else:
        regridder = xe.Regridder(thetao_control_global_rename,grid_out,"bilinear",reuse_weights=True)
    thetao_control_global_regrid = regridder(thetao_control_global_rename)
    del(thetao_control_global_rename)

    regridder.clean_weight_file()  # clean-up

    #print(thetao_exp_global_regrid)

    if lat2d is None:
        lat_regrid = thetao_exp_global_regrid.lat.values
        lon_regrid = thetao_exp_global_regrid.lon.values
        lonname = "lon"
    else:
        lat_regrid = thetao_exp_global_regrid.lat.values[:,0]
        lon_regrid = thetao_exp_global_regrid.lon.values[0,:]
        lonname = "x"

    lat_indices = np.where(np.logical_and(lat_regrid >= lat_min, lat_regrid <= lat_max))[0]
    lon_indices = np.where(np.logical_and(lon_regrid >= lon_min_180, lon_regrid <= lon_max_180))[0]

    # convert to celsius
    if np.nanmax(thetao_exp_global_regrid.values) >= 270:
        subtract = 273.15
    else:
        subtract = 0

    depth_profile_exp = thetao_exp_global_regrid[:,:,lat_indices,lon_indices].mean(axis=2) - subtract
    depth_profile_control = thetao_control_global_regrid[:,:,lat_indices,lon_indices].mean(axis=2) - subtract
    #del(thetao_exp_global_regrid)
    #del(thetao_control_global_regrid)

    if ((model == "CESM104") & ((exp == "abrupt4x") | (exp == "abrupt8x"))):
        time_exp = depth_profile_exp.coords["year"]
    else:
        time_exp = depth_profile_exp.coords["time"]
    time_control = depth_profile_control.coords["time"]
    #lon_profile = lon #[lon_indices]


    # now calculate nino4 depth profile
    #lon_min_nino4_180 = 160
    #lon_max_nino4_180 = -150
    lon_indices_nino4_1 = np.where(np.logical_and(lon_regrid[(lon_regrid>=-178.75)&(lon_regrid<=-1.25)] >= -180, lon_regrid[(lon_regrid>=-178.75)&(lon_regrid<=-1.25)] <= -150))[0]
    lon_indices_nino4_2 = np.where(np.logical_and(lon_regrid[(lon_regrid>=1.25)&(lon_regrid<=178.75)] >= 160, lon_regrid[(lon_regrid>=1.25)&(lon_regrid<=178.75)] <= 180))[0]
    thetao_exp_global_regrid_1 = thetao_exp_global_regrid[:,:,:,(lon_regrid>=-178.75)&(lon_regrid<=-1.25)]
    thetao_exp_global_regrid_2 = thetao_exp_global_regrid[:,:,:,(lon_regrid>=1.25)&(lon_regrid<=178.75)]
    thetao_control_global_regrid_1 = thetao_control_global_regrid[:,:,:,(lon_regrid>=-178.75)&(lon_regrid<=-1.25)]
    thetao_control_global_regrid_2 = thetao_control_global_regrid[:,:,:,(lon_regrid>=1.25)&(lon_regrid<=178.75)]

    depth_profile_exp_nino4 = xr.concat([thetao_exp_global_regrid_2[:,:,lat_indices,lon_indices_nino4_2],thetao_exp_global_regrid_1[:,:,lat_indices,lon_indices_nino4_1]],dim=lonname).mean(axis=2) - subtract
    depth_profile_control_nino4 = xr.concat([thetao_control_global_regrid_2[:,:,lat_indices,lon_indices_nino4_2],thetao_control_global_regrid_1[:,:,lat_indices,lon_indices_nino4_1]],dim=lonname).mean(axis=2) - subtract
    #depth_profile_exp_nino4_1 = thetao_exp_global_regrid[:,:,lat_indices,lon_indices_nino4_1].mean(axis=2) - subtract
    #depth_profile_exp_nino4_2 = thetao_exp_global_regrid[:,:,lat_indices,lon_indices_nino4_2].mean(axis=2) - subtract


    # write out
    depth_profile_exp.name = "profile"
    depth_profile_exp.attrs["creation_date"] = str(datetime.datetime.now())
    depth_profile_exp.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
    depth_profile_exp.attrs["data_description"] = "Nino3 depth profile, lat avg, from "+model+" "+exp
    depth_profile_exp.attrs["created_from"] = "Calculate_Depth_Profile_Nino3.py"
    depth_profile_exp.attrs["units"] = "celsius"

    fname_out_exp = loc_out+"nino3_depth_profile_"+model+"_"+exp+".nc"
    depth_profile_exp.to_netcdf(fname_out_exp,mode="w")
    print(fname_out_exp)


    # write out
    depth_profile_control.name = "profile"
    depth_profile_control.attrs["creation_date"] = str(datetime.datetime.now())
    depth_profile_control.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
    depth_profile_control.attrs["data_description"] = "Nino3 depth profile, lat avg, from "+model+" control"
    depth_profile_control.attrs["created_from"] = "Calculate_Depth_Profile_Nino3.py"
    depth_profile_control.attrs["units"] = "celsius"

    fname_out_control = loc_out+"nino3_depth_profile_"+model+"_control.nc"
    depth_profile_control.to_netcdf(fname_out_control,mode="w")
    print(fname_out_control)




    # write out
    depth_profile_exp_nino4.name = "profile"
    depth_profile_exp_nino4.attrs["creation_date"] = str(datetime.datetime.now())
    depth_profile_exp_nino4.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
    depth_profile_exp_nino4.attrs["data_description"] = "Nino4 depth profile, lat avg, from "+model+" "+exp
    depth_profile_exp_nino4.attrs["created_from"] = "Calculate_Depth_Profile_Nino3.py"
    depth_profile_exp_nino4.attrs["units"] = "celsius"

    fname_out_exp = loc_out+"nino4_depth_profile_"+model+"_"+exp+".nc"
    depth_profile_exp_nino4.to_netcdf(fname_out_exp,mode="w")
    print(fname_out_exp)


    # write out
    depth_profile_control_nino4.name = "profile"
    depth_profile_control_nino4.attrs["creation_date"] = str(datetime.datetime.now())
    depth_profile_control_nino4.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
    depth_profile_control_nino4.attrs["data_description"] = "Nino4 depth profile, lat avg, from "+model+" control"
    depth_profile_control_nino4.attrs["created_from"] = "Calculate_Depth_Profile_Nino3.py"
    depth_profile_control_nino4.attrs["units"] = "celsius"

    fname_out_control = loc_out+"nino4_depth_profile_"+model+"_control.nc"
    depth_profile_control_nino4.to_netcdf(fname_out_control,mode="w")
    print(fname_out_control)
