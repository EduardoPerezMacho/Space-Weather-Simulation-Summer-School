#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Welcome to Space Weather Simulation Summer School Day 3

Today, we will be working with various file types, doing some simple data 
manipulation and data visualization

We will be using a lot of things that have been covered over the last two days 
with minor vairation.

Goal: Getting comfortable with reading and writing data, doing simple data 
manipulation, and visualizing data.

Task: Fill in the cells with the correct codes

@author: Peng Mun Siew
"""

#%% 
"""
This is a code cell that we can use to partition our data. (Similar to Matlab cell)
We hace the options to run the codes cell by cell by using the "Run current cell" button on the top.
"""
print ("Hello World")

#%%
"""
Writing and reading numpy file
"""
# Importing the required packages
import numpy as np

# Generate a random array of dimension 10 by 5
data_arr = np.random.randn(10,5)
print(data_arr)

# Save the data_arr variable into a .npy file
np.save('test_np_save.npy', data_arr)

#Load data from a .npy file
data_arr_loaded = np.load('test_np_save.npy')

## Verification that the loaded data matches the initial data exactly
print(np.equal(data_arr,data_arr_loaded))
print(data_arr==data_arr_loaded)

#%%
"""
Writing and reading numpy zip archive/file
"""
# Generate a second random array of dimension 8 by 1
data_arr2 = np.random.randn(8,1)
print(data_arr2)

# Save the data_arr and data_arr2 variables into a .npz file
np.savez('test_savez.npz', data_arr, data_arr2)

# Load the numpy zip file\n
npzfile = np.load('test_savez.npz')

# Loaded file is not a numpy array, but is a Npzfile object. You are not able to print the values directly.
print(npzfile)

# To inspect the name of the variables within the npzfile\n
print('Variable names within this file:', sorted(npzfile.files))

# We will then be able to use the variable name as a key to access the data.
print(npzfile['arr_0'])

# Verification that the loaded data matches the initial data exactly
print((data_arr==npzfile['arr_0']).all())
print((data_arr2==npzfile['arr_1']).all())

#%%
"""
Error and exception
"""



#%%
"""
Loading data from Matlab
"""

# Import required packages
import numpy as np
from scipy.io import loadmat

dir_density_Jb2008 = 'C:/Users/edu_p/Desktop/Eduardo/Doutorado/Seminários/2022 Boulder SWSS/Codes/Data1/JB2008/2002_JB2008_density.mat'

# Load Density Data
try:
    loaded_data = loadmat(dir_density_Jb2008)
    print (loaded_data)
except:
    print("File not found. Please check your directory")

# Uses key to extract our data of interest
JB2008_dens = loaded_data['densityData']

# The shape command now works
print(JB2008_dens.shape)


#%%
"""
Data visualization I

Let's visualize the density field for 400 KM at different time.
"""
# Import required packages
import matplotlib.pyplot as plt

# Before we can visualize our density data, we first need to generate the discretization grid of the density data in 3D space. We will be using np.linspace to create evenly sapce data between the limits.

localSolarTimes_JB2008 = np.linspace(0,24,24)
latitudes_JB2008 = np.linspace(-87.5,87.5,20)
altitudes_JB2008 = np.linspace(100,800,36)
nofAlt_JB2008 = altitudes_JB2008.shape[0]
nofLst_JB2008 = localSolarTimes_JB2008.shape[0]
nofLat_JB2008 = latitudes_JB2008.shape[0]

# We can also impose additional constratints such as forcing the values to be integers.
time_array_JB2008 = np.linspace(0,8760,20, dtype = int)

# For the dataset that we will be working with today, you will need to reshape them to be lst x lat x altitude
JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008,8760), order='F') # Fortra-like index order

# Look for data that correspond to an altitude of 400 KM
alt = 400
hi = np.where(altitudes_JB2008==alt)

# Create a canvas to plot our data on. Here we are using a subplot with 5 spaces for the plots.
fig, axs = plt.subplots(5, figsize=(15, 10*2), sharex=True)

for ik in range (5):
    cs = axs[ik].contourf(localSolarTimes_JB2008, latitudes_JB2008, JB2008_dens_reshaped[:,:,hi,time_array_JB2008[ik]].squeeze().T)
    axs[ik].set_title('JB2008 density at 400 km, t = {} hrs'.format(time_array_JB2008[ik]), fontsize=18)
    axs[ik].set_ylabel("Latitudes", fontsize=18)
    axs[ik].tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(cs,ax=axs[ik])
    cbar.ax.set_ylabel('Density')

axs[ik].set_xlabel("Local Solar Time", fontsize=18)  
#%%
# Only 1 plot:
fig, axs = plt.subplots(1, figsize=(15, 4), sharex=True)
ik = 0
cs = axs.contourf(localSolarTimes_JB2008, latitudes_JB2008, JB2008_dens_reshaped[:,:,hi,time_array_JB2008[ik]].squeeze().T)
axs.set_title('JB2008 density at 400 km, t = {} hrs'.format(time_array_JB2008[ik]), fontsize=18)
axs.set_ylabel("Latitudes", fontsize=18)
axs.tick_params(axis = 'both', which = 'major', labelsize = 16)

cbar = fig.colorbar(cs,ax=axs)
cbar.ax.set_ylabel('Density')

axs.set_xlabel("Local Solar Time", fontsize=18)

#%%
# 20 plots:

time_array_JB2008 = np.linspace(0,8759,20, dtype = int)
fig, axs = plt.subplots(20, figsize=(15, 40*2), sharex=True)

for ik in range (20):
    cs = axs[ik].contourf(localSolarTimes_JB2008, latitudes_JB2008, JB2008_dens_reshaped[:,:,hi,time_array_JB2008[ik]].squeeze().T)
    axs[ik].set_title('JB2008 density at 400 km, t = {} hrs'.format(time_array_JB2008[ik]), fontsize=18)
    axs[ik].set_ylabel("Latitudes", fontsize=18)
    axs[ik].tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(cs,ax=axs[ik])
    cbar.ax.set_ylabel('Density')

axs[ik].set_xlabel("Local Solar Time", fontsize=18)  

#%%
"""
Assignment 1

Can you plot the mean density for each altitude at February 1st, 2002?
"""

# First identidy the time index that corresponds to  February 1st, 2002. Note the data is generated at an hourly interval from 00:00 January 1st, 2002
time_index = 31*24
dens_data_feb1 = JB2008_dens_reshaped[:,:,:,time_index]

# Generate a random array of dimension 10 by 5
data_arr = np.random.randn(2,5)
print(data_arr)

print('The mean of all elements is: ',np.mean(data_arr))
print('The mean along the 0 axis is: ',np.mean(data_arr, axis = 0))

mean_density = [(np.mean(dens_data_feb1[:,:,altitude])) for altitude in range(len(altitudes_JB2008))]

print(mean_density)

plt.subplots(1, figsize=(10,6))
plt.semilogy(altitudes_JB2008, mean_density)
plt.grid()
plt.xlabel('Altitude', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.title('Mean Density vs Altitude', fontsize=18)

#%%
"""
Data Visualization II

Now, let's us work with density data from TIE-GCM instead, and plot the density field at 310km
"""
# Import required packages
import h5py
loaded_data = h5py.File('C:/Users/edu_p/Desktop/Eduardo/Doutorado/Seminários/2022 Boulder SWSS/Codes/Data2/TIEGCM/2002_TIEGCM_density.mat')
# This is a HDF5 dataset object, some similarity with a dictionary
print('Key within database:',list(loaded_data.keys()))

tiegcm_dens = (10**np.array(loaded_data['density'])*1000).T # convert from g/cm3 to kg/m3
altitudes_tiegcm = np.array(loaded_data['altitudes']).flatten()
latitudes_tiegcm = np.array(loaded_data['latitudes']).flatten()
localSolarTimes_tiegcm = np.array(loaded_data['localSolarTimes']).flatten()
nofAlt_tiegcm = altitudes_tiegcm.shape[0]
nofLst_tiegcm = localSolarTimes_tiegcm.shape[0]
nofLat_tiegcm = latitudes_tiegcm.shape[0]
time_array_tiegcm = np.linspace(0,8759,20, dtype = int)

# Each data correspond to the density at a point in 3D space. 
# We can recover the density field by reshaping the array.
# For the dataset that we will be working with today, you will need to reshape them to be lst x lat x altitude
tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,nofAlt_tiegcm,8760), order='F')
#%%
#Plotting:
# Look for data that correspond to an altitude of 400 KM
alt = 310
hi = np.where(altitudes_tiegcm==alt)

# Create a canvas to plot our data on. Here we are using a subplot with 5 spaces for the plots.
fig, axs = plt.subplots(5, figsize=(15, 10*2), sharex=True)

for ik in range (5):
    cs = axs[ik].contourf(localSolarTimes_tiegcm, latitudes_tiegcm, tiegcm_dens_reshaped[:,:,hi,time_array_tiegcm[ik]].squeeze().T)
    axs[ik].set_title('tiegcm density at 310 km, t = {} hrs'.format(time_array_tiegcm[ik]), fontsize=18)
    axs[ik].set_ylabel("Latitudes", fontsize=18)
    axs[ik].tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(cs,ax=axs[ik])
    cbar.ax.set_ylabel('Density')

axs[ik].set_xlabel("Local Solar Time", fontsize=18)
#%%
# First identidy the time index that corresponds to  February 1st, 2002. Note the data is generated at an hourly interval from 00:00 January 1st, 2002
time_index = 31*24
dens_data_feb1 = tiegcm_dens_reshaped[:,:,:,time_index]
dens_data_feb1b = JB2008_dens_reshaped[:,:,:,time_index]


# Mean Density vs Altitude:
mean_density = [(np.mean(dens_data_feb1[:,:,altitude])) for altitude in range(len(altitudes_tiegcm))]
mean_densityb = [(np.mean(dens_data_feb1b[:,:,altitude])) for altitude in range(len(altitudes_JB2008))]

print(mean_density)

plt.subplots(1, figsize=(10,6))
plt.semilogy(altitudes_tiegcm, mean_density, label='TIE-GCM', color='g')
plt.semilogy(altitudes_JB2008, mean_densityb, '-', label='JB2008', color ='r')
plt.grid()
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlabel('Altitude', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.title('Mean Density vs Altitude', fontsize=18)

#%%
"""
Data Interpolation (1D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy import interpolate

# Let's first create some data for interpolation
x = np.arange(0, 10)
y = np.exp(-x/3.0)
# Generate 1D interpolant function
interp_func_1D = interpolate.interp1d(x, y)
# Let's select some new points
xnew = np.arange(0, 9, 0.1)
# Use interpolation function returned by interp1d
ynew = interp_func_1D(xnew)   # use interpolation function returned by `interp1d`

plt.subplots(1, figsize=(10, 6))
plt.plot(x, y, 'o', xnew, ynew, '*',linewidth = 2)
plt.legend(['Inital Points','Interpolated Points'], fontsize = 16)
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.title('1D interpolation', fontsize=18)
plt.grid()
plt.tick_params(axis = 'both', which = 'major', labelsize = 16)

#%%
"""
Data Interpolation (3D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy.interpolate import RegularGridInterpolator

# First create a set of sample data that we will be using 3D interpolation on
def function_1(x, y, z):
    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)

sample_data = function_1(xg, yg, zg)

# Generate Interpolant (interpolating function)
interpolated_function_1 = RegularGridInterpolator((x, y, z), sample_data)

# Say we are interested in the points [[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]]
pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
print('Using interpolation method:', interpolated_function_1(pts))
print('From true function:', function_1(pts[:,0],pts[:,1],pts[:,2]))

#%%
"""
Saving mat file

Now, let's us look at how to we can save our data into a mat file
"""
# Import required packages
from scipy.io import savemat

a = np.arange(20)
mdic = {"a": a, "label": "experiment", "v":1} # Using dictionary to store multiple variables
# multiple variables
savemat("matlab_matrix.mat", mdic)

#%%
"""
Assignment 2 (a)

The two data that we have been working on today have different discretization grid.
Use 3D interpolation to evaluate the TIE-GCM density field at 400 KM on February 1st, 2002, with the discretized grid used for the JB2008 ((nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008).

localSolarTimes_JB2008 = np.linspace(0,24,24)
latitudes_JB2008 = np.linspace(-87.5,87.5,20)
altitudes_JB2008 = np.linspace(100,800,36)
"""

import argparse

def parse_args():
  # Create an argument parser:
  parser = argparse.ArgumentParser(description = \
                                   'Select altitude and time')

  # in_scalar: scalar value, type float:
  parser.add_argument('-alt', \
                      help = 'altitude', \
                      type = int)
  parser.add_argument('-doy', \
                      help = 'time', \
                      type = int)
  # actually parse the data now:
  args = parser.parse_args()
  return args
# ------------------------------------------------------
# My Main code:
# ------------------------------------------------------
# parse the input arguments:
args = parse_args()
alt = args.alt
doy = args.doy
print('Args = ', args)
print('Altitude = ', alt)
print('Doy = ', doy)

time_index = doy*24
tiegcm_function = RegularGridInterpolator((localSolarTimes_tiegcm, latitudes_tiegcm, altitudes_tiegcm), tiegcm_dens_reshaped[:,:,:,time_index], bounds_error=False, fill_value=None)
# bound_error = False - can ask for points outside of the initial range
# fill_value = None - extrapolate based on data for those outside the range

print('TIE-GCM density at lst=20, lat=12, alt=400km:', tiegcm_function((20,12,400)))
tiegcm_jb2008_grid = np.zeros((24,20))
for lst_i in range(24):
    for lat_i in range(20):
        tiegcm_jb2008_grid[lst_i, lat_i] = tiegcm_function((localSolarTimes_JB2008[lst_i],latitudes_JB2008[lat_i],400))

fig, axs = plt.subplots(2, figsize=(15, 10), sharex=True)

cs = axs[0].contourf(localSolarTimes_JB2008, latitudes_JB2008, tiegcm_jb2008_grid.T)
axs[0].set_title('TIE-GCM density at 400 km, t = {} hrs'.format(time_index), fontsize=18)
axs[0].set_ylabel("Latitudes", fontsize=18)
axs[0].tick_params(axis = 'both', which = 'major', labelsize = 16)

cbar = fig.colorbar(cs,ax=axs[0])
cbar.ax.set_ylabel('Density')

axs[0].set_xlabel("Local Solar Time", fontsize=18)
#alt=400
hi = np.where(altitudes_JB2008==alt)

cs = axs[1].contourf(localSolarTimes_JB2008, latitudes_JB2008, JB2008_dens_reshaped[:,:,hi,time_index].squeeze().T)
axs[1].set_title('JB2008 density at 400 km, t = {} hrs'.format(time_index), fontsize=18)
axs[1].set_ylabel("Latitudes", fontsize=18)
axs[1].tick_params(axis = 'both', which = 'major', labelsize = 16)

cbar = fig.colorbar(cs,ax=axs[1])
cbar.ax.set_ylabel('Density')

axs[1].set_xlabel("Local Solar Time", fontsize=18)
#plt.savefig('Assignment')

#%%
"""
Assignment 2 (b)

Now, let's find the difference between both density models and plot out this difference in a contour plot.
"""

#%%
"""
Assignment 2 (c)

In the scientific field, it is sometime more useful to plot the differences in terms of mean absolute percentage difference/error (MAPE). Let's plot the MAPE for this scenario.
"""





