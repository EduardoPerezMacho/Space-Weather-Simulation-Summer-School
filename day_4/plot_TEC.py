# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:51:04 2022

@author: edu_p
"""
__author__ = 'Eduardo Macho'
__email__ = 'edu_point@hotmail.com'

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_args():
  """This is a function that parses one input file.
  The file should be in the same directory of command"""  
  # Create an argument parser:
  parser = argparse.ArgumentParser(description = 'Inputfile')
  # in_var: list of 1:
  parser.add_argument('infilename', type = str, nargs='+', \
                      help = 'Input filename')
  # actually parse the data now:
  args = parser.parse_args()
  return args

def plot_tec(dataset, figsize=(12,6)):
    """ Show a plot of Total Electron Content file, in TECU """
    fig, ax = plt.subplots(1, figsize=figsize)
    plt.pcolormesh(dataset[0], dataset[1], dataset[2])
    cb = plt.colorbar()  # add color bar to a plot
    cb.set_label(label=unit_z, size =18) # Increase font size of colorbar label
    cb.ax.tick_params(labelsize=18)  # Set the font size of the color scale
    plt.xlabel(f"Longitude ({unit_x})", fontsize=18)
    plt.ylabel(f"Latitude ({unit_y})", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("Total Electron Content", fontsize=22)
    plt.savefig(f'{file}.png')
    return fig, ax

# parse the input arguments:
args = parse_args()
infilename = args.infilename
print(infilename)

# Define x amd y coordinates
num_of_x = 90
num_of_y = 91
x = np.linspace(0, 360, num_of_x)
y = np.linspace(-90, 90, num_of_y)

for file in infilename:
    # Read the file (get the numpy array of the data)
    data = nc.Dataset(file)
    data1 = data['tec'][:]
    # Define z coordinate
    z = data1
    dataset = (x, y, z)
    # Get units from data
    unit_x = data['lon'].units
    unit_y = data['lat'].units
    unit_z = data['tec'].units
    # Call the function
    plot_tec(dataset)
