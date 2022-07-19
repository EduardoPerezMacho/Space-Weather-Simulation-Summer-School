# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 18:32:32 2022

@author: edu_p
"""
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


def read_ascii_file(filename, index):
    """This reads an ascii file of omni data"""
    nLines=3
    with open(filename) as f:
        # skip 3 lines doing nothing
        for iLines in range(nLines):
            tmp=f.readline()
        
        # line 4: read in variables line and convert to variables names
        header=f.readline()
        vars=header.split()
        
        data_dic={"time":[],
                  "year":[],
                  "day":[],
                  "hour":[],
                  "minute":[],
                  "symh":[]}
        
        # read in data line, convert to numerical values
        for line in f:
            tmp=line.split()
            
            #create datetime in each line
            time0= dt.datetime(int(tmp[0]),1,1,int(tmp[2]),int(tmp[3]),0)+dt.timedelta(days=int(tmp[1])-1)
            data_dic["time"].append(time0)
            data_dic["year"].append(int(tmp[0]))
            data_dic["day"].append(int(tmp[1]))
            data_dic["hour"].append(int(tmp[2]))
            data_dic["minute"].append(int(tmp[3]))
            data_dic["symh"].append(int(tmp[index]))
        return data_dic


import argparse

def parse_args():
  # Create an argument parser:
  parser = argparse.ArgumentParser(description = \
                                   'Get from file and save a plot')

  # in_scalar: scalar value, type float:
  parser.add_argument('-inp', \
                      help = 'input file', \
                      type = str)
  parser.add_argument('-out', \
                      help = 'output file', \
                      type = str)
  # actually parse the data now:
  args = parser.parse_args()
  return args
# ------------------------------------------------------
# My Main code:
# ------------------------------------------------------
# parse the input arguments:
args = parse_args()
inp = args.inp
out = args.out
print('Args = ', args)
print('Input file = ', inp)
print('Output file = ', out)


file = inp
index=-1
data = read_ascii_file(file, index)
time=np.array(data['time'])
data1=np.array(data['symh'])

fig, ax = plt.subplots()

ax.plot(time, data1, marker='.', c='gray', label='All Events', alpha=0.5)

lp=data1 < -100

ax.plot(time[lp],data1[lp],marker='+', linestyle='', c='orange',label='<-100 nT',alpha=0.6)

ax.set_xlabel('Year of 2013')
ax.grid(True)
ax.legend()
indexmax = np.argmax(data['symh'])
indexmin = np.argmin(data['symh'])
ax.axvline(x=data['time'][indexmax])
ax.axvline(x=data['time'][indexmin])
print('Min Sym-H = ', data['symh'][indexmin])
print('Time of min Sym-H = ', data['time'][indexmin].isoformat())
plt.savefig(out)