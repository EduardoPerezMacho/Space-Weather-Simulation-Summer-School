#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt



def read_ascii_file(file,index):

    with open(file) as f:
        year = []
        day = []
        hour = []
        minute = []
        time = []
        data = []
        for line in f:
            tmp = line.split()
            year.append(int(tmp[0]))
            day.append(int(tmp[1]))
            hour.append(int(tmp[2]))
            minute.append(int(tmp[3]))
            data.append(float(tmp[index]))

            #create datetime
            time0 = dt.datetime(int(tmp[0]),1,1,int(tmp[2]),int(tmp[3]),0)\
                    + dt.timedelta(days=int(tmp[1])-1)
            time.append(time0)

    time = np.array(time)
    data = np.array(data)
    year = np.array(year)
    day = np.array(day)
    hour = np.array(hour)
    minute = np.array(minute)

    return time,year,day,hour,minute,data

if __name__ == "__main__":

    file = '../day_2/omni_min_case_2013b.lst'
    index = -1
    time,year,day,hour,minute,data = read_ascii_file(file,index)




    fig,ax = plt.subplots()
    ax.plot(time,data,marker='.',
            c='gray',label='All Events',alpha=0.5)

    lp = data <-100
    ax.plot(time[lp],data[lp],c='tab:orange',
            marker='+',
            linestyle='',
            label='<-100 nT',
            alpha=0.6)
    ax.set_xlabel('Year of 2013')
    ax.set_ylabel('SYMH (nT)')
    ax.grid(True)
    ax.legend()

    plt.show()