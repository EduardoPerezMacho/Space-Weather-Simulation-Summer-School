# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:35:20 2022

@author: edu_p
"""
#%%
# FIRST TASK
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

sme_2013 = np.genfromtxt('sme_2013.txt', skip_header=105, encoding='UTF-8')
print(sme_2013)
print(sme_2013.shape)
#num_lines = 60*24*7 # Select the first week
start_line = 0
end_line = 60*24*7
data_dic={"time":[], "sml":[]}

for line in range(start_line, end_line):
    time=dt.datetime(int(sme_2013[line, 0]),int(sme_2013[line, 1]),int(sme_2013[line, 2]),int(sme_2013[line, 3]),int(sme_2013[line, 4]))
    data_dic["time"].append(time)
    data_dic["sml"].append(int(sme_2013[line, 7]))    
        
#print(data_dic)

time=np.array(data_dic["time"])
sml=np.array(data_dic["sml"])

print(time)
print(sml)

plt.plot(time, sml)
#%%
onset_time=0
onset_sml=0
for each_time in range(len(time)-30):
    cond=[False]*4
    cond[0] = sml[each_time+1]-sml[each_time]<-15
    cond[1] = sml[each_time+2]-sml[each_time]<-30
    cond[2] = sml[each_time+3]-sml[each_time]<-45
    cond[3] = sum([sml[each_time+i] for i in range(4,30)])/26-sml[each_time]<-100
    if all(cond)==True:
        onset_time=time[each_time]
        onset_sml=sml[each_time]
        break

print(cond)
print(all(cond))
print(onset_time)
print(onset_sml)

#%%
time_30=[]
sml_30=[]
for time_window in range(each_time, each_time+30):
    time_30.append(time[time_window])
    sml_30.append(sml[time_window])
print(f'The first onset of 2013 is {onset_time}')
print(f'The min SML value in this period is: {min(sml_30)} nT')
plt.plot(time_30, sml_30)
plt.xlabel('Time')
plt.ylabel('SML (nT)')
plt.title('First Storm of 2013')

#%%
# SECOND TASK
start_line = 0
end_line = 60*24*31 # minutes * hours * days
data_dic={"time":[], "sml":[]}

for line in range(start_line, end_line): # 0=year, 1=month, 2=day, 3=hour, 4=minute, 7=SML
    time=dt.datetime(int(sme_2013[line, 0]),int(sme_2013[line, 1]),int(sme_2013[line, 2]),int(sme_2013[line, 3]),int(sme_2013[line, 4]))
    data_dic["time"].append(time)
    data_dic["sml"].append(int(sme_2013[line, 7]))    

time=np.array(data_dic["time"])
sml=np.array(data_dic["sml"])

print(time)
print(sml)

plt.plot(time, sml)

#%%
onset_time=[]
onset_sml=[]

it = iter(list(range(60*24*31-30))) # skip last 30 min window
for each_time in it:
    cond=[False]*4
    cond[0] = sml[each_time+1]-sml[each_time]<-15
    cond[1] = sml[each_time+2]-sml[each_time]<-30
    cond[2] = sml[each_time+3]-sml[each_time]<-45
    cond[3] = sum([sml[each_time+i] for i in range(4,30)])/26-sml[each_time]<-100
    if all(cond)==True:
        onset_time.append(time[each_time])
        onset_sml.append(sml[each_time])
        for num_of_next in range(30):
            next(it)

print(cond)
print(all(cond))
print(f'List of onset times: {onset_time}')
print(f'List of onset values: {onset_sml}')
print(f'Number of events: {len(onset_time)}')
#%%
# THIRD TASK
start_line = 0
end_line = 60*24*365 # minutes * hours * days
data_dic={"time":[], "sml":[]}

for line in range(start_line, end_line): # 0=year, 1=month, 2=day, 3=hour, 4=minute, 7=SML
    time=dt.datetime(int(sme_2013[line, 0]),int(sme_2013[line, 1]),int(sme_2013[line, 2]),int(sme_2013[line, 3]),int(sme_2013[line, 4]))
    data_dic["time"].append(time)
    data_dic["sml"].append(int(sme_2013[line, 7]))    

time=np.array(data_dic["time"])
sml=np.array(data_dic["sml"])

print(time)
print(sml)

plt.plot(time, sml)

#%%
onset_time=[]
onset_sml=[]
min_sml=[]

it = iter(list(range(60*24*365-30))) # skip last 30 min window
for each_time in it:
    cond=[False]*4
    cond[0] = sml[each_time+1]-sml[each_time]<-15
    cond[1] = sml[each_time+2]-sml[each_time]<-30
    cond[2] = sml[each_time+3]-sml[each_time]<-45
    cond[3] = sum([sml[each_time+i] for i in range(4,30)])/26-sml[each_time]<-100
    if all(cond)==True:
        time_30=[]
        sml_30=[]
        for time_window in range(each_time, each_time+30):
            time_30.append(time[time_window])
            sml_30.append(sml[time_window])
        min_sml.append(min(sml_30))
        onset_time.append(time[each_time])
        onset_sml.append(sml[each_time])
        for num_of_next in range(30):
            next(it)

print(cond)
print(all(cond))
print(f'List of onset times: {onset_time}')
print(f'List of onset values: {onset_sml}')
print(f'Number of events: {len(onset_time)}')
print(f'Distribution of minimum values: {min_sml}')

#%%
plt.plot(min_sml)




