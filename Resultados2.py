#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:33:09 2020

@author: jkpl - eddyazg7
"""
import csv
from collections import defaultdict
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import statistics as sta
import numpy as np, scipy.stats as st
import seaborn as sns
from matplotlib import pyplot
import math


m_mol = list()
m_delay = list()
m_pulse = list()
i_mol = list()
i_delay = list()
i_pulse = list()

NumPaq= 1

dist = ['0.000002','0.000004','0.000006','0.000008','0.000010','0.000012','0.000014',
        '0.000016','0.000018','0.000020','0.000022','0.000024','0.000026']

BM = 0 # Tipo de BM     0 Libre ---- 1 Deriva
NumMol=10000
D = 1*10**(-9)
tw = 1050
    
for d in range (len(dist)):
   
    columns = defaultdict(list) # each value in each column is appended to a list
    #archivo = "RxTime-0-0-"+str(dist[d])+".csv" #Nodo-Tipo-Distancia 
    archivo = "RxTime-0-"+str(BM)+"-"+dist[d]+".csv" #Nodo-Tipo-Distancia 
    with open(archivo) as f:
       reader = csv.reader(f)
       for row in reader:
           for (i,v) in enumerate(row):
   	        columns[i].append(v)
    
    max_delay = list()
    max_mol = list()
    max_pulse = list()
    time = list()
    time1 = list()
    cont = list()
    contador=0
    t0=0
    
    delay = pow(float(dist[d]),2)/(D*4)
    ts = delay + (tw/2)
    TimeBit = np.zeros((NumPaq*8*NumMol, 3))
    cg = 0
   #tp = list()

    for i in range(NumPaq*8):
       time = list()
       for j in range(len(row)):
          if ((columns[j][i] != '0') & (columns[j][i] != '')):
             time.append(columns[j][i])
       time.sort()
            
       if (len(time) != 0):
           Tiempo_1Bit_Sup = list() 
           Num_Mol = list()
           contador = 1
           for ii in range(len(time)):
               if ii == len(time)-1:
                   Tiempo_1Bit_Sup.append(time[len(time)-1])
                   Num_Mol.append(contador) 
               elif time[ii] == time[ii+1]:
                   contador = contador + 1
               else:
                   Tiempo_1Bit_Sup.append(time[ii])
                   Num_Mol.append(contador)
                   contador = 1                 
                               
           aux = max(Num_Mol)
           max_mol.append(aux)
           aux = Num_Mol.index(aux)
           max_delay.append(float(Tiempo_1Bit_Sup[aux]))
           
           aux = max(Num_Mol)
           aux = aux/2
               
           bol = True
           for lk in range(len(Num_Mol)):
               if (Num_Mol[lk] >= aux) & (bol==True) :
                   p1 = Tiempo_1Bit_Sup[lk]
                   bol=False
                   
               if (Num_Mol[lk] < aux) & (bol==False) :
                   p2 = Tiempo_1Bit_Sup[lk]
                   break
           p1=Tiempo_1Bit_Sup[0]
           max_pulse.append(float(p2)-float(p1))
       
    i_mol.append(max_mol)
    i_delay.append(max_delay)
    i_pulse.append(max_pulse)
    m_mol.append(sta.mean(max_mol))
    m_delay.append(sta.mean(max_delay))
    m_pulse.append(sta.mean(max_pulse))


interval_mol = np.zeros((5,len(dist)))
for i in range(len(i_mol)):
    for j in range(5):
        interval_mol[j][i] = i_mol[i][j]

interval_delay = np.zeros((5,len(dist)))
for i in range(len(i_delay)):
    for j in range(5):
        interval_delay[j][i] = i_delay[i][j]    

interval_pulse = np.zeros((5,len(dist)))
for i in range(len(i_pulse)):
    for j in range(5):
        interval_pulse[j][i] = i_pulse[i][j] 



plt.figure(1)
plt.plot(dist,m_mol,'r')
sns.tsplot([interval_mol[0], interval_mol[1], interval_mol[2], interval_mol[3], interval_mol[4]] , err_style="ci_bars", interpolate=False)
plt.xlabel("Transmission distance [m]",fontsize=21)
plt.ylabel("Pulse amplitude [molecules]",fontsize=21)   
plt.xticks(fontsize=15) 
plt.yticks(np.linspace(0, 450, 10, endpoint=True),fontsize=15)
plt.grid()  
pyplot.savefig("amplitud"+".png")
plt.show()


plt.figure(2)
plt.plot(dist,m_delay,'r')
sns.tsplot([interval_delay[0], interval_delay[1], interval_delay[2], interval_delay[3], interval_delay[4]] , err_style="ci_bars",interpolate=False)
plt.xlabel("Transmission distance [m]",fontsize=21)
plt.ylabel("Pulse delay [s]",fontsize=21) 
plt.xticks(fontsize=15) 
plt.yticks(fontsize=15)
plt.grid()
pyplot.savefig("delay"+".png")
plt.show()


plt.figure(3)
plt.plot(dist,m_pulse,'r')
sns.tsplot([interval_pulse[0], interval_pulse[1], interval_pulse[2], interval_pulse[3], interval_pulse[4]] , err_style="ci_bars",interpolate=False)
plt.xlabel("Transmission distance [m]",fontsize=21)
plt.ylabel("Pulse width [s]",fontsize=21)
plt.xticks(fontsize=15) 
plt.yticks(fontsize=15)
plt.grid() 
pyplot.savefig("pulse"+".png")
plt.show()



