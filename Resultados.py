#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:48:12 2021

@author: eddyazg
"""

import csv
from collections import defaultdict
from scipy import stats
import numpy as np
from numpy import empty
import matplotlib.pyplot as plt
from matplotlib import pyplot
import math
#import pandas as pd

NumPaq= 1
distan = "0.000050"
lim = 0.000050
distancia = float(distan)
radioTX = 0.0000012
radioRX = 0.000005

pulsotx = np.array([1,0,0,0,0,0,0,0])
pulso = list()
contador=0
t0=0
NumMol=100000
D = 1*10**(-9)
delay = pow(distancia,2)/(D*4)
tw = 5
ts = delay + (tw/2)
BM = 0#Tipo de BM     0 Libre ---- 1 Deriva

for p in range((NumPaq*8)-7):
   
   #################### - Graficar el Movimiento de una molecula - #################### 
   
   columns = defaultdict(list) # each value in each column is appended to a list
   archivo = "example-0-"+str(BM)+"-"+str(p)+"-"+distan+".csv" #Nodo-Tipo-Bit 
   with open(archivo) as f:
      reader = csv.reader(f)
      for row in reader:
          for (i,v) in enumerate(row):
   	       columns[i].append(v)
            
   Num_Col = i
   Bit = list()
   State = list()
   Time = list()
   X = list()
   Y = list ()
   limit = list()
   
   Bit = columns[0]
   State = columns[1]
   Time = columns[2]
   X = empty((Num_Col-3)/2)
   Y = empty((Num_Col-3)/2)
   limit = empty((Num_Col-3)/2)
   
   for i in range(len(Bit)):
      for j in range((Num_Col-3)/2):
         X[j] = columns[2*j+3][i]
         Y[j] = columns[2*j + 4][i]
         limit[j] = lim
      if( i == 1):                     ########## Elige que molecula graficar 
         break
   
   plt.figure(p+1)
   rx = plt.Circle((distancia, 0), radioRX, color='g')
   tx = plt.Circle((0, 0), radioTX, color='black')
   fig, ax = plt.subplots()
   ax.add_patch(rx)
   ax.add_patch(tx)
   plt.plot(X,limit,'r-', linewidth=3.0)
   plt.plot(X,(limit*-1),'r-', linewidth=3.0)
   plt.plot(X,Y)   
   plt.xlabel("Distance [m]",fontsize=21)
   plt.ylabel("Distance [m]",fontsize=21)   
   plt.xticks(fontsize=16)
   plt.yticks(fontsize=16)   
   #plt.title("MOVIMIENTO DE UNA MOLECULA CON DISTANCIA DE: "+str(distan)+" [m]")
   pyplot.savefig("output"+str(p+1)+".png")
   plt.show()
######################################### Fin grafica movimiento de una molecula ###########################

#################### - ISI - ########################## 

columns1 = defaultdict(list) # each value in each column is appended to a list
archivo1 = "RxTime-0-"+str(BM)+"-"+distan+".csv" #Nodo-Tipo
with open(archivo1) as f:
   reader = csv.reader(f)
   for row in reader:
       for (i,v) in enumerate(row):
	       columns1[i].append(v)
           
time = list()
time1 = list()
cont = list()
pulso = list()
TimeBit = np.zeros((NumPaq*8*NumMol*2, 3))
cg = 0

for i in range(NumPaq*8-7):
   time = list()
   for j in range(len(row)):
      if ((columns1[j][i] != '0') & (columns1[j][i] != '')):
         time.append(float(columns1[j][i]))
   time.sort()
   
   ####################################   Graficar tiempo vs Concentracion de cada bit ###########################33
   Tiempo_1Bit_Sup = list() 
   Num_Mol = list()
   contador0 = 1
   for ii in range(len(time)):
       if ii == len(time)-1:
           Tiempo_1Bit_Sup.append(float(time[len(time)-1])*1)
           Num_Mol.append(contador0) 
       elif time[ii] == time[ii+1]:
           contador0 = contador0 + 1
       else:
           Tiempo_1Bit_Sup.append(float(time[ii])*1)
           Num_Mol.append(contador0)
           contador0 = 1

   
   plt.figure(i+1)

   plt.plot(Tiempo_1Bit_Sup, Num_Mol)   
   plt.xlabel("Time [ms]",fontsize=21)
   plt.ylabel("Received pulse amplitude [particles]",fontsize=21)
   plt.xticks(np.linspace(0, 5000, 11, endpoint=True),fontsize=16)
   plt.yticks(np.linspace(0, 380, 11, endpoint=True),fontsize=16)
   pyplot.savefig("FINAL1_Concentracion individual-"+str(i+1)+".png")
   plt.grid()
   plt.show()
   
#################################### Fin de Grafica ################################################################
   
   t2=ts
   t1=0
   if (i>0):
      t0=delay-(tw/2)
   for k in range(1,NumPaq*8 + 1,1):
      if (k>1):
         t0=ts
         t1 = (k-2)*tw
         t2 = ts+(k-1)*tw         
      else:
         if (i==0):
            t0=0
            t1=0
      for l in range (len(time)):
         if ((float(time[l])>(t0+t1)) & (float(time[l])<=t2)):
            TimeBit[cg][0] = float(time[l])+(i)*tw
            TimeBit[cg][1] = (k-1)
            TimeBit[cg][2] = i
            cg = cg+1
            contador = contador + 1
      
      cont.append(contador)
      contador = 0
      
   time1.append(cont)
   cont = list()
   
cnt=0
bitISI = np.zeros(len(time1))
for i in range((NumPaq*8)-1,-1,-1):
   for j in range((NumPaq*8)-cnt):
      bitISI[i] = bitISI[i] + float(time1[i-j][j])
   cnt = cnt+1

TimeBit1 = TimeBit[:,0]  
TimeBit1 = [elemento for elemento in TimeBit1 if elemento !=0] 
TimeBit1.sort()
TimeBitSup = list()
Num_Mol = list()
contador = 1
   
for i in range(len(TimeBit1)):
    if i == len(TimeBit1)-1:
        TimeBitSup.append(TimeBit1[len(TimeBit1)-1])
        Num_Mol.append(contador) 
    elif TimeBit1[i] == TimeBit1[i+1]:
        contador = contador + 1
    else:
        TimeBitSup.append(TimeBit1[i])
        Num_Mol.append(contador)
        contador = 1                 

for i in range(len(TimeBitSup)):
   if (TimeBitSup[i]<=(8*tw)):
      TimeBitSup[i] = TimeBitSup[i]
      Num_Mol[i] = Num_Mol[i]
   else:
      TimeBitSup[i] = 0
      Num_Mol[i] = 0

TimeBitSup = [elemento for elemento in TimeBitSup if elemento !=0] 
Num_Mol = [elemento for elemento in Num_Mol if elemento !=0] 

pulso = list()
t2 = np.arange(0.01, 1.060, 0.001)
ttot = np.arange(0.01, (tw*8)+0.01, 0.01)
cont = 0
for i in range(len(pulsotx)):
   for j in range(len(t2)):
      if (cont< tw*100):
         pulso.append(pulsotx[i]+0)
         cont = cont+1
   cont = 0

linea = np.arange(0,1,0.01)
umbral = np.ones(len(ttot))
umbral = umbral*max(Num_Mol)/2
ceros = np.zeros(len(linea))
fin = np.ones(len(linea))
fin = fin*len(ttot)/100

plt.figure(NumPaq*8+2)

plt.plot(TimeBitSup, Num_Mol)
plt.xlabel("Time [s]",fontsize=21)
plt.ylabel("Measured particle concentration [particles/um2]",fontsize=21)
plt.xticks(np.linspace(0, 9, 10, endpoint=True),fontsize=16)
plt.yticks(np.linspace(0, 24, 13, endpoint=True),fontsize=16)
pyplot.savefig("TRENDEPULSOS.png")
plt.grid()
plt.show()


plt.figure(NumPaq*8+3)

plt.plot(ttot,umbral, 'r',linewidth=3.0)
plt.plot(ceros,linea, 'black',linewidth=5.0)
plt.plot(fin,linea, 'black',linewidth=5.0)
plt.plot(ttot,pulso, 'black',linewidth=5.0)
plt.plot(TimeBitSup, Num_Mol)
plt.xlabel("Time [s]",fontsize=21)
plt.ylabel("Measured particle concentration [particles/um2]",fontsize=21)
plt.xticks(np.linspace(0, 9, 10, endpoint=True),fontsize=16)
plt.yticks(np.linspace(0, 24, 13, endpoint=True),fontsize=16)
pyplot.savefig("TRENDEPULSOS_A.png")
plt.grid()
plt.show()

################################### Fin ISI ##########################################

