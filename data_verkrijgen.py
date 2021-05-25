"""
Created on Wed Sep 16 14:32:35 2020
@author: joche
"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
np.random.seed(24)
import time
begin = time.time()

class StatPhys:
    """Een model voor spin."""
    
    def __init__(self,begin_state,KbT):
        self.M    = begin_state #M is de matrix waarin de spin wordt bijgehouden duur +1 of -1
        self.KbT  = KbT 
        self.xlen = np.shape(self.M)[0] 
        self.ylen = np.shape(self.M)[1]
        self.GemiddeldeEnergieBerekenen()
        self.N    = self.xlen * self.ylen
    
    def GemiddeldeEnergieBerekenen(self):
        """Deze functie berekent de gemiddelde energie per spin."""
        Etot = 0
        for i in range(self.xlen):
            for j in range(self.ylen):
                Etot += self.EnergieDeeltje(i,j)
        self.Egem = Etot/(self.xlen*self.ylen)
        
    def EnergieDeeltje(self, i, j):
        """Manier van energie berekenen, op de von Neumann manier."""
        
        """We gebruiken de modulaire functie om ervoor te zorgen dat de randen aan de andere kant van het veld doorlopen."""
        ParticleEnergy  = 0
        for k in [-1, 1]:
            ParticleEnergy  += -J*self.M[(i +k) % self.ylen][j] * self.M[i][j]
        for k in [-1, 1]:                  
            ParticleEnergy  += -J*self.M[i][(j + k) % self.xlen] * self.M[i][j]
        return ParticleEnergy
    
    
    def Lijst_maken(self, hoeveelheid):
        """We maken hier een lijst, met alleen toestanden van het systeem voor een x aantal iteraties."""
        lijst = [self.M for i in range(hoeveelheid)]              #Een lijst met de juiste dimensies zodat we geen append hoeven te gebruiken dit scheelt in complexiteit.
        # self.lijst_Turn = [[0,0] for i in range(hoeveelheid)]
        self.lijst_magnetization = [np.sum(self.M)/self.N for i in range(hoeveelheid)]
        self.lijst_energie = [self.Egem for i in range(hoeveelheid)]
        
        for i in enumerate(lijst[:-1]):  
            self.Iteratie(i[0])
            lijst[i[0]+1] = self.M
        return lijst
    
    def Iteratie(self, n):
        """Hier creeëren we de volgende stap."""
        i = np.random.randint(0,self.xlen)
        j = np.random.randint(0,self.ylen) 
        
        #Energie berekenen
        DeltaE = -2*self.EnergieDeeltje(i,j)  #j = 1
        DeeltjeIsGeflipt = False
        if DeltaE <= 0:
            self.M[i][j] = -1 * self.M[i][j] 
            DeeltjeIsGeflipt =True
        else:
            # self.lijst_T[n+1][0] += 1
            p = np.exp(-DeltaE/self.KbT)
            if p>= np.random.uniform(0,1):
                # self.lijst_Turn[n+1][1] += 1
                self.M[i][j] = -1 * self.M[i][j]
                DeeltjeIsGeflipt = True
            else:
                DeeltjeIsGeflipt = False
        
        '''De volgende energie en magnetisatie uitrekenen'''
        if DeeltjeIsGeflipt:
            self.lijst_energie[n+1] = self.lijst_energie[n] + 2*(DeltaE/self.N)
            self.lijst_magnetization[n+1] = self.lijst_magnetization[n]+2*self.M[i][j]
        else:
            self.lijst_energie[n+1] = self.lijst_energie[n]
            self.lijst_magnetization[n+1] = self.lijst_magnetization[n]
        
   
    def laatste_10waardes(self, hoeveelheid = 10**4):
        lijst = self.Lijst_maken(hoeveelheid)
        "nu willen we de laatste en de 10 daarvoor met spacing 100."
        final_lijst = [[self.KbT,lijst[-1]]]
        for i in range(1,10):
            final_lijst.append([self.KbT,lijst[-steps*i]])
        return final_lijst
         
      
Tk = 2.27
J = 1
n = 100 #hoeveel verschillende temp
m = 5   #hoeveel random states per temp
steps = 100 # hoeveel stappen tussen de laatste states van een grid
Temperatuur = np.linspace(0.001,2*Tk, n) 
"alle temperaturen waarvoor we data gaan verzamelen."

final = []

for temp in enumerate(Temperatuur):
    print(temp[0])
    t = 0 
    sub_lijst = []
    while t<m:
        t += 1
        State = np.random.choice([-1,1],  size = [20,20])
        sim = StatPhys(State,KbT=temp[1])
        sub_lijst.append(sim.laatste_10waardes())
    final.append(np.concatenate(sub_lijst))

print(Temperatuur)

with open('random_data.pkl','wb') as file:
    pickle.dump(file, final)