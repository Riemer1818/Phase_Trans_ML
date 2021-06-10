import numpy as np
import datetime
import matplotlib.pyplot as plt
import numpy as np
#import pandas_datareader as pdr
import timeit
import pickle
import os
import sys


def recover_graph(verhoudingen,beginwaarde):
    new_data = np.zeros(len(verhoudingen))
    new_data[0] = beginwaarde
    for i in range(len(verhoudingen)-1):
        new_data[i+1] = new_data[i]* verhoudingen[i+1]    
    return new_data    
        

def sigmoid(X):
    """Sigmoid Function."""
    return  1/(1+np.exp(-X))


def diff_sigmoid(X):
    """Afgeleiden Sigmoid Function."""
    return (X)*(1.0-(X))



def ReLU(x, factor = 0.15):
    """Relu Function."""
    y1 = ((x > 0) * x)                                            
    return y1


def ReLU_leaky(x, factor = 0.15):
    """Relu Function."""
    y1 = ((x > 0) * x)                                            
    y2 = ((x <= 0) * x * factor)                                         
    leaky_way = y1 + y2  
    return leaky_way


def diff_ReLU(x):
    y1 = ((x > 0) * 1)                                                                                 
    return y1


def diff_ReLU_leaky(x,factor = 0.15):
    y1 = ((x > 0) * 1)                                            
    y2 = ((x <= 0) * factor)                                         
    leaky_way = y1 + y2
    return leaky_way


def cap(lijst, maxiwaarde):
    lijst= np.array(lijst)
    x = lijst > maxiwaarde
    y = lijst - lijst*x + maxiwaarde*x
    x = lijst < -maxiwaarde
    y = y - y*x - maxiwaarde*x
    return y


def softmax(output):
    totaal = sum(np.e ** output)
    return np.e ** output/totaal

            
def learning_rate_function(X,grens = 0.1): #  kies je factor je eerste begin foutmarge, factor/2 ^ 2 is je sig
    """wil proberen de eerste fout op 4sigma te zetten dat lijkt logisch omdat dat nog redelijk fout is dan dan zit
    bij de minimale fout .
    kies je sigma dus eerstfout /4"""
    sigma= 3
    learning_rate = -1* np.e ** -((X/sigma) ** 2) - 2  #chech in geogabra mooie functie die -4 is laagste learning rate
    if -grens < X < grens:
        return -5
    else:
        return learning_rate


def potentiale(x, grens = 1.5, minimum = -10, verhouding = 50):
    if x < grens and x > -grens:
        return -10
    else:
        return -1/(np.abs(x)**2/verhouding) - 5


def history_learningrate(lijst_foutmarge, new_fout,stimulans = 0.3, lengte_geschiedenis= 30):
    
    y = 0
    if len(lijst_foutmarge)< lengte_geschiedenis+1:
        return 0
    else:
        for i in range(1,lengte_geschiedenis+1):
            y += lijst_foutmarge[-i]

        y = y/lengte_geschiedenis #gemiddelde fout

        # print('fout', np.abs(new_fout-y),  0.20 * y)
        if np.abs(new_fout-y) < 0.04 * np.abs(y): #tienprocent foutmarge
            # print('history')
            return stimulans
        else:
            return 0

def unpickle_dir(directory):
    data = []
    number_of_data = 0 
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            f = os.path.join(directory, filename)
            with open(f, "rb") as file:
                totdata = pickle.load(file)
                dims = totdata[0]
                n = totdata[1]
                dataset = totdata[2]
                number_of_data += 1

        else:
            pass

        data.append(dataset)
        
    return [dims, n, number_of_data, data]


class NeuralNetwork:
    """Een neuraal Netwerk."""

    def __init__(self, shape, weights, bias, data, number_of_training_data, Tk = None): #shape [784,16,16,10] data is de volledige input, input_size
        
        self.data_for_DO    = data
        self.full_data      = np.array([np.reshape(i[1],-1) for i in data])
        self.traing_data    = self.full_data 
        self.shape          = shape
        self.weight         = weights
        self.bias           = bias 
        self.Tk             = Tk
        self.input_size     = int(self.shape[0])
        self.output_size    = int(self.shape[-1])
        self.weight_aanpas_groote = 0
    
    def Desired_Out(self):
        """Deze function defineerd wat de desired output word."""
        self.DO_all = []   #aantal output neuronen
        kritische_temp = 2.4
        for i in self.data_for_DO:
            if i[0]< kritische_temp:
                self.DO_all.append([1,0])
            elif i[0] >= kritische_temp :
                self.DO_all.append([0,1])                
            

    def feedforward(self, aantal = 30 , normaal = 0): 
        """Feedforward van data."""
        "normaal 0 is gwn feedforward"
        "normaal 1 test op alle trainings data"
        "normaal 2 test op alle data ook nog nooit geziene data"
        
        """instellen random gekozen data."""
        """Er zijn twee opties of random een bathc of alles dat alles is om te checken
        let hier bij op dat er twee opties zijn voor random of alles van je ranodm training set of van je daadwerkelijk data"""
        if normaal == 0:
            aantal = aantal
        elif normaal == 1:
            aantal = len(self.traing_data)
        elif normaal == 2:
            aantal = len(self.full_data)
        else:
            print('something went wrong with normaal parameter')
            
        self.index = np.zeros(aantal,dtype = int) #dit is voor de desired output
        shape = self.shape[0] #shape input data
        self.input = np.zeros([aantal,shape])
            

        if normaal == 0:
            'random choice out of trainingdata'
            for i in enumerate(np.random.choice(len(self.traing_data),size = aantal)):
                self.index[i[0]]= int(i[1])                                    #sla index op zodat de DO matched
                self.input[i[0]] = self.full_data[i[1]][0:self.input_size] 
        elif normaal == 1:
            'all trainingdata'
            for i in enumerate(range(len(self.traing_data))):
                self.index[i[0]]= int(i[1])                                    #sla index op zodat de DO matched
                self.input[i[0]] = self.full_data[i[1]][0:self.input_size] 
        elif normaal == 2:
            for i in enumerate(range(len(self.full_data))):
                self.index[i[0]]= int(i[1])                                    #sla index op zodat de DO matched
                self.input[i[0]] = self.full_data[i[1]][0:self.input_size] 
            
                
        self.DO = np.zeros([aantal,self.shape[-1]])  #creeer lijst met aantal batch met lijsten van lengte output
        for i in enumerate(self.index):
            self.DO[i[0]] = self.DO_all[i[1]]
        
            
        """Daadwerkelijke feedforword."""
        combi = zip(self.weight, self.bias) 
        self.layer = [self.input]                     #we willen elke layer opslaan
        invoer = self.input
        for i in enumerate(combi):
            if i[0] == len(self.weight)-1:
                layer = ReLU_leaky(np.add(np.dot(invoer, i[1][0]),i[1][1]))
                layer = np.array([softmax(i) for i in layer])
                # print(layer)
                # layer = sigmoid(np.add(np.dot(invoer, i[1][0]),i[1][1]))
            else:
                
                layer = ReLU_leaky(np.add(np.dot(invoer, i[1][0]),i[1][1]))
                # layer = sigmoid(np.add(np.dot(invoer, i[1][0]),i[1][1]))
            self.layer.append(layer)
            invoer = layer

    def backprop(self,learning_rate):
        """Backprop."""
        learning_rate = learning_rate
        pre_error = 2*(self.DO - self.layer[-1]) #[ [] , [] , [] ]
        """gebruik andere costfucntie"""
        for i in range(len(self.layer)-1):        #layer is van de feedforward
            output = self.layer[-i-1] # pak eerst output
            
            # error = pre_error * diff_sigmoid(output)  #[[cost voor 1 inputs], [ cost voor 2de inputs]]
            error = pre_error * diff_ReLU_leaky(output)

            samen_W = zip(error,self.layer[-i-2])
            
            #weights
            d_weight = [[j*k[0] for j in k[1]] for k in samen_W]
            d_weight = (np.sum(d_weight,0)/(len(error))) * learning_rate #gemiddelde
           
            #biases 
            d_bias = (np.sum(error,0)/(len(error))) * learning_rate  #gemiddelde
            
            #update zodat ze niet te groot worden
            d_weight    = cap(d_weight,0.01)
            d_bias      = cap(d_bias,0.2)
            # print(d_weight)
            self.weight[-i-1]  += d_weight
            self.bias[-i-1]     += d_bias
            
            self.weight_aanpas_groote = np.sum(d_weight)/len(d_weight) * 1000


            if i != len(self.layer)-2: #laatste hoeft niet
                pre_error = [np.dot(laag_error,self.weight[-1-i].T) for laag_error in error]
                
   
    def random_weight_bias(self,highed):
        for i in enumerate(self.weight):
            self.weight[i[0]] += np.random.uniform(-highed,highed,np.shape(i[1]))
        for i in enumerate(self.bias):
                self.bias[i[0]] += np.random.uniform(-highed,highed,np.shape(i[1]))
    
    def test_op_alle_traindata(self):
        self.feedforward(normaal = 1)
        output = self.layer[-1]
        combi = zip(output,self.DO)
        som = 0
        for i in combi:
            som += np.sum(np.abs(np.abs(i[0])-np.abs(i[1])))
        return -som/len(output) #genormaliseerd
    
    def test_ongeziene_data(self):
        self.feedforward(normaal = 2)
        output = self.layer[-1]
        combi = zip(output,self.DO)
        som = 0
        for i in combi:
            som += np.sum(np.abs(np.abs(i[0])-np.abs(i[1])))
        return -som/len(output) #genormaliseerd

    def conclusieT(self):
         self.feedforward(normaal = 2)
         output = self.layer[-1]
         # print(self.data_for_DO[0:50])
         T  = []
         y1 = []
         y1_std = []
         y2 = []
         y2_std = []
         lengte = int(len(output)/50)
         "50 want iedere T heeft 50 grids"
         for i in range(0,lengte):
               sub_output = output[i*50:(i+1)*50]
               lijsty1 = [i[0] for i in sub_output]
               y1.append(np.mean(lijsty1))
               y1_std.append(np.std(lijsty1))
               lijsty2 = [i[1] for i in sub_output]
               y2.append(np.mean(lijsty2))
               y2_std.append(np.std(lijsty2))
               T.append(self.data_for_DO[i*50][0])
         return T,y1,y1_std,y2,y2_std
            
