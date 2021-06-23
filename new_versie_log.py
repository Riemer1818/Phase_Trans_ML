import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import timeit
import random
import os, sys
import pickle

np.random.seed(5)

"""--------------------------------------------------------------------------------------------------------""" 


def unpickle_dir(directory):
    data = []
    number_of_data = 0 
    for filename in os.listdir(directory):

        if filename.endswith('.pkl'):
            
            f = os.path.join(directory, filename)
            with open(f, "rb") as file:
                totdata = pickle.load(file)
                number_of_data += 1
            data.append(totdata)

        else:
            pass
        
    data = np.array(data)

    return [number_of_data, data]

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

    output = output - max(output)[0]
    # if (max(np.abs(output))[0])/2 > 10:
    totaal = sum(np.e ** output)
    return np.e ** output/totaal

def Jacobian_softmax(output):
    N = len(output)
    s1 = output[0][0]
    s2 = output[1][0]
    jacobian = np.array([[-s2,s2],[s1,-s1]])

    return jacobian

                                                            
def Jacobian_ReLU_leaky(x,factor = 0.15):
    y1 = ((x > 0) * 1)                                            
    y2 = ((x <= 0) * factor)                                         
    leaky = y1 + y2
    N = len(leaky)
    matrix = np.zeros((N,N))
    for i in range(N):
        matrix[i][i] = leaky[i]
    return matrix                                                           
            

       
def makematrix(inp,lengte):
    matrix = []
    for i in range(lengte):
        matrix.append(inp)
        
    return matrix
    


class NeuralNetwork:
    """Een neuraal Netwerk."""

    def __init__(self, shape,train_data,test_data, number_of_training_data, number_of_test_data, Tk): #shape [784,16,16,10] data is de volledige input, input_size
        
        self.shape          = shape
        
        # self.beginwaardes   = [i[1] for i in data]
        self.input_size     = int(shape[0])
        self.output_size    = int(shape[-1])
        #100 data per temp
        self.train_data = train_data
        self.test_data = test_data
        
        'wat in de volgende 9 regels gebeurt is vrij gevoelig qua data input.'
        'dat kan dus beteken dat hier wat aangepast moet worden als we het op de surver draaien'
        'heb het nu zo gemaakt dat het werkt voor een georderde lijst van [[temp,grid],etc]'
        self.DO_train    = self.Desired_Out(self.train_data, Tk)
        self.DO_test     = self.Desired_Out(self.test_data, Tk)
        
        self.temp_train  = [i[0] for i in self.train_data]
        self.temp_test   = [i[0] for i in self.test_data]

        self.train_data    = np.array([np.reshape(i[1],(self.input_size,1)) for i in self.train_data]) 
        self.test_data     = np.array([np.reshape(i[1],(self.input_size,1)) for i in self.test_data]) 
        
        self.weight     = [np.random.uniform(-0.1,0.1,(shape[i+1],shape[i])) for i in range(len(shape)-1)]
        self.bais       = [np.random.uniform(-1,1,(shape[i+1],1)) for i in range(len(shape)-1)]

        self.number_of_training_data    = number_of_training_data
        self.number_of_test_data        = number_of_test_data

    def Desired_Out(self, data, kritische_temp):
        """Deze function defineerd wat de desired output word."""
        DO = []   #aantal output neuronen
        for i in data:
            if i[0]< kritische_temp:
                DO.append([[1],[0]])
            elif i[0] >= kritische_temp:
                DO.append([[0],[1]])
                
        return DO
        # print(self.DO_all)
                
    def feedforward(self, aantal = 30 ,normaal= 0): 
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
            aantal = len(self.train_data)
        elif normaal == 2:
            aantal = len(self.test_data)
        else:
            print('something went wrong with normaal parameter')
            
        self.index = np.zeros(aantal,dtype = int) #dit is voor de desired output
        shape = self.shape[0] #shape input data
        self.input = np.zeros([aantal,shape,1])


        if normaal == 0:
            'random choice out of trainingdata'
            for i in enumerate(np.random.choice(len(self.train_data),size = aantal)):
                self.index[i[0]]= int(i[1])                                    #sla index op zodat de DO matched
                self.input[i[0]] = self.train_data[i[1]][0:self.input_size] 
        elif normaal == 1:
            'all trainingdata'
            for i in enumerate(range(len(self.train_data))):
                self.index[i[0]]= int(i[1])                                    #sla index op zodat de DO matched
                self.input[i[0]] = self.train_data[i[1]][0:self.input_size] 
        elif normaal == 2:
            'alle test data'
            for i in enumerate(range(len(self.test_data))):
                self.index[i[0]]= int(i[1])                                    #sla index op zodat de DO matched
                self.input[i[0]] = self.test_data[i[1]][0:self.input_size] 
            
        
        shapeDO = np.shape(self.DO_train[0]) #gwn shape van de DO output, dit is zo nodig
        self.DO = np.zeros([aantal,shapeDO[0],shapeDO[1]])  
        if normaal != 2:
            #creeer lijst met aantal batch met lijsten van lengte output
            for i in enumerate(self.index):
                self.DO[i[0]] = self.DO_train[i[1]]
        else:
            for i in enumerate(self.index):
                self.DO[i[0]] = self.DO_test[i[1]]
        
            
        """Daadwerkelijke feedforword."""
        combi = zip(self.weight, self.bais) 
        self.layer = [[self.input,self.input]]                  #we willen elke layer opslaan
        invoer = self.input
        for i in enumerate(combi):
            if i[0] == len(self.weight)-1:          #laatste layer
                z = np.add((i[1][0] @ invoer),i[1][1])
                layer = z
                'doe geen Relu meer voor mijn softmax'
                # layer = ReLU_leaky(z)
                layer = np.array([softmax(i) for i in layer])

                # layer = sigmoid(np.add(np.dot(invoer, i[1][0]),i[1][1]))
            else:

                z = np.add((i[1][0]@invoer),i[1][1])
                layer = ReLU_leaky(z)
                
                # layer = sigmoid(np.add(np.dot(invoer, i[1][0]),i[1][1]))

            self.layer.append([z,layer]) #z is raw output, layer is met Activtion er over heen
            invoer = layer
   

    """raar dat ik ouput gebruik ipv z """


    def backprop(self,learning_rate):

        """Backprop."""

        learning_rate = learning_rate

        pre_softmax = [Jacobian_softmax(i) for i in self.layer[-1][1]] #2,2 shape

        DO = [-i.T[0] for i in self.DO] #deze min deed alles fantastisch omdat het logg

        combi = zip(DO ,pre_softmax)
        
        pre_error = [i[0]@i[1] for i in combi] 
        self.preloss = pre_error
        self.update = []
        """gebruik andere costfucntie"""
        for i in range(len(self.layer)-1):        #layer is van de feedforward
            inp = self.layer[-i-2][1] #dit is zonder relu met relu, pak eerst input van uitachter gerekend
            lengte = len(self.preloss[0])  #gewoon 1 pakken voor de dimensie
            inp = np.array([makematrix(i.T[0], lengte) for i in inp])
            combi = zip(self.preloss, inp)
            Update_W = [np.multiply(np.reshape(j[0],(-1,1)),j[1])for j in combi]


    
            Update_W = (sum(Update_W,0)/len(Update_W)) * learning_rate #normaliseer over batch size


            self.update.append(Update_W)
            self.weight[-i-1]  += Update_W
            # print(Update_W)

            """for bais we have DL(b) = I identiteit matrix"""

            Update_B = (np.sum(self.preloss,0)/len(pre_error)) * learning_rate
            Update_B = np.reshape(Update_B,(-1,1))
            
            'omdat pre_erro shape (1,2) heeft maar moet (2,1) worden moeten we transpose in bais.'
            
            self.bais[-i-1]     += Update_B
           
            "DL(I) berekenen zodat we de volgende update voor W en B kunnen berekenen"
            "voor de laatste laag hoeft dit niet meer gedaan te worden want daar doen we niks meer mee."
            if i != len(self.layer)-2:
                

                DL = [self.weight[-i-1] for j in pre_error]
                pre_ReLU = [Jacobian_ReLU_leaky(i) for i in self.layer[-i-2][0]]
                combi = zip(self.preloss,DL,pre_ReLU)
                self.preloss = [j[0]@j[1]@j[2] for j in combi]
   
    def random_weight_bais(self,highed):

        for i in enumerate(self.weight):
            self.weight[i[0]] += np.random.uniform(-highed,highed,np.shape(i[1]))
        for i in enumerate(self.bais):
                self.bais[i[0]] += np.random.uniform(-highed,highed,np.shape(i[1]))
    
    def testen_nn(self, normaal): 

        #normaal = 1 of 2
        self.feedforward(normaal = normaal)
        output = self.layer[-1][1]
        combi = zip(output,self.DO)
        som = 0
        for i in combi:
            som += np.sum(np.abs(np.abs(i[0])-np.abs(i[1])))
        return -som/len(output) #genormaliseerd
    
    def conclusieT(self, train = False):

         """'dit  is om te testen hoe het netwerk het doet op de data'
        'als train = False dan test je het op """
         if train:
            self.feedforward(normaal = 1)
         else:
            self.feedforward(normaal = 2)
         output = self.layer[-1][-1]
         # print(output)
         # print(self.data_for_DO[0:50])
         T  = []
         y1 = []
         y1_std = []
         y2 = []
         y2_std = []

         if train: 
            numberT = self.number_of_training_data
         else: 
            numberT = self.number_of_test_data
           
         
         lengte = int(len(output)/numberT)

         
         for i in range(0,lengte):

            sub_output = output[i*numberT:(i+1)*numberT]
            lijsty1 = [i[0] for i in sub_output]
            y1.append(np.mean(lijsty1))
            y1_std.append(np.std(lijsty1))
            lijsty2 = [i[1] for i in sub_output]
            y2.append(np.mean(lijsty2))
            y2_std.append(np.std(lijsty2))
            if train:
                T.append(self.temp_train[i*numberT])
            else:
                T.append(self.temp_test[i*numberT])
         return T,y1,y1_std,y2,y2_std
            
def learning_rate_function(X,grens = 0.1): 
    #  kies je factor je eerste begin foutmarge, factor/2 ^ 2 is je sig
    """wil proberen de eerste fout op 4sigma te zetten dat lijkt logisch omdat dat nog redelijk fout is dan dan zit
    bij de minimale fout .
    kies je sigma dus eerstfout /4"""
    sigma= 1/4
    learning_rate = -1* np.e ** -(1*(X) ** 2) -1  #chech in geogabra mooie functie die -4 is laagste learning rate
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
    
        
if __name__ == '__main__':

    dirname = sys.argv[1]
    print("using: ", dirname, " as training input directory")

    dims = int(sys.argv[2])

    n = int(sys.argv[3])

    epochs = int(sys.argv[4])

    hidden_layer = int(sys.argv[5])

    Tk_dict = {2:2.27, 3:4.5, 4:6.68}

    Tk = Tk_dict[dims]

    train_dirname = os.path.join("./train_data", dirname)
    test_dirname = os.path.join("./test_data", dirname)

    number_of_training_data, train_data = unpickle_dir(train_dirname)
    number_of_test_data, test_data = unpickle_dir(test_dirname)

    train_data = np.concatenate(train_data, axis=0)
    test_data = np.concatenate(test_data, axis=0)

    train_data = train_data[train_data[:,0].argsort()]
    test_data = test_data[test_data[:,0].argsort()]

    size = n**dims 

    shape = [size,hidden_layer,2]
    out_dirname = f"{dirname}_output_NN_{epochs}_{hidden_layer}"
    os.mkdir(out_dirname)  

    nn = NeuralNetwork(shape, train_data, test_data, number_of_training_data, number_of_test_data, Tk) 

    foutmarge_training_data = []
    foutmarge_ongeziene_data = []
    weight_aanpas_groote = []
    nfactor_lijst = []        
    
    nfactor = -1

    i = 0
    while i <= epochs:
        nn.feedforward(aantal = number_of_training_data, normaal = 0)
        nn.backprop(10 ** nfactor)
        
        if i%10 == 0:
            print(i)
            fouttrain = nn.testen_nn(normaal = 1)
            fouttest = nn.testen_nn(normaal = 2)  
            foutmarge_training_data.append(fouttrain)
            foutmarge_ongeziene_data.append(fouttest)

        i += 1  


    T,y1,y1_std,y2,y2_std = nn.conclusieT()
    plt.figure()
    plt.title("test")
    plt.errorbar(T,y1,y1_std,fmt='o', label = 'test')
    plt.savefig(os.path.join(out_dirname, "conclusie_test.png"))
    np.save(os.path.join(out_dirname,"conclusie_test_data.npy"), [T,y1,y1_std,y2,y2_std], allow_pickle=True)

    T,y1,y1_std,y2,y2_std = nn.conclusieT(train = True)
    plt.figure()
    plt.title("train")
    plt.errorbar(T,y1,y1_std,fmt='o', label = 'train')  
    plt.savefig(os.path.join(out_dirname,"conclusie_train.png"))
    np.save(os.path.join(out_dirname,"conclusie_train_data.npy"), [T,y1,y1_std,y2,y2_std], allow_pickle=True)
     
    plt.figure()
    plt.title("fout plot")
    plt.plot(foutmarge_training_data, label = "fout op trainingdata")  
    plt.plot(foutmarge_ongeziene_data, label = "fout op alle data inclusie ongeziene data")
    plt.legend()
    plt.plot(np.zeros(len(foutmarge_ongeziene_data)))
    plt.xlabel("epochs")
    plt.ylabel("genormaliserede fout")
    plt.savefig(os.path.join(out_dirname,"conclusie_fout_plot.png"))
    np.save(os.path.join(out_dirname,"foutmarge_training_data.npy"), foutmarge_training_data, allow_pickle=True)
    np.save(os.path.join(out_dirname,"foutmarge_ongeziene_data.npy"), foutmarge_ongeziene_data, allow_pickle=True)
             
    np.save(os.path.join(out_dirname,"weights.npy"), nn.weight)
    np.save(os.path.join(out_dirname,"bias.npy"), nn.bais)
