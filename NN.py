import numpy as np
import math
from sklearn.metrics import log_loss
from random import *

class Loss_functions():
    def __init__(self):
        pass

    # Binary cross entropy
    def logLoss(self,y_true,y_pred,eps = 1e-15):
        loss = []
        p = np.clip(y_pred, eps, 1 - eps)
        for i in range(len(y_true)):
            if y_true[i] == 1:
                loss.append(-1* math.log(p[i]))
            else:
                loss.append(-1* math.log(1 - p[i]))
        loss = np.array(loss)
        return np.mean(loss)
        

    def sparse_categorical_cross_entropy(self):
        pass

    def root_mean_square(self):
        pass

    def Lipshitz_loss(self):
        pass


class Activation_functions():
    # Encoding: 1-Sigmoid, 2-Relu, 3- Softmax, 4-SBAF

    def weighted_avg(self,wts,inputs):
        sm = 0
        for i in range(len(wts)):
            sm+= wts[i]*inputs[i]
        sm = sm/(len(wts))
        return sm

    def Sigmoid(self,wts,inputs):
        sm = self.weighted_avg(wts,inputs)
        sm = math.exp(-sm)
        sm = 1/(1+sm)
        return sm

    def Relu(self,wts,inputs):
        sm = self.weighted_avg(wts,inputs)
        sm = max(sm)
        if(0>sm):
            return 0
        else:
            return sm

    def Softmax(self,wts,inputs):
        pass

    def SBAF(self):
        pass

    def encode(self,act):
        act = act.lower()
        if(act =='sigmoid'):
            return(1)
        elif(act == 'relu'):
            return(2)
        elif(act == 'softmax'):
            return(3)
        elif(act == 'SBAF'):
            return(4)

class Scaler:
    def minmax(self,array):
        array = np.array(array)
        mn = np.min(array)
        mx = np.max(array)
        
        scaled_array = []
        for element in array:
            s = (element-mn)/(mx-mn)
            scaled_array.append(s)

        return(np.array(scaled_array))


    def standard_scale(self,array):
        array = np.array(array)
        sm = np.sum(array)
        scaled_array = []
        for element in array:
            scaled_array.append(element/sm)
        return(np.array(scaled_array))

class Optimizer:
    def find_neighbours(self,weights,n):
        pass

    def gradient_descent(self):
        pass

    def genetic_algorithm(self):
        pass

    def stochastic_gradient_descent(self):
        pass

class Metrics:
    def confusion_matrix(self,y_pred,y_true):
        if (len(y_pred)!=len(y_true)):
            raise Exception("Predicted values and True values not compatible")

        TN = 0
        TP = 0
        FN = 0
        FP = 0

        for i in range(len(y_pred)):
            if(y_pred[i]==1):
                if(y_true[i] == 1):
                    TP+=1
                else:
                    FP+=1
            else:
                if(y_true[i]==0):
                    TN+=1
                else:
                    FN+=1
        return([[TP,FP],[FN,TN]])
        

    def accuracy(self,y_pred,y_true):
        mat = self.confusion_matrix(y_pred,y_true)
        acc = mat[0][0]+mat[1][1]
        acc = acc/(mat[0][0]+mat[0][1]+mat[1][0]+mat[1][1])
        return(acc)

    def precision(self):
        # mat = self.confusion_matrix(y_pred,y_true)
        pass


    def fscore(self):
        pass

class NeuralNet(Loss_functions,Activation_functions,Optimizer,Scaler,Metrics):
    def __init__(self,n):
        self.num_hidden_layers = n-2
        self.layers = n-1
        self.hidden_neurons = []
        self.hd_weights = [[] for i in range(self.num_hidden_layers)]
        self.op_weights = []
        self.weights = []
        self.hd_bias = []
        self.op_bias = []
        self.bias = []
        self.parameters = []

    def set_ip_layer(self,num):
        self.ip_neurons = [num,-1]

    # call multiple times to set more hidden layers
    def set_hd_layer(self,num,act):
        self.hidden_neurons.append([num,self.encode(act)])
        return(len(self.hidden_neurons))

    def set_op_layer(self,num,act):
        self.op_neurons = [num,self.encode(act)]

    def set_weights(self,wts,layer):
        if(layer ==0):
            raise Exception ("Input layer cannot have weights")
        elif(layer > self.layers):
            raise Exception("Layer does not exist")
        else:
            if(layer ==self.layers):
                if(len(wts)<self.op_neurons[0] or len(wts)>self.op_neurons[0]):
                    raise Exception("Number of weights don't match number of neurons")
                self.op_weights=wts
            else:
                if(len(wts)<self.hidden_neurons[layer-1][0] or len(wts)>self.hidden_neurons[layer-1][0]):
                    raise Exception("Number of weights don't match number of neurons")
                self.hd_weights[layer-1] = wts 


    def get_weights(self,layer):
        if(layer ==0):
            raise Exception("Input layer cannot have weights")
        elif(layer > self.layers):
            raise Exception("Layer does not exist")
        else:
            if(layer ==self.layers):
                return (self.op_weights)
            else:
                return (self.hd_weights[layer-1])

    def get_neurons(self,layer):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def cross_validation(self,nfolds,dataset):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / nfolds)
        for i in range(nfolds):
                fold = list()
                while len(fold) < fold_size:
                        index = randrange(len(dataset_copy))
                        fold.append(dataset_copy.pop(index))
                dataset_split.append(fold)
        return dataset_split

    def upsample(self):
        pass

    def flatten(self):
        pass

    def unflatten(self):
        pass
# NOTES:
# # input layer is layer 0
# # output layer is n-1

# TESTS:
#   # l = Loss_functions()
#   # a=[1,0]
#   # b=[0.4,0.6]
#   # print(l.logLoss(a,b))
#   # print(log_loss(a,b))
# 

#   # s = Scaler()
#   # a=[4,6]
#   # print(s.standard_scale(a))
# 

# # nn = NeuralNet(3)
# # nn.set_ip_layer(3)
# # nn.set_hd_layer(2,"Sigmoid")
# # nn.set_op_layer(1,"Sigmoid")
# # print(nn.hidden_neurons)
# # nn.set_weights([1],2)
# # print(nn.ip_neurons)
# # print(nn.hidden_neurons)
# # print(nn.op_neurons)
# # print(nn.get_weights(2))
# 
