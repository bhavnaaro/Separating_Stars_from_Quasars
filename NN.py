import numpy as np
import math
from random import *
                
class Loss_functions():
    def __init__(self):
        pass

    # input 2 lists
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

    def root_mean_square(self):
        pass

    def Lipshitz_loss(self):
        pass

# inputs: 2 lists with inputs[] and weights[], 1 number
# neuron wise
def weighted_sum(wts,inputs,bias):
        sm = 0
        for i in range(len(wts)):
            sm+= wts[i]*inputs[i]
        sm += bias
        return sm


class Activation_functions():
    # Encoding: 1-Sigmoid, 2-Relu, 3- Softmax, 4-SBAF

    # inputs: 2 lists [] and []. bias is a number
    def Sigmoid(self,wts,inputs,bias):
        sm = weighted_sum(wts,inputs,bias)
        sm = math.exp(-sm)
        sm = 1/(1+sm)
        return sm

    # inputs: 2 lists [],[]. bias and act is a number
    def derivative(self,act,wts,inputs,bias):
        if(act==1):
            z = self.Sigmoid(wts,inputs,bias)
            return(z*(1-z))
        elif(act==2):
            z = self.Relu(wts,inputs,bias)
            if(z>0):
                return 1
            else:
                return 0


    def Relu(self,wts,inputs,bias):
        sm = weighted_sum(wts,inputs,bias)
        if(0>sm):
            return 0
        else:
            return sm

    def Softmax(self,wts,inputs,bias):
        pass

    def SBAF(self,wts,inputs,bias):
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

    # per layer calc of activation and output
    def activate(self,inputs,wts,bias,code):
        if(code ==1):
            return (self.Sigmoid(wts,inputs,bias))
        elif(code == 2):
            return (self.Relu(wts,inputs,bias))
        elif(code == 3):
            return (self.Softmax(wts,inputs,bias))
        elif(code == 4):
            return (self.SBAF(wts,inputs,bias)) 
        else:
            raise Exception("Unknown activation function")  
    


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

# class Optimizer:
#     def gradient_descent(self):
#         pass

#     def stochastic_gradient_descent(self):
#         pass

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

    # TP=00, FP=01, FN=10, TN=11
    def accuracy(self,y_pred,y_true):
        mat = self.confusion_matrix(y_pred,y_true)
        acc = mat[0][0]+mat[1][1]
        acc = acc/(mat[0][0]+mat[0][1]+mat[1][0]+mat[1][1])
        return(acc)

    def precision(self,y_pred,y_true):
        mat = self.confusion_matrix(y_pred,y_true)
        prec = mat[0][0]/(mat[0][0]+mat[0][1])
        return(prec)
        
    def recall(self,y_pred,y_true):
        mat = self.confusion_matrix(y_pred,y_true)
        rec = mat[0][0]/(mat[0][0]+mat[1][0])
        return(rec)

    def fscore(self,y_pred,y_true):
        rec = self.recall(y_pred,y_true)
        prec = self.precision(y_pred,y_true)
        fsc = (2*prec*rec)/(prec+rec)
        return(fsc)

class NeuralNet(Loss_functions,Activation_functions,Optimizer,Scaler,Metrics):
    def __init__(self,n):
        self.num_hidden_layers = n-2
        self.layers = n-1
        self.hidden_neurons = []
        self.hd_weights = [[] for i in range(self.num_hidden_layers)]
        self.op_weights = []
        self.weights = [ [] for i in range(self.layers)]
        self.hd_bias = [0 for i in range(self.num_hidden_layers)]
        self.op_bias = []
        self.bias = [0 for i in range(self.layers)]
        self.parameters = []
        # print(self.weights)

    # all_wts = [   [[],[],[],[]] ,[hd2 weights],[hd3 weights],[op_weights]   ]

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
                # print(self.op_neurons)
                if(len(wts)!=self.op_neurons[0]):
                    raise Exception("Number of weights don't match number of neurons")
                self.op_weights=wts
                self.weights[layer-1] = wts
                # print("w:",self.weights)
            else:
                # print(self.hidden_neurons[layer-1][0])
                # print(len(wts))
                if(len(wts)!=self.hidden_neurons[layer-1][0]):
                    raise Exception("Number of weights don't match number of neurons")
                self.hd_weights[layer-1] = wts 
                self.weights[layer-1]=wts
                # print("w:",self.weights)
                # print()
        # print("&&&",self.weights)

    def set_bias(self, num, layer):
        if(layer ==0):
            raise Exception ("Input layer cannot have weights")
        elif(layer > self.layers):
            raise Exception("Layer does not exist")
        else:
            if(layer ==self.layers):
                self.op_bias = num
                self.bias[layer-1]=num
            else:
                self.hd_bias[layer-1] = num
                self.bias[layer-1]=num 



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
        # feedforward --- DONE!
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

    def upsample(self,y_true):
        pass
            

    def flatten(self):
        pass

    def unflatten(self):
        pass
    
    def feed_forward(self,all_wts,all_bias,inputs):
        # all_wts = [   [hs1 weights],[hd2 weights],[hd3 weights],[op_weights]   ]
        # layer 0-ip 1-hd 2-hd 3-op layers=3 hidden_layers=2
        layer = 1
        inputs2 = [list(inputs)]
        while(layer<=self.layers):
            # next_ip = []
            inputs2.append([])
            if(layer<self.layers):
                for neuron in range(self.hidden_neurons[layer-1][0]):
                    # print("%",inputs,all_wts[layer-1][neuron],all_bias[layer-1])
                    op = self.activate(inputs,all_wts[layer-1][neuron],all_bias[layer-1],self.hidden_neurons[layer-1][1])
                    inputs2[layer].append(op)
                    # next_ip.append(op)
                inputs = inputs2[layer]
            else:
                for neuron in range(self.op_neurons[0]):
                    # print("%",inputs,all_wts[layer-1][neuron],all_bias[layer-1])
                    op=self.activate(inputs,all_wts[layer-1][neuron],all_bias[layer-1],self.op_neurons[1])
                    inputs2[layer].append(op)
            layer+=1
        # print("inputs:",inputs2)
        return (op,inputs2)

    def get_delta(self, y_true, inputs):
        delta = []
        for layer in range(self.layers,0,-1):
            if(layer == self.layers):
                delta.append([])
                # print("^&^",self.weights[layer-1][0])
                # print("^&$",inputs[layer-1])
                z = self.Sigmoid(self.weights[layer-1][0],inputs[layer-1],self.bias[layer-1])
                delta[self.layers - layer].append(z-y_true)
                # print("()",z-y_true)
                # print("-------------")
            else:
                # print("d",delta[self.layers-layer-1])
                # print("w",self.weights[layer])
                wt = self.weights[layer]
                wt = [[wt[j][i] for j in range(len(wt))] for i in range(len(wt[0]))]
                # wt = wt.transpose()
                ab = np.dot(wt,delta[self.layers-layer-1])
                # print("done")
                # wt = self.weights[layer]
                # print("i",ip)
                # print("wt",wt)
                # print(ab)
                # print("+_+",inputs)
                # print(self.bias[layer-1])
                # print(self.hidden_neurons[layer-1])
                # print("0000",inputs[layer])
                # de = []
                # for neuron in range(self.hidden_neurons[layer-1][0]):
                #     # ip = [inputs[layer-1][i] for i in range(len)]
                #     d = 0
                #     print("%",inputs[layer-1][neuron])
                #     for wt in self.weights[layer-1][neuron]:
                #         d+=(wt*inputs[layer][neuron])
                #     de.append(d)
                # print("+)_+",de)
                s=[]
                for neuron in range(self.hidden_neurons[layer-1][0]):
                    sig = nn.Sigmoid(self.weights[layer-1][neuron],inputs[layer],self.bias[layer-1])
                    sig = sig*(1-sig)
                    s.append(sig)
                print("SIG:",s)
                print("+_+_",ab)
                
                d=[]
                for i in range(len(s)):
                    d.append(s[i]*ab[i])
                
                delta.append(d)
                # print(ab)
                

                # print("^&^",self.weights[layer-1][0])
                # print("^&$",delta[self.layers - layer])
                # sm = weighted_sum(self.weights[layer-1],delta[self.layers - layer-1],0)
                # d = self.derivative(self.hidden_neurons[layer-1][1],self.weights[layer-1],inputs[layer-1],self.bias[layer-1])
                # delta[self.layers - layer].append(sm*d)
        # print("^^^^^^^^^^")
        # print(delta)
        # print("^^^^^^^^^^")
        return(delta)

    def get_avg_delta(self,ips,y_true):
        d=[]
        for i in range(len(ips)):
            (o,ip1) = self.feed_forward(self.weights,self.bias,ips[i])
            # print("&&",ip1)
            d.append(self.get_delta(y_true[i],ip1))
        # print(d)
        # print("===============")
        avg_d = []
        for j in range(len(d[0])):
            a = []
            for k in range(len(d[0][j])):
                s = 0
                for i in range(len(d)):
                    s+=d[i][j][k]
                s=s/len(d)
                a.append(s)
            avg_d.append(a)
        return(avg_d)

    def weight_inc(self,ips,true):
        pass

    def backpropogation(self,inputs,y_true,lr):
        delta = self.get_avg_delta(inputs,y_true)
        del_w = []
        for layer in range(self.layers,0,-1):
            d = []
            if layer == self.layers:
                for i in range(len(self.weights[layer-1])):
                    d.append(delta[0][0]*inputs[layer-1][i])
                del_w.append(d)
            else:
                d = []
                for i in range(len(self.weights[layer-1])):
                    d.append(delta[self.layers - layer][i]*inputs[layer-1][i])
                del_w.append(d)
        print(del_w)
            # for i in len(self.weights[layer-1]):
            #     d.append(delta[self.layers - layer]*inputs[layer-1][i])
            
                
                # dw = np.dot(inputs,delta[self.layers-layer])


        

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

# nn = NeuralNet(3)
# nn.set_ip_layer(2)
# nn.set_hd_layer(3,"Relu")
# nn.set_op_layer(1,"Sigmoid")
# # print(nn.hidden_neurons)
# nn.set_weights([[1,-1]],2)
# nn.set_bias(3,2)
# nn.set_weights([[1,2],[1,1],[-1,2]],1)
# nn.set_bias(4,1)
# print(nn.weights)
# print(nn.ip_neurons)
# print(nn.hidden_neurons)
# print(nn.op_neurons)
# print(nn.get_weights(1))
# (x,ip) = nn.feed_forward(nn.weights, nn.bias, [2,1])
# print(ip)

# a = Activation_functions()
# print(a.derivative(1,[1,2],[1,1],2))

nn = NeuralNet(3)
nn.set_ip_layer(2)
nn.set_hd_layer(2,"Relu")
nn.set_op_layer(1,"Sigmoid")
nn.set_weights([[0.1,0.3],[0.2,0.4]],1)
nn.set_weights([[0.5, 0.6]],2)
nn.set_bias(2,1)
nn.set_bias(4,2)
# print(nn.weights)
print(nn.get_avg_delta([[1,1],[-1,-1],[3,3]],[1,0,1]))
# print(nn.get_delta(1,[0.3,-0.4]))

# # m = Metrics()
# # yp = [1,0,1,1,1,0,0]
# # yt = [1,0,1,0,1,0,1]

# # print(m.confusion_matrix(yp,yt))
# # print("\n\n")
# # print("acc")
# # print(m.accuracy(yp,yt))
# # print(accuracy_score(yt,yp))
# # print("\n\n")
# # print("rec")
# # print(m.recall(yp,yt))
# # print(recall_score(yt,yp))
# # print("\n\n")
# # print("prec")
# # print(m.precision(yp,yt))
# # print(precision_score(yt,yp))
# # print("\n\n")
# # print("fsc")
# # print(m.fscore(yp,yt))
# # print(f1_score(yt,yp))
# # print("\n\n")
# 