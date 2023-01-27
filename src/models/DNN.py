import torch
import torch.nn as nn

class NN(nn.Module):
   
    def __init__(self, layer_array=[28, 40, 1]):
        super(NN, self).__init__()

        if len(layer_array)<3:
            print("Network must contain at least one input layer, one output layer and one hidden layer!")
            layer_array = [28, 40, 1]
        
        print("------- Initializing DNN -------")
        print("Input layer size: ", layer_array[0])
        print("Number of hidden layers : ", len(layer_array))
        for i in range(len(layer_array[1:-1])):
           print("Hidden layer {} size: {}".format(1+i, layer_array[1+i]))
        print("Output layer size: ", layer_array[-1])
        print("-------------------------------")
        
        self.input_size = layer_array[0]
        self.output_size = layer_array[-1]
        self.hidden_layers = layer_array[1:-1]

        seq_ = []
        for i in range(1,len(layer_array)):
            seq_.append(nn.Linear(layer_array[i-1], layer_array[i]))
            seq_.append(nn.ReLU())
        seq_.append(nn.Softplus())
        self.network = nn.Sequential(*seq_)
        self.float()
    
    def forward(self, data):
        return self.network(data).reshape(-1,)
