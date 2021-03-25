import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class TargetNet(nn.Module):

    def __init__(self,dataset,input_feature_number=0,output_feature_number=0):
        super(TargetNet, self).__init__()
        self.dataset = dataset
        self.input_feature_number = input_feature_number
        self.output_feature_number = output_feature_number

        if (self.dataset == 'cifar100'):
            self.model = models.densenet161()

        if (self.dataset == 'cifar10'):
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, padding=2),
                #nn.BatchNorm2d(32),
                nn.Dropout2d(),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                #nn.BatchNorm2d(64),
                nn.Dropout2d(),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.fc1 = nn.Linear(8*8*64, 128)
            self.dropout = nn.Dropout()
            self.fc2 = nn.Linear(128,10)

        if (self.dataset == 'mnist'):
            self.conv1 =  nn.Conv2d(1, 32, kernel_size=5, padding=2)
            self.drop1 = nn.Dropout2d()
            self.act1 = nn.ReLU()
            self.max1 =  nn.MaxPool2d(2)
            self.conv2 =  nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.drop2 = nn.Dropout2d()
                #nn.BatchNorm2d(32),
            self.act2 = nn.ReLU()
            self.max2 =  nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(7*7*64, 128)
            self.drop3 = nn.Dropout()
            self.fc2 = nn.Linear(128,10)

        if (self.dataset == 'adult'):
            self.fc1 = nn.Linear(input_feature_number,128)
            self.dropout = nn.Dropout()
            self.fc2 = nn.Linear(128,output_feature_number)
        
        if (self.dataset == 'texas'):
            #self.fc = nn.Linear(input_feature_number,2048)
            self.fc1 = nn.Linear(input_feature_number,1024)
            self.fc2 = nn.Linear(1024,512)
            self.fc3 = nn.Linear(512,256)
            self.fc4 = nn.Linear(256,100)

        if (self.dataset == 'purchase'):
            self.fc1 = nn.Linear(input_feature_number,1024)
            self.fc2 = nn.Linear(1024,512)
            self.fc3 = nn.Linear(512,256)
            self.fc4 = nn.Linear(256,100)

        if (self.dataset == 'titanic'):
            self.fc1 = nn.Linear(input_feature_number,128)
            self.dropout = nn.Dropout()
            self.fc2 = nn.Linear(128,output_feature_number)
        
    def forward(self, x):
        
        if (self.dataset == 'mnist'):
            out = self.conv1(x)
            out = self.drop1(out)
            out = self.act1(out)
            out = self.max1(out)

            #print (out.size())

            out = self.conv2(out)
            out = self.drop2(out)
            out = self.act2(out)
            out = self.max2(out)

            #print (out.size())
            out = self.flatten(out)

            #print (out.size())

            out = self.fc1(out)
            out = self.drop3(out)
            out = self.fc2(out)

        if (self.dataset == 'cifar10' or self.dataset == 'cifar100'):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.dropout(out)
            out = self.fc2(out)
            #out = F.softmax(out, dim=1)

        if (self.dataset == 'adult' or self.dataset == 'titanic'):
            out = x.view(x.size(0),-1)
            out = self.fc1(out)
            out = self.dropout(out)
            out = self.fc2(out)
            #out = F.log_softmax(out,dim=1)

        if (self.dataset == 'texas'):
            out = x.view(x.size(0),-1)
            out = self.fc1(out)
            out = F.tanh(out)
            out = self.fc2(out)
            out = F.tanh(out)
            out = self.fc3(out)
            out = F.tanh(out)
            out = self.fc4(out)
            out = F.tanh(out)

        if (self.dataset == 'purchase'):
            out = x.view(x.size(0),-1)
            out = self.fc1(out)
            out = F.tanh(out)
            out = self.fc2(out)
            out = F.tanh(out)
            out = self.fc3(out)
            out = F.tanh(out)
            out = self.fc4(out)
            out = F.tanh(out)

        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
