import torch 
import torch.nn as nn
import torch.nn.functional as F

class Onehot_AttackNet(nn.Module):

    ###  FROM Machine Learning with Membership Privacy using Adversarial Regularization
    def __init__(self,dem1,dem2):
        super(Onehot_AttackNet,self).__init__()
        self.fc1_1 = nn.Linear(dem1,1024)
        self.fc1_2 = nn.Linear(1024,512)
        self.fc1_3 = nn.Linear(512,64)
        self.fc2_1 = nn.Linear(dem2,512)
        self.fc2_2 = nn.Linear(512,64)
        self.fc1 = nn.Linear(128,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,2)
    
    def forward(self,x1,x2):
        out1 = x1.view(x1.size(0),-1)
        out1 = self.fc1_1(out1)
        out1 = F.relu(out1)
        out1 = self.fc1_2(out1)
        out1 = F.relu(out1)
        out1 = self.fc1_3(out1)
        out1 = F.relu(out1)
        out2 = x2.view(x2.size(0),-1)
        out2 = self.fc2_1(out2)
        out2 = F.relu(out2)
        out2 = self.fc2_2(out2)
        out2 = F.relu(out2)
        out = torch.cat((out1,out2),dim=1)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.softmax(out,dim=0)
        return out 