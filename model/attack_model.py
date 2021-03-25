import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class gan_AttackNet(nn.Module):
    
    def __init__(self,dem):
        super(gan_AttackNet, self).__init__()
        self.fc1 = nn.Linear(dem,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,64)

        self.fc4 = nn.Linear(dem,512)
        self.fc5 = nn.Linear(512,64)

        self.fc6 = nn.Linear(128,64)
        self.fc7 = nn.Linear(64,2)
        
    def forward(self, x1,x2):
        x1 = x1.view(x1.size(0),-1)
        x2 = x2.view(x2.size(0),-1)

        out1 = self.fc1(x1)
        out1 = F.relu(out1)
        out1 = self.fc2(out1)
        out1 = F.relu(out1)
        out1 = self.fc3(out1)
        out1 = F.relu(out1)

        out2 = self.fc4(x2)
        out2 = F.relu(out2)
        out2 = self.fc5(out2)
        out2 = F.relu(out2)

        out = torch.cat((out1,out2),dim=1)

        out = self.fc6(out)
        out = F.relu(out)
        out = self.fc7(out)
        out = F.softmax(out,dim=1)

        return out
    ## TYPE ONE


class onelayer_AttackNet(nn.Module):

    def __init__(self, dem):
        super(onelayer_AttackNet, self).__init__()
        self.fc1 = nn.Linear(dem, 64)
        self.fc2 = nn.Linear(64,2)

    def forward(self, x1):
        x1 = x1.view(x1.size(0), -1)
        #print (x1.size())
        out = self.fc1(x1)
        out = F.relu(out)
        #print (out.size())
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out

