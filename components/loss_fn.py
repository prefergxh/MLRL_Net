import torch
import torch.nn as nn
class CenterLoss(nn.Module):
    def __init__(self,num_classes=6,feat_dim=1024):
        super(CenterLoss,self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes,feat_dim))
    def forward(self,features,labels):
        batch_size = features.size(0)
        centers_batch = self.centers.index_select(0,labels)
        diff = features-centers_batch
        loss = (diff.pow(2).sum()) / (2.0*batch_size)
        return loss