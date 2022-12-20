import torch
import torch.nn as nn

class SPM(nn.Module):
    """ Structure Perception Module """
    def __init__(self, in_dim):
        super(SPM, self).__init__()
        self.chanel_in = in_dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = out + x

        return out
