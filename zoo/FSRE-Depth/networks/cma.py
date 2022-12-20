from networks.depth_decoder import DepthDecoder
from networks.multi_embedding import MultiEmbedding
from networks.seg_decoder import SegDecoder
from utils.depth_utils import *


class CMA(nn.Module):
    def __init__(self, num_ch_enc=None, opt=None):
        super(CMA, self).__init__()

        self.scales = opt.scales
        cma_layers = opt.cma_layers
        self.opt = opt
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        in_channels_list = [32, 64, 128, 256, 16]

        num_output_channels = 1

        self.depth_decoder = DepthDecoder(num_ch_enc, num_output_channels=num_output_channels,
                                          scales=opt.scales,
                                          opt=self.opt)
        self.seg_decoder = SegDecoder(num_ch_enc, num_output_channels=19,
                                      scales=[0])

        att_d_to_s = {}
        att_s_to_d = {}
        for i in cma_layers:
            att_d_to_s[str(i)] = MultiEmbedding(in_channels=in_channels_list[i],
                                                num_head=opt.num_head,
                                                ratio=opt.head_ratio)
            att_s_to_d[str(i)] = MultiEmbedding(in_channels=in_channels_list[i],
                                                num_head=opt.num_head,
                                                ratio=opt.head_ratio)
        self.att_d_to_s = nn.ModuleDict(att_d_to_s)
        self.att_s_to_d = nn.ModuleDict(att_s_to_d)

    def forward(self, input_features):

        depth_outputs = {}
        seg_outputs = {}
        x = input_features[-1]
        x_d = None
        x_s = None
        for i in range(4, -1, -1):
            if x_d is None:
                x_d = self.depth_decoder.decoder[-2 * i + 8](x)
            else:
                x_d = self.depth_decoder.decoder[-2 * i + 8](x_d)

            if x_s is None:
                x_s = self.seg_decoder.decoder[-2 * i + 8](x)
            else:
                x_s = self.seg_decoder.decoder[-2 * i + 8](x_s)

            x_d = [upsample(x_d)]
            x_s = [upsample(x_s)]

            if i > 0:
                x_d += [input_features[i - 1]]
                x_s += [input_features[i - 1]]

            x_d = torch.cat(x_d, 1)
            x_s = torch.cat(x_s, 1)

            x_d = self.depth_decoder.decoder[-2 * i + 9](x_d)
            x_s = self.seg_decoder.decoder[-2 * i + 9](x_s)

            if (i - 1) in self.opt.cma_layers:
                if len(self.opt.cma_layers) == 1:
                    x_d_att = self.att_d_to_s(x_d, x_s)
                    x_s_att = self.att_s_to_d(x_s, x_d)
                    x_d = x_d_att
                    x_s = x_s_att
                else:
                    x_d_att = self.att_d_to_s[str(i - 1)](x_d, x_s)
                    x_s_att = self.att_s_to_d[str(i - 1)](x_s, x_d)
                    x_d = x_d_att
                    x_s = x_s_att

            if self.opt.sgt:
                depth_outputs[('d_feature', i)] = x_d
                seg_outputs[('s_feature', i)] = x_s
            if i in self.scales:
                outs = self.depth_decoder.decoder[10 + i](x_d)
                depth_outputs[("disp", i)] = F.sigmoid(outs[:, :1, :, :])
                if i == 0:
                    outs = self.seg_decoder.decoder[10 + i](x_s)
                    seg_outputs[("seg_logits", i)] = outs[:, :19, :, :]

        return depth_outputs, seg_outputs
