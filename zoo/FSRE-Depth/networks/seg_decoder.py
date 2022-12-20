from utils.depth_utils import *

class SegDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(1), num_output_channels=19, use_skips=True):
        super(SegDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        convs = []
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            convs.append(ConvBlock(num_ch_in, num_ch_out))

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            convs.append(ConvBlock(num_ch_in, num_ch_out))

        for s in self.scales:
            convs.append(Conv3x3(self.num_ch_dec[s], self.num_output_channels))

        self.decoder = nn.ModuleList(convs)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, input_features):
        outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.decoder[-2 * i + 8](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.decoder[-2 * i + 9](x)
            outputs[('s_feature', i)] = x
            if i in self.scales:
                out = self.decoder[10 + i](x)
                outputs[('seg_logits', i)] = out

        return outputs
