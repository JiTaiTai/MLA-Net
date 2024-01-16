import torch
from torch import nn

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=8):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1,
                                kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # torch.max returns (values, indices)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv2d(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, channel, reduction=4, kernel_size=5, padding=2):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel, reduction)
        self.spatial_attention = SpatialAttentionModule(kernel_size, padding)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
