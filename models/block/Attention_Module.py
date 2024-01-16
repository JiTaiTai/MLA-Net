import torch
import torch.nn as nn
from models.block.High_Frequency_Module import HighFrequencyModule


class TH(nn.Module):
    def __init__(self, input_channel):
        super(TH, self).__init__()
        self.input_channel = input_channel
        self.patch_size = 8
        self.query_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)

        self.query_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.query_conv2_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.key_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.value_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1) 
        self.Sigmoid = nn.Sigmoid()

    def compute_attention(self, x, query_conv, key_conv, value_conv):
        m_batchsize, C, width, height = x.size()
        proj_query = query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        energy = self.softmax(energy)
        proj_value = value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, energy.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        return out

    def forward(self, x):
        residual = x
        m_batchsize, C, width, height = x.size()
        self.patch_size = 8
        # 计算patch数量
        num_patches_x = width // self.patch_size
        num_patches_y = height // self.patch_size

        #块内注意力
        x_patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x_patches = x_patches.contiguous().view(m_batchsize, C, num_patches_x, self.patch_size, num_patches_y, self.patch_size)
        x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.patch_size, self.patch_size)
        att_patches = self.compute_attention(x_patches, self.query_conv, self.key_conv, self.value_conv)
        att_patches = att_patches.view(m_batchsize, num_patches_x, num_patches_y, C, self.patch_size, self.patch_size)
        att_patches = att_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        res1 = att_patches.view(m_batchsize, C, width, height)

        # 块间注意力计算
        x_patches_inter = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x_patches_inter = x_patches_inter.contiguous().view(m_batchsize, C, num_patches_x, self.patch_size, num_patches_y, self.patch_size)
        x_patches_inter = x_patches_inter.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.patch_size, self.patch_size)

        # 重塑每个patch以增加通道数
        x_patches_inter = x_patches_inter.view(-1, C * self.patch_size * self.patch_size, 1, 1)

        # 计算块间注意力
        att_patches_inter = self.compute_attention(x_patches_inter, self.query_conv_, self.key_conv_, self.value_conv_)
        att_patches_inter = att_patches_inter.view(m_batchsize, num_patches_x, num_patches_y, C, self.patch_size, self.patch_size)
        att_patches_inter = att_patches_inter.permute(0, 3, 1, 4, 2, 5).contiguous()
        res2 = att_patches_inter.view(m_batchsize, C, width, height)

        # 将块内和块间注意力相加并应用激活函数
        res = res1 + res2
        # res = res1
        # res = res2
        x = self.Sigmoid(res)
        mask = torch.mul(residual, x)
        output = residual + mask
        return output
    
class TH2(nn.Module):
    def __init__(self, input_channel):
        super(TH2, self).__init__()
        self.input_channel = input_channel
        self.patch_size = 8

        self.query_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)

        self.query_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.query_conv2_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.key_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)
        self.value_conv_ = nn.Conv2d(in_channels=input_channel*64, out_channels=input_channel*64, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1) 
        self.Sigmoid = nn.Sigmoid()

    def compute_attention(self, x, query_conv, key_conv, value_conv):
        m_batchsize, C, width, height = x.size()
        proj_query = query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        energy = self.softmax(energy)
        proj_value = value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, energy.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        return out

    def forward(self, x):
        residual = x
        m_batchsize, C, width, height = x.size()
        self.patch_size = 8
        # 计算patch数量
        num_patches_x = width // self.patch_size
        num_patches_y = height // self.patch_size

        # 块内注意力
        x_patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x_patches = x_patches.contiguous().view(m_batchsize, C, num_patches_x, self.patch_size, num_patches_y, self.patch_size)
        x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.patch_size, self.patch_size)
        att_patches = self.compute_attention(x_patches, self.query_conv, self.key_conv, self.value_conv)
        att_patches = att_patches.view(m_batchsize, num_patches_x, num_patches_y, C, self.patch_size, self.patch_size)
        att_patches = att_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        res1 = att_patches.view(m_batchsize, C, width, height)

        # 块间注意力计算
        x_patches_inter = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x_patches_inter = x_patches_inter.contiguous().view(m_batchsize, C, num_patches_x, self.patch_size, num_patches_y, self.patch_size)
        x_patches_inter = x_patches_inter.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.patch_size, self.patch_size)

        # 重塑每个patch以增加通道数
        x_patches_inter = x_patches_inter.view(-1, C * self.patch_size * self.patch_size, 1, 1)

        # 计算块间注意力
        att_patches_inter = self.compute_attention(x_patches_inter, self.query_conv_, self.key_conv_, self.value_conv_)
        att_patches_inter = att_patches_inter.view(m_batchsize, num_patches_x, num_patches_y, C, self.patch_size, self.patch_size)
        att_patches_inter = att_patches_inter.permute(0, 3, 1, 4, 2, 5).contiguous()
        res2 = att_patches_inter.view(m_batchsize, C, width, height)

        # 将块内和块间注意力相加并应用激活函数
        res = res1 + res2

        x = self.Sigmoid(res)
        mask = torch.mul(residual, x)
        output = residual + mask
        return output

