import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import time
from models.block.Base import Conv3Relu, Conv1Relu
from models.block.Drop import DropBlock
from models.block.Field import PPM, ASPP4, SPP, ASPP3
from models.block.PAM import PAM
from models.block.CAM import CAM
from models.block.BAM import BAM
from models.block.Attention_Module import TH,TH2
from models.block.MSCAM import AFF,iAFF,absAFF,multiAFF
from models.block.cbam import CBAM
import os
class FPNNeck(nn.Module):
    def __init__(self, inplanes, neck_name='fpn+ppm+fuse'):
        super().__init__()

        self.stageabs1_Conv1 = Conv3Relu(inplanes , inplanes)  # channel: 2*inplanes ---> inplanes
        self.stageabs2_Conv1 = Conv3Relu(inplanes * 2, inplanes * 2)  # channel: 4*inplanes ---> 2*inplanes
        self.stageabs3_Conv1 = Conv3Relu(inplanes * 4, inplanes * 4)  # channel: 8*inplanes ---> 4*inplanes
        self.stageabs4_Conv1 = Conv3Relu(inplanes * 8, inplanes * 8)  # channel: 16*inplanes ---> 8*inplanes


        self.stage_Conv_after_up = Conv3Relu(inplanes , inplanes)
        self.stage2_Conv_after_up = Conv3Relu(inplanes * 2, inplanes)
        self.stage3_Conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage4_Conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4)

        self.stage3_Conv_after_up2 = Conv3Relu(inplanes * 8, inplanes * 2)
        self.stage2_Conv_after_up3 = Conv3Relu(inplanes * 4, inplanes)
        self.stage2_Conv_after_up2 = Conv3Relu(inplanes * 8, inplanes)

        self.stage_Conv2 = Conv1Relu(inplanes , inplanes)
        self.stage1_Conv2 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv2 = Conv3Relu(inplanes * 8, inplanes * 4)
        self.stage4_Conv2 = Conv3Relu(inplanes * 16, inplanes * 8)
        self.stage5_Conv2 = Conv3Relu(inplanes * 3, inplanes)

        self.stage1_Conv20 = Conv3Relu(inplanes * 4, inplanes)
        self.stage2_Conv20 = Conv3Relu(inplanes * 6, inplanes * 2)

        self.aspp3 = ASPP3(inplanes*4)
        self.aspp4 = ASPP4(inplanes*8)

        self.thfab1 = TH(inplanes)
        self.thfab2 = TH2(inplanes*2)

        self.Sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # PPM/ASPP比SPP好
        if "+ppm+" in neck_name:
            self.expand_field = PPM(inplanes*8)
        elif "+aspp+" in neck_name:
            self.expand_field = ASPP4(inplanes*8)
        elif "+spp+" in neck_name:
            self.expand_field = SPP(inplanes*8)
        else:
            self.expand_field = None

        if "fuse" in neck_name:
            self.stage1_Conv3 = Conv3Relu(inplanes * 2, inplanes)
            self.stage2_Conv3 = Conv3Relu(inplanes * 2, inplanes)   # 降维
            self.stage3_Conv3 = Conv3Relu(inplanes * 4, inplanes)
            self.stage4_Conv3 = Conv3Relu(inplanes * 8, inplanes)

            self.final_Conv = Conv3Relu(inplanes * 4, inplanes)

            self.fuse = True
        else:
            self.fuse = False

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)


        if "drop" in neck_name:
            rate, size, step = (0.15, 7, 30)
            self.drop = DropBlock(rate=rate, size=size, step=step)
        else:
            self.drop = DropBlock(rate=0, size=0, step=0)

    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats
        #fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4, fx1, fx2, fx3, fx4= ms_feats
        change1_h, change1_w = fa1.size(2), fa1.size(3)
        
        [fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4] = self.drop([fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4])  # dropblock

        fa1 = self.thfab1(fa1)
        fa2 = self.thfab2(fa2)

        fb1 = self.thfab1(fb1)
        fb2 = self.thfab2(fb2)

        if self.expand_field is not None:
            fa4 = self.aspp4(fa4)
            fb4 = self.aspp4(fb4)
            

        change3_2 = self.stage4_Conv_after_up(self.up(fa4))

        fa3 = self.stage3_Conv2(torch.cat([fa3, change3_2], 1))
        change2_2 = self.stage3_Conv_after_up(self.up(fa3))
        change2_2 = self.thfab2(change2_2)

        fa2 = self.stage2_Conv2(torch.cat([fa2, change2_2], 1))    
        change1_2 = self.stage2_Conv_after_up(self.up(fa2))
        change1_2 = self.thfab1(change1_2)

        fa1 = self.stage1_Conv2(torch.cat([fa1, change1_2], 1))

        change3_2 = self.stage4_Conv_after_up(self.up(fb4))
        fb3 = self.stage3_Conv2(torch.cat([fb3, change3_2], 1))
        change2_2 = self.stage3_Conv_after_up(self.up(fb3))
        change2_2 = self.thfab2(change2_2)
        fb2 = self.stage2_Conv2(torch.cat([fb2, change2_2], 1))

        change1_2 = self.stage2_Conv_after_up(self.up(fb2))
        change1_2 = self.thfab1(change1_2)
        fb1 = self.stage1_Conv2(torch.cat([fb1, change1_2], 1))


        mask1a = self.stageabs1_Conv1(torch.abs(fa1-fb1))  # inplanes
        mask2a = self.stageabs2_Conv1(torch.abs(fa2-fb2))  # inplanes * 2
        mask3a = self.stageabs3_Conv1(torch.abs(fa3-fb3))  # inplanes * 4
        mask4a = self.stageabs4_Conv1(torch.abs(fa4-fb4))  # inplanes * 8

        mask2 = self.tanh(mask2a)
        mask1 = self.tanh(mask1a)

       
        m2 = mask2
        m1 = mask1

        change1a = mask1a
        change2a = mask2a
        change3a = mask3a
        change4a = mask4a

        if self.expand_field is not None:
            change4 = self.aspp4(change4a)
            change3a = self.aspp3(change3a)
            #change4 = self.bam4(change4a)
        change3_2 = self.stage4_Conv_after_up(self.up(change4))
        change3 = self.stage3_Conv2(torch.cat([change3a, change3_2], 1))


        change2_2 = self.stage3_Conv_after_up(self.up(change3))
        change2_3 = self.stage3_Conv_after_up2(self.up2(change4))
        change2a = torch.mul(change2a, m2)
        change2_2 = torch.mul(change2_2, m2)
        change2_3 = torch.mul(change2_3, m2)
        change2 = self.stage2_Conv20(torch.cat([change2a, change2_2, change2_3], 1))
 
        change1_2 = self.stage2_Conv_after_up(self.up(change2))
        change1_3 = self.stage2_Conv_after_up3(self.up2(change3))
        change1_4 = self.stage2_Conv_after_up2(self.up3(change4))
        change1a = torch.mul(change1a, m1)
        change1_2 = torch.mul(change1_2, m1)
        change1_2 = torch.mul(change1_3, m1)
        change1_4 = torch.mul(change1_4, m1)

        change1 = self.stage1_Conv20(torch.cat([change1a, change1_2,change1_3,change1_4], 1))

        

        if self.fuse:
            change4 = self.stage4_Conv3(F.interpolate(change4, size=(change1_h, change1_w),
                                                      mode='bilinear', align_corners=True))
            change3 = self.stage3_Conv3(F.interpolate(change3, size=(change1_h, change1_w),
                                                      mode='bilinear', align_corners=True))
            change2 = self.stage2_Conv3(F.interpolate(change2, size=(change1_h, change1_w),
                                                      mode='bilinear', align_corners=True))
            mask2a = self.stage1_Conv3(F.interpolate(mask2a, size=(change1_h, change1_w),
                                                      mode='bilinear', align_corners=True))

            [change1, change2, change3, change4, mask2a, mask1a] = self.drop([change1, change2, change3, change4, mask2a, mask1a])  # dropblock

            change = self.final_Conv(torch.cat([change1, change2, change3, change4], 1))
        else:
            change = change1


        return change, mask1a, mask2a

