import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class unet(nn.Module):

    def __init__(self):
        super(unet, self).__init__()

        # Downsampling path
        self.con1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.con1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d(2, stride=2)

        self.con2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.con2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.mp2 = nn.MaxPool2d(2, stride=2)

        self.con3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.con3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.mp3 = nn.MaxPool2d(2, stride=2)

        self.con4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.con4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)
        self.mp4 = nn.MaxPool2d(2, stride=2)

        self.con5_1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.con5_2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(1024)
        self.bn10 = nn.BatchNorm2d(1024)
        self.mp5 = nn.MaxPool2d(2, stride=2)

        # bottleneck path
        self.con6_1 = nn.Conv2d(1024, 2048, 3, padding = 1)
        self.con6_2 = nn.Conv2d(2048, 2048, 3, padding = 1)
        self.bn11 = nn.BatchNorm2d(2048)
        self.bn12 = nn.BatchNorm2d(2048)
        # self.dropout1 = nn.Dropout(p=0.5)
        # self.dropout2 = nn.Dropout(p=0.5)

        # self.fcn1 = nn.Linear(in_features=2048 * 9 * 13, out_features=256)
        # self.bn_fcn1 = nn.BatchNorm1d(256)
        # self.fcn2 = nn.Linear(in_features=256, out_features=256)
        # self.bn_fcn2 = nn.BatchNorm1d(256)
        # self.fcn3 = nn.Linear(in_features=256, out_features=1)

        # Upsampling path
        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size = 2, stride= 2)
        self.con7_1 = nn.Conv2d(2048, 1024, 3, padding = 1)
        self.con7_2 = nn.Conv2d(1024, 1024, 3, padding = 1)
        self.bn13 = nn.BatchNorm2d(1024)
        self.bn14 = nn.BatchNorm2d(1024)

        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride = 2)
        self.con8_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.con8_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn15 = nn.BatchNorm2d(512)
        self.bn16 = nn.BatchNorm2d(512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride = 2)
        self.con9_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.con9_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn17 = nn.BatchNorm2d(256)
        self.bn18 = nn.BatchNorm2d(256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride = 2)
        self.con10_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.con10_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn19 = nn.BatchNorm2d(128)
        self.bn20 = nn.BatchNorm2d(128)

        self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride = 2)
        self.con11_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.con11_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn21 = nn.BatchNorm2d(64)
        self.bn22 = nn.BatchNorm2d(64)

        self.con12 = nn.Conv2d(64, 1, 1)

    def forward(self, image):

        # Downsampling
        fp1 = F.relu(self.bn2(self.con1_2(F.relu(self.bn1(self.con1_1(image))))))
        down1 = self.mp1(fp1)

        fp2 = F.relu(self.bn4(self.con2_2(F.relu(self.bn3(self.con2_1(down1))))))
        down2 = self.mp2(fp2)

        fp3 = F.relu(self.bn6(self.con3_2(F.relu(self.bn5(self.con3_1(down2))))))
        down3 = self.mp3(fp3)

        fp4 = F.relu(self.bn8(self.con4_2(F.relu(self.bn7(self.con4_1(down3))))))
        down4 = self.mp4(fp4)

        fp5 = F.relu(self.bn10(self.con5_2(F.relu(self.bn9(self.con5_1(down4))))))
        down5 = self.mp5(fp5)

        # bottleneck
        fp6 = F.relu(self.bn12(self.con6_2(F.relu(self.bn11(self.con6_1(down5))))))
        #
        # fc1 = F.relu(self.bn_fcn1(self.fcn1(fp6.view(-1, fp6.shape[1] * fp6.shape[2] * fp6.shape[3]))))
        # fc2 = F.relu(self.bn_fcn2(self.fcn2(fc1)))
        # fc3 = F.relu(self.fcn3(fc2))
        #
        # Upsampling
        up1 = self.up1(fp6)
        fp7 = F.relu(self.bn14(self.con7_2(F.relu(self.bn13(self.con7_1(torch.cat((fp5, up1), 1)))))))

        up2 = self.up2(fp7)
        fp8 = F.relu(self.bn16(self.con8_2(F.relu(self.bn15(self.con8_1(torch.cat((fp4, up2), 1)))))))

        up3 = self.up3(fp8)
        fp9 = F.relu(self.bn18(self.con9_2(F.relu(self.bn17(self.con9_1(torch.cat((fp3, up3), 1)))))))

        up4 = self.up4(fp9)
        fp10 = F.relu(self.bn20(self.con10_2(F.relu(self.bn19(self.con10_1(torch.cat((fp2, up4), 1)))))))

        up5 = self.up5(fp10)
        fp11 = F.relu(self.bn22(self.con11_2(F.relu(self.bn21(self.con11_1(torch.cat((fp1, up5), 1)))))))

        output = torch.sigmoid(self.con12(fp11))

        return output
