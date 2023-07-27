import torch
import torch.nn as nn
import torch.nn.functional as F
######################2D Models#########################################################################################

#####################2D Blocks Definition######################################
class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, batch_norm=True):
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.batch_norm = batch_norm
        self.dropout_p = dropout
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.bn_shortcut = nn.BatchNorm2d(out_channels)
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        if self.batch_norm:
            out = self.bn1(out)
        out = F.relu(self.conv2(out))
        if self.batch_norm:
            out = self.bn2(out)
        if self.dropout_p > 0:
            out = self.dropout(out)
        shortcut = self.shortcut(x)
        if self.batch_norm:
            shortcut = self.bn_shortcut(shortcut)
        return F.relu(out + shortcut)

class GatingSignal(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(GatingSignal, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.conv(x))
        if self.batch_norm:
            out = self.bn(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.theta_x = nn.Conv2d(in_channels, inter_channels, kernel_size=2, stride=2, padding=0)
        self.phi_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, padding=0)
        self.psi_channel = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.result = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.result_bn = nn.BatchNorm2d(in_channels)

    def forward(self, x, g):
        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)
        upsample_g = F.interpolate(phi_g, size=(theta_x.size(2), theta_x.size(3)))
        concat_xg = upsample_g + theta_x
        act_xg = F.relu(concat_xg)
        psi = self.psi(act_xg)
        sigmoid_xg = torch.sigmoid(psi)
        upsample_psi = F.interpolate(sigmoid_xg, size=(x.size(2), x.size(3)))
        psi_channel = self.psi_channel(x)
        sigmoid_channel = torch.sigmoid(psi_channel)
        y_channel = sigmoid_channel * x
        y = upsample_psi * y_channel
        result = self.result(y)
        result_bn = self.result_bn(result)
        return result_bn
################################2D Attention Gate ResU-Net#######################
class AttentionResUNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(AttentionResUNet, self).__init__()

        self.encoder1 = ResConvBlock(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.encoder2 = ResConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.encoder3 = ResConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.encoder4 = ResConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.center = ResConvBlock(512, 1024)

        # Attention Blocks
        self.attention_block4 = AttentionBlock(512, 512, 512)
        self.attention_block3 = AttentionBlock(256, 256, 256)
        self.attention_block2 = AttentionBlock(128, 128, 128)
        self.attention_block1 = AttentionBlock(64, 64, 64)

        # Gating Signal
        self.gating_signal4 = GatingSignal(1024, 512)
        self.gating_signal3 = GatingSignal(512, 256)
        self.gating_signal2 = GatingSignal(256, 128)
        self.gating_signal1 = GatingSignal(128, 64)

        self.decoder4 = ResConvBlock(1024, 512)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        self.decoder3 = ResConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        self.decoder2 = ResConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.decoder1 = ResConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        # Center
        c = self.center(self.pool4(e4))

        # Decoder with Attention Blocks
        g4 = self.gating_signal4(c)
        a4 = self.attention_block4(e4, g4)
        d4 = self.up4(c)
        d4 = torch.cat((a4, d4), dim=1)
        d4 = self.decoder4(d4)

        g3 = self.gating_signal3(d4)
        a3 = self.attention_block3(e3, g3)
        d3 = self.up3(d4)
        d3 = torch.cat((a3, d3), dim=1)
        d3 = self.decoder3(d3)

        g2 = self.gating_signal2(d3)
        a2 = self.attention_block2(e2, g2)
        d2 = self.up2(d3)
        d2 = torch.cat((a2, d2), dim=1)
        d2 = self.decoder2(d2)

        g1 = self.gating_signal1(d2)
        a1 = self.attention_block1(e1, g1)
        d1 = self.up1(d2)
        d1 = torch.cat((a1, d1), dim=1)
        d1 = self.decoder1(d1)

        out = torch.sigmoid(self.final(d1))

        return out

###################################3D Models############################################################################
#######################3D Blocks Definition#####################################
class ThreeDResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0, batch_norm=False):
        super(ThreeDResConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.batch_norm = batch_norm
        self.dropout_p = dropout
        if self.batch_norm:
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.bn_shortcut = nn.BatchNorm3d(out_channels)
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        if self.batch_norm:
            out = self.bn1(out)
        out = F.relu(self.conv2(out))
        if self.batch_norm:
            out = self.bn2(out)
        if self.dropout_p > 0:
            out = self.dropout(out)
        shortcut = self.shortcut(x)
        if self.batch_norm:
            shortcut = self.bn_shortcut(shortcut)
        return F.relu(out + shortcut)


class ThreeDGatingSignal(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super(ThreeDGatingSignal, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = F.relu(self.conv(x))
        if self.batch_norm:
            out = self.bn(out)
        return out

class ThreeDAttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(ThreeDAttentionBlock, self).__init__()
        self.theta_x = nn.Conv3d(in_channels, inter_channels, kernel_size=2, stride=2, padding=0)
        self.phi_g = nn.Conv3d(gating_channels, inter_channels, kernel_size=1, padding=0)
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1, padding=0)
        self.psi_channel = nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        self.result = nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        self.result_bn = nn.BatchNorm3d(in_channels)

    def forward(self, x, g):
        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)
        upsample_g = F.interpolate(phi_g, size=(theta_x.size(2), theta_x.size(3), theta_x.size(4)), mode='trilinear', align_corners=False)
        concat_xg = upsample_g + theta_x
        act_xg = F.relu(concat_xg)
        psi = self.psi(act_xg)
        sigmoid_xg = torch.sigmoid(psi)
        upsample_psi = F.interpolate(sigmoid_xg, size=(x.size(2), x.size(3), x.size(4)), mode='trilinear', align_corners=False)
        psi_channel = self.psi_channel(x)
        sigmoid_channel = torch.sigmoid(psi_channel)
        y_channel = sigmoid_channel * x
        y = upsample_psi * y_channel
        result = self.result(y)
        result_bn = self.result_bn(result)
        return result_bn


class ThreeDAttentionResUNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ThreeDAttentionResUNet, self).__init__()

        self.encoder1 = ThreeDResConvBlock(input_channels, 64)
        self.pool1 = nn.MaxPool3d(2, 2)

        self.encoder2 = ThreeDResConvBlock(64, 128)
        self.pool2 = nn.MaxPool3d(2, 2)

        self.encoder3 = ThreeDResConvBlock(128, 256)
        self.pool3 = nn.MaxPool3d(2, 2)

        self.encoder4 = ThreeDResConvBlock(256, 512)
        self.pool4 = nn.MaxPool3d(2, 2)

        self.center = ThreeDResConvBlock(512, 1024)

        # Attention Blocks
        self.attention_block4 = ThreeDAttentionBlock(512, 512, 512)
        self.attention_block3 = ThreeDAttentionBlock(256, 256, 256)
        self.attention_block2 = ThreeDAttentionBlock(128, 128, 128)
        self.attention_block1 = ThreeDAttentionBlock(64, 64, 64)

        # Gating Signal
        self.gating_signal4 = ThreeDGatingSignal(1024, 512)
        self.gating_signal3 = ThreeDGatingSignal(512, 256)
        self.gating_signal2 = ThreeDGatingSignal(256, 128)
        self.gating_signal1 = ThreeDGatingSignal(128, 64)

        self.decoder4 = ThreeDResConvBlock(1024, 512)
        self.up4 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)

        self.decoder3 = ThreeDResConvBlock(512, 256)
        self.up3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)

        self.decoder2 = ThreeDResConvBlock(256, 128)
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)

        self.decoder1 = ThreeDResConvBlock(128, 64)
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)

        self.final = nn.Conv3d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        # Center
        c = self.center(self.pool4(e4))

        # Decoder with Attention Blocks
        g4 = self.gating_signal4(c)
        a4 = self.attention_block4(e4, g4)
        d4 = self.up4(c)

        d4 = torch.cat((a4, d4), dim=1)
        d4 = self.decoder4(d4)

        g3 = self.gating_signal3(d4)
        a3 = self.attention_block3(e3, g3)
        d3 = self.up3(d4)
        d3 = torch.cat((a3, d3), dim=1)
        d3 = self.decoder3(d3)

        g2 = self.gating_signal2(d3)
        a2 = self.attention_block2(e2, g2)
        d2 = self.up2(d3)
        d2 = torch.cat((a2, d2), dim=1)
        d2 = self.decoder2(d2)

        g1 = self.gating_signal1(d2)
        a1 = self.attention_block1(e1, g1)
        d1 = self.up1(d2)
        d1 = torch.cat((a1, d1), dim=1)
        d1 = self.decoder1(d1)

        # Final layer
        out = self.final(d1)
        return torch.sigmoid(out)

