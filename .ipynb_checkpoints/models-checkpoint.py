"""
Dual-Region Network (DRN) Model Architecture with Ablation Support.
"""

import math
import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """Factory function for activation functions."""
    activations = {
        'leakyrelu': nn.LeakyReLU(0.1, inplace=True),
        'relu': nn.ReLU(inplace=True),
        'gelu': nn.GELU(),
        'elu': nn.ELU(inplace=True),
        'silu': nn.SiLU(inplace=True),
    }
    return activations.get(name, nn.LeakyReLU(0.1, inplace=True))


def _after_maxpool2(L_in):
    return math.floor((L_in - 2) / 2 + 1)


def _after_stride2_k3p1(L_in):
    return math.floor((L_in + 2 - 2) / 2 + 1)


class Identity(nn.Module):
    def forward(self, x):
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False))
        self.max_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.shape
        avg = self.avg_pool(x).view(b, c)
        mx = self.max_pool(x).view(b, c)
        out = self.avg_mlp(avg) + self.max_mlp(mx)
        return x * self.sigmoid(out).view(b, c, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 8, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(8)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(8, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg, mx], dim=1)
        y = self.relu(self.bn(self.conv1(y)))
        return x * self.sigmoid(self.conv2(y))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_skip=True,
                 use_channel_attention=True, activation='leakyrelu'):
        super().__init__()
        self.use_skip = use_skip
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.act1 = get_activation(activation)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = get_activation(activation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.ca = ChannelAttention(out_channels) if use_channel_attention else Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        identity = x
        out = self.conv1(self.act1(self.bn1(x)))
        out = self.conv2(self.act2(self.bn2(out)))
        out = self.ca(out)
        if self.use_skip:
            out = out + self.shortcut(identity)
        return out


class RegionSpecificNet(nn.Module):
    def __init__(self, num_channels, num_bands, *, dropout_rate=0.25, base_channels=64,
                 max_channels=512, num_residual_blocks=None, max_residual_blocks_cap=8,
                 min_len_after_blocks=1, use_channel_attention=True, use_spatial_attention=True,
                 use_residual_connections=True, activation='leakyrelu'):
        super().__init__()
        
        self.channel_attention = ChannelAttention(num_channels) if use_channel_attention else Identity()
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_channels, base_channels, 5, 1, 2, bias=False),
            nn.BatchNorm1d(base_channels), get_activation(activation),
            nn.Conv1d(base_channels, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(base_channels), get_activation(activation),
            nn.MaxPool1d(2, 2))

        L = _after_maxpool2(num_bands)
        planned_blocks = 0
        ch = base_channels
        if num_residual_blocks is None:
            while planned_blocks < max_residual_blocks_cap and ch * 2 <= max_channels:
                L_next = _after_stride2_k3p1(L)
                if L_next < min_len_after_blocks:
                    break
                planned_blocks += 1
                L, ch = L_next, ch * 2
        else:
            planned_blocks = int(num_residual_blocks)
            for _ in range(planned_blocks):
                if ch * 2 <= max_channels:
                    ch *= 2

        self.num_residual_blocks = planned_blocks
        self.last_channels = base_channels * (2 ** planned_blocks)

        blocks = []
        in_c = base_channels
        for _ in range(planned_blocks):
            out_c = in_c * 2
            blocks.append(ResidualBlock(in_c, out_c, 2, use_residual_connections,
                                        use_channel_attention, activation))
            in_c = out_c
        self.residual_blocks = nn.Sequential(*blocks)

        self.spatial_attn = SpatialAttention() if use_spatial_attention else Identity()
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.global_maxpool = nn.AdaptiveMaxPool1d(1)

        self.feature_size = 2 * self.last_channels
        hidden_fp = self.feature_size // 2
        self.output_dim = 4 * hidden_fp

        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_size, hidden_fp), nn.BatchNorm1d(hidden_fp),
            get_activation(activation), nn.Dropout(dropout_rate),
            nn.Linear(hidden_fp, self.output_dim), nn.BatchNorm1d(self.output_dim),
            get_activation(activation), nn.Dropout(dropout_rate * 0.5))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.conv1(x)
        if self.num_residual_blocks > 0:
            x = self.residual_blocks(x)
        x = self.spatial_attn(x)
        avg = self.global_avgpool(x).view(x.size(0), -1)
        mx = self.global_maxpool(x).view(x.size(0), -1)
        return self.feature_projection(torch.cat([avg, mx], dim=1))


class DualRegionNet(nn.Module):
    def __init__(self, ob_channels, ob_bands, pcx_channels, pcx_bands, num_classes, *,
                 dropout_rate=0.25, base_channels=64, max_channels=512,
                 max_residual_blocks_cap=8, min_len_after_blocks=1,
                 use_channel_attention=True, use_spatial_attention=True,
                 use_fusion_attention=True, use_skip_classifier=True,
                 use_residual_connections=True, activation='leakyrelu'):
        super().__init__()
        self.use_fusion_attention = use_fusion_attention
        self.use_skip_classifier = use_skip_classifier

        def plan_blocks(num_bands):
            L = _after_maxpool2(num_bands)
            ch, blocks = base_channels, 0
            while blocks < max_residual_blocks_cap and ch * 2 <= max_channels:
                L_next = _after_stride2_k3p1(L)
                if L_next < min_len_after_blocks:
                    break
                blocks += 1
                ch *= 2
                L = L_next
            return blocks

        shared_blocks = min(plan_blocks(ob_bands), plan_blocks(pcx_bands))
        rsn_output_dim = 4 * (base_channels * (2 ** shared_blocks))

        self.ob_net = RegionSpecificNet(
            ob_channels, ob_bands, dropout_rate=dropout_rate, base_channels=base_channels,
            max_channels=max_channels, num_residual_blocks=shared_blocks,
            use_channel_attention=use_channel_attention, use_spatial_attention=use_spatial_attention,
            use_residual_connections=use_residual_connections, activation=activation)
        self.pcx_net = RegionSpecificNet(
            pcx_channels, pcx_bands, dropout_rate=dropout_rate, base_channels=base_channels,
            max_channels=max_channels, num_residual_blocks=shared_blocks,
            use_channel_attention=use_channel_attention, use_spatial_attention=use_spatial_attention,
            use_residual_connections=use_residual_connections, activation=activation)

        self.feature_dim = rsn_output_dim
        if use_fusion_attention:
            self.fusion_attention = nn.Sequential(
                nn.Linear(2 * self.feature_dim, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 2), nn.Softmax(dim=1))
        else:
            self.fusion_attention = None

        hidden1 = int(max(64, min(1024, self.feature_dim // 2)))
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden1), nn.BatchNorm1d(hidden1),
            get_activation(activation), nn.Dropout(dropout_rate),
            nn.Linear(hidden1, 32), nn.BatchNorm1d(32),
            get_activation(activation), nn.Dropout(dropout_rate),
            nn.Linear(32, 16), nn.BatchNorm1d(16),
            get_activation(activation), nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes))

        self.skip = nn.Linear(self.feature_dim, num_classes) if use_skip_classifier else None

    def forward(self, ob_x, pcx_x):
        ob_feat = self.ob_net(ob_x)
        pcx_feat = self.pcx_net(pcx_x)

        if self.use_fusion_attention and self.fusion_attention is not None:
            w = self.fusion_attention(torch.cat([ob_feat, pcx_feat], dim=1))
            combined = w[:, 0:1] * ob_feat + w[:, 1:2] * pcx_feat
        else:
            combined = 0.5 * ob_feat + 0.5 * pcx_feat

        logits = self.classifier(combined)
        if self.use_skip_classifier and self.skip is not None:
            logits = logits + self.skip(combined)
        return logits
