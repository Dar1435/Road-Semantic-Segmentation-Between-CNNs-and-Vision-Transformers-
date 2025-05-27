import torch
import torch.nn as nn
import torch.nn.functional as F

# ConvNeXt Architecture Configurations
ConvNeXt_Archs = {
    'Tiny': [[3, 3, 9, 3], [96, 192, 384, 768], 0.4, 1.0],  # [Depths, FeatureDimensions, DropPathRate, LayerScale]
    'Small': [[3, 3, 27, 3], [96, 192, 384, 768], 0.4, 1.0],
    'Base': [[3, 3, 27, 3], [128, 256, 512, 1024], 0.4, 1.0],
    'Large': [[3, 3, 27, 3], [192, 384, 768, 1536], 0.4, 1.0],
    'XLarge': [[3, 3, 27, 3], [256, 512, 1024, 2048], 0.4, 1.0]
}

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class LayerNorm(nn.Module):
    def __init__(self, _shape, eps=1e-6, _format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(_shape))
        self.bias = nn.Parameter(torch.zeros(_shape))
        self.eps = eps
        self._format = _format
        self._shape = (_shape, )

    def forward(self, x):
        if self._format == "channels_last":
            return F.layer_norm(x, self._shape, self.weight, self.bias, self.eps)
        elif self._format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class PatchifyStem(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4),
            LayerNorm(out_channels, eps=1e-6, _format="channels_first")
        )

class DownsamplingConv(nn.Sequential):
    def __init__(self, feature_dimensions, i):
        super().__init__(
            LayerNorm(feature_dimensions[i], eps=1e-6, _format="channels_first"),
            nn.Conv2d(feature_dimensions[i], feature_dimensions[i+1], kernel_size=2, stride=2)
        )

class DropPath(nn.Module):
    def __init__(self, drop_path_rate, training):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.training = training

    def forward(self, x):
        if self.drop_path_rate == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor

class ConvNeXtBlock(nn.Module):
    def __init__(self, feature_dim, expansion, drop_path_rate, layer_scale_initial, training):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim)) * layer_scale_initial, requires_grad=True) if layer_scale_initial > 0 else None
        self.drop_path = DropPath(drop_path_rate, training) if drop_path_rate > 0.0 else nn.Identity()
        self.conv_next_block = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=7, padding=3, groups=feature_dim),
            Permute([0, 2, 3, 1]),  # Replaced nn.Permute with custom Permute
            LayerNorm(feature_dim, eps=1e-6),
            nn.Linear(feature_dim, expansion * feature_dim),
            nn.GELU(),
            nn.Linear(expansion * feature_dim, feature_dim)
        )

    def forward(self, x):
        input_x = x
        x = self.conv_next_block(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        x = self.drop_path(x) + input_x
        return x

class ConvNeXtStage(nn.Sequential):
    def __init__(self, depth, feature_dim, expansion, drop_index, drop_path_rates, layer_scale_initial, training):
        super().__init__(*[
            ConvNeXtBlock(feature_dim, expansion, drop_path_rates[drop_index + j], layer_scale_initial, training)
            for j in range(depth)
        ])

class Encoder(nn.Module):
    def __init__(self, in_channels, depths, feature_dims, stochastic_depth_rate, layer_scale_initial, training):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(PatchifyStem(in_channels, feature_dims[0]))
        for i in range(3):
            self.downsample_layers.append(DownsamplingConv(feature_dims, i))
        
        self.feature_resolution_stages = nn.ModuleList()
        self.drop_path_rates = [x.item() for x in torch.linspace(0, stochastic_depth_rate, sum(depths))]
        drop_index = 0
        for i in range(len(feature_dims)):
            self.feature_resolution_stages.append(
                ConvNeXtStage(depths[i], feature_dims[i], 4, drop_index, self.drop_path_rates, layer_scale_initial, training)
            )
            drop_index += depths[i]
        
        for i in range(len(feature_dims)):
            self.add_module(f"LN{i}", LayerNorm(feature_dims[i], _format="channels_first"))

    def forward(self, x):
        outputs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.feature_resolution_stages[i](x)
            norm_layer = getattr(self, f"LN{i}")
            outputs.append(norm_layer(x))
        return outputs

class CBA(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

class Decoder(nn.Module):
    def __init__(self, in_channels, fpn_dim, num_classes=2, scales=(1, 2, 3, 6), input_size=256):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Pyramid Pooling Module (PPM)
        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                CBA(in_channels[-1], fpn_dim, 1)
            ) for scale in scales
        ])
        self.ppm_bottleneck = CBA(in_channels[-1] + fpn_dim * len(scales), fpn_dim, 3, 1, 1)
        
        # FPN input and output layers
        self.fpn_in = nn.ModuleList([CBA(in_ch, fpn_dim, 1) for in_ch in in_channels[:-1]])
        self.fpn_out = nn.ModuleList([CBA(fpn_dim, fpn_dim, 3, 1, 1) for _ in in_channels[:-1]])
        
        # Lateral convolutions for feature fusion
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(True)
            ) for _ in range(4)  # For P2, P3, P4, P5
        ])
        
        # Final fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fpn_dim * 4, fpn_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(True)
        )
        
        # Output convolution for road segmentation
        self.conv_out = nn.Conv2d(fpn_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features):
        # PPM on C5
        ppm_features = [features[-1]] + [
            F.interpolate(stage(features[-1]), size=features[-1].shape[-2:], mode='bilinear', align_corners=True)
            for stage in self.ppm
        ][::-1]
        fpn_feature = self.ppm_bottleneck(torch.cat(ppm_features, dim=1))
        
        # FPN: Generate P4, P3, P2
        fpn_features = [fpn_feature]  # P5
        for i in reversed(range(len(features)-1)):
            feature = self.fpn_in[i](features[i])
            fpn_feature = feature + F.interpolate(
                fpn_feature, size=feature.shape[-2:], mode='bilinear', align_corners=False
            )
            fpn_features.append(self.fpn_out[i](fpn_feature))
        
        fpn_features.reverse()  # [P2, P3, P4, P5]
        
        # UPerNet: Process each feature map and upsample for fusion
        fused_features = []
        for i, (feature, lateral_conv) in enumerate(zip(fpn_features, self.lateral_convs)):
            # Process feature with 3x3 conv
            feature = lateral_conv(feature)
            # Upsample to P2's size (64x64 for 256x256 input)
            if i > 0:  # P2 is already at 64x64
                feature = F.interpolate(
                    feature, size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=False
                )
            fused_features.append(feature)
        
        # Concatenate and fuse
        fused = torch.cat(fused_features, dim=1)
        fused = self.fusion_conv(fused)
        
        # Final output: 3x3 conv and upsample to input size (256x256) with nearest neighbor
        output = self.conv_out(self.dropout(fused))
        output = F.interpolate(
            output, size=(self.input_size, self.input_size),
            mode='nearest'
        )
        
        return output

class ConvNeXtRoadSegmentation(nn.Module):
    def __init__(self, model_arch="Base", num_classes=2, input_size=256):
        super().__init__()
        assert model_arch in ConvNeXt_Archs, f"Model architecture {model_arch} not supported."
        depths, feature_dims, stochastic_depth_rate, layer_scale_initial = ConvNeXt_Archs[model_arch]
        
        self.encoder = Encoder(3, depths, feature_dims, stochastic_depth_rate, layer_scale_initial, self.training)
        self.decoder = Decoder(feature_dims, feature_dims[0], num_classes, input_size=input_size)

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def dice_loss(self, pred, target, smooth=1):
        pred = torch.softmax(pred, dim=1)[:, 1]
        target = target.float()
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice