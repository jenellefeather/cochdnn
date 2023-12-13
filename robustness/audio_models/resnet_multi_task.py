import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .custom_modules import SequentialWithArgs, FakeReLU

__all__ = ['ResNet', 'resnet_multi_task18', 'resnet_multi_task34', 'resnet_multi_task50', 'resnet_multi_task101',
           'resnet_multi_task152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, fake_relu=False, no_relu=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        pre_out = out.clone()

        if fake_relu:
            return FakeReLU.apply(out)
        if no_relu:
            return out
        return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, fake_relu=False, no_relu=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if fake_relu:
            return FakeReLU.apply(out)
        if no_relu:
            return out

        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if isinstance(num_classes, dict): # Make multiple fully conected layers
            all_fc_layers = {}
            for task in num_classes.keys():
                all_fc_layers[task] = nn.Linear(512 * block.expansion, num_classes[task]) 
            self.fc = nn.ModuleDict(all_fc_layers)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return SequentialWithArgs(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        if with_latent:
            all_outputs = {}
            all_outputs['input_after_preproc'] = x
        x = self.conv1(x)
        if with_latent:
            all_outputs['conv1'] = x 
        x = self.bn1(x) 
        if with_latent:
            all_outputs['bn1'] = x
        x = self.relu(x)
        if with_latent:
            all_outputs['conv1_relu1'] = x
        x = self.maxpool(x)
        if with_latent:
            all_outputs['maxpool1'] = x

        x = self.layer1(x)
        if with_latent:
            all_outputs['layer1'] = x
        x = self.layer2(x)
        if with_latent:
            all_outputs['layer2'] = x
        x = self.layer3(x)
        if with_latent:
            all_outputs['layer3'] = x
        x = self.layer4(x, fake_relu=fake_relu, no_relu=no_relu)
        if with_latent:
            all_outputs['layer4'] = x

        x = self.avgpool(x)
        if with_latent:
            all_outputs['avgpool'] = x
        pre_out = x.view(x.size(0), -1)
        if isinstance(self.fc, nn.ModuleDict): 
            final = {}
            for task, fc_l in self.fc.items():
                final[task] = fc_l(pre_out)
                if with_latent:
                    all_outputs['final/'+task] = final[task]
        else:
            final = self.fc(pre_out)

        if with_latent:
            all_outputs['final'] = final
        
        if with_latent:
            return final, pre_out, all_outputs
        return final

def resnet_multi_task18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet_multi_task18']))
    return model


def resnet_multi_task34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet_multi_task34']))
    return model


def resnet_multi_task50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet_multi_task50']))
    return model


def resnet_multi_task101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet_multi_task101']))
    return model


def resnet_multi_task152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet_multi_task152']))
    return model
