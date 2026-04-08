from models.layers import *
from models.TCMMAC import TCMMAC

class MS_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, time_step=6, TCMMAC_ON=True, dvs=None, NA_ON=True, XA_ON=True, Chao_ON=True):
        super(MS_ResNet, self).__init__()
        
        self.dvs = dvs

        self.T = time_step # time-step
        norm_layer = tdBatchNorm
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        self.groups = 1
        self.base_width = 64
        if self.dvs is True: 
            self.input_conv = tdLayer(nn.Conv2d(2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), 
                                    norm_layer(self.inplanes))
        else:
            self.input_conv = tdLayer(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), 
                                  norm_layer(self.inplanes))

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = tdLayer(nn.Linear(512 * block.expansion, num_classes))
        self.LIF = LIFSpike()
        
        if TCMMAC_ON==True:
            self.encoding = TCMMAC(T=self.T, out_channels=64, enable_na=NA_ON, enable_xa=XA_ON, enable_chao=Chao_ON)
        else: 
            self.encoding = None
            
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tdLayer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # TCMMAC ON or OFF
        if self.encoding is not None:
            x = self.input_conv(x)
            img = x
            x = self.LIF(x)
            x = self.encoding(img,x)
        else:
            x = self.input_conv(x)
    

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.LIF(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x

    def forward(self, x):
        if self.dvs is True: 
            return self._forward_impl(x)    
        
        else: 
            x = add_dimention(x, self.T) 
            return self._forward_impl(x)


class MS_ResNet_34(nn.Module):
    def __init__(self, block, layers, num_classes=1000, time_step=6, TCMMAC_ON=True, NA_ON=True, XA_ON=True, Chao_ON=True,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(MS_ResNet_34, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.T = time_step
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)  

        self.bn1 = norm_layer(self.inplanes)
        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc1_s = tdLayer(nn.Linear(512 * block.expansion, num_classes))
        self.LIF = LIFSpike()
        
        if TCMMAC_ON==True:
            self.encoding = TCMMAC(T=self.T , out_channels = 64, enable_na=NA_ON, enable_xa=XA_ON, enable_chao=Chao_ON)
        else: 
            self.encoding = None

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tdLayer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = x.repeat(self.T, 1, 1, 1, 1).permute(1,0,2,3,4)
        # TCMMAC ON or OFF
        if self.encoding is not None:
            x = self.conv1_s(x)
            img = x
            x = self.LIF(x)
            x = self.encoding(img,x)
        else:
            x = self.conv1_s(x)

        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.LIF(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc1_s(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

def ms_resnet(block, layers, **kwargs):
    model = MS_ResNet(block, layers, **kwargs)
    return model

def ms_resnet_34(block, layers, **kwargs):
    model = MS_ResNet_34(block, layers, **kwargs)
    return model 

def TCMMAC_SNN_18(**kwargs):
    return ms_resnet(BasicBlock_MS, [3, 3, 2],
                   **kwargs)
def TCMMAC_SNN_34(**kwargs):
    return ms_resnet_34(BasicBlock_MS, [3, 4, 6, 3],
                    **kwargs)  
