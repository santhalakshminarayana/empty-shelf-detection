import torch
import torch.nn as nn
from torchsummary import summary

mobile_net = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True)
# _ = mobile_net.eval()
for param in mobile_net.parameters():
    param.required_grad = False

'''
summary(mobile_net.to(device), (3, 512, 512))
'''

def weight_initialize(mob_net_v2, mobile_net):
    curr_bl = 0
    # phase - 1
    for i in range(len(mob_net_v2.phase_1)):
        phase_type = mob_net_v2.phase_1_type[i]
        if phase_type == 'convbnrelu':
            # order conv, bn and relu
            # conv - weight
            mob_net_v2.phase_1[i][0].weight.data.copy_(mobile_net.features[curr_bl][0].weight.data)
            # bn - weight, bias, running_mean, running_var
            mob_net_v2.phase_1[i][1].weight.data.copy_(mobile_net.features[curr_bl][1].weight.data)
            mob_net_v2.phase_1[i][1].bias.data.copy_(mobile_net.features[curr_bl][1].bias.data)
            mob_net_v2.phase_1[i][1].running_mean.data.copy_(mobile_net.features[curr_bl][1].running_mean.data)
            mob_net_v2.phase_1[i][1].running_var.data.copy_(mobile_net.features[curr_bl][1].running_var.data)

        if phase_type == 'inv_res_1':
            # first move to inv_res block
            # 0 is convbnrelu, 1 is conv and 2 is bn
            # order conv, bn and relu

            # 0 - convbnrelu
            # conv - weight
            mob_net_v2.phase_1[i].inv_res[0][0].weight.data.copy_(mobile_net.features[curr_bl].conv[0][0].weight.data)
            # bn - weight, bias, running_mean, running_var
            mob_net_v2.phase_1[i].inv_res[0][1].weight.data.copy_(mobile_net.features[curr_bl].conv[0][1].weight.data)
            mob_net_v2.phase_1[i].inv_res[0][1].bias.data.copy_(mobile_net.features[curr_bl].conv[0][1].bias.data)
            mob_net_v2.phase_1[i].inv_res[0][1].running_mean.data.copy_(mobile_net.features[curr_bl].conv[0][1].running_mean.data)
            mob_net_v2.phase_1[i].inv_res[0][1].running_var.data.copy_(mobile_net.features[curr_bl].conv[0][1].running_var.data)

            # 1 - conv
            mob_net_v2.phase_1[i].inv_res[1].weight.data.copy_(mobile_net.features[curr_bl].conv[1].weight.data)

            # 2 - bn
            mob_net_v2.phase_1[i].inv_res[2].weight.data.copy_(mobile_net.features[curr_bl].conv[2].weight.data)
            mob_net_v2.phase_1[i].inv_res[2].bias.data.copy_(mobile_net.features[curr_bl].conv[2].bias.data)
            mob_net_v2.phase_1[i].inv_res[2].running_mean.data.copy_(mobile_net.features[curr_bl].conv[2].running_mean.data)
            mob_net_v2.phase_1[i].inv_res[2].running_var.data.copy_(mobile_net.features[curr_bl].conv[2].running_var.data)

        if phase_type == 'inv_res_2':
            # first move to inv_res block
            # 0 is convbnrelu, 1 is convrbnrelu, 2 is conv, 3 is bn
            
            # 0 - convbnrelu
            # conv - weight
            mob_net_v2.phase_1[i].inv_res[0][0].weight.data.copy_(mobile_net.features[curr_bl].conv[0][0].weight.data)
            # bn - weight, bias, running_mean, running_var
            mob_net_v2.phase_1[i].inv_res[0][1].weight.data.copy_(mobile_net.features[curr_bl].conv[0][1].weight.data)
            mob_net_v2.phase_1[i].inv_res[0][1].bias.data.copy_(mobile_net.features[curr_bl].conv[0][1].bias.data)
            mob_net_v2.phase_1[i].inv_res[0][1].running_mean.data.copy_(mobile_net.features[curr_bl].conv[0][1].running_mean.data)
            mob_net_v2.phase_1[i].inv_res[0][1].running_var.data.copy_(mobile_net.features[curr_bl].conv[0][1].running_var.data)
            

            # 1 - convbnrelu
            # conv - weight
            mob_net_v2.phase_1[i].inv_res[1][0].weight.data.copy_(mobile_net.features[curr_bl].conv[1][0].weight.data)
            # bn - weight, bias, running_mean, running_var
            mob_net_v2.phase_1[i].inv_res[1][1].weight.data.copy_(mobile_net.features[curr_bl].conv[1][1].weight.data)
            mob_net_v2.phase_1[i].inv_res[1][1].bias.data.copy_(mobile_net.features[curr_bl].conv[1][1].bias.data)
            mob_net_v2.phase_1[i].inv_res[1][1].running_mean.data.copy_(mobile_net.features[curr_bl].conv[1][1].running_mean.data)
            mob_net_v2.phase_1[i].inv_res[1][1].running_var.data.copy_(mobile_net.features[curr_bl].conv[1][1].running_var.data)

            # 2 - conv
            mob_net_v2.phase_1[i].inv_res[2].weight.data.copy_(mobile_net.features[curr_bl].conv[2].weight.data)

            # 3 - bn
            mob_net_v2.phase_1[i].inv_res[3].weight.data.copy_(mobile_net.features[curr_bl].conv[3].weight.data)
            mob_net_v2.phase_1[i].inv_res[3].bias.data.copy_(mobile_net.features[curr_bl].conv[3].bias.data)
            mob_net_v2.phase_1[i].inv_res[3].running_mean.data.copy_(mobile_net.features[curr_bl].conv[3].running_mean.data)
            mob_net_v2.phase_1[i].inv_res[3].running_var.data.copy_(mobile_net.features[curr_bl].conv[3].running_var.data)
        
        curr_bl += 1

    # phase - 2 
    for i in range(len(mob_net_v2.phase_2)):

        if phase_type == 'inv_res_2':
            # first move to inv_res block
            # 0 is convbnrelu, 1 is convrbnrelu, 2 is conv, 3 is bn
            
            # 0 - convbnrelu
            # conv - weight
            mob_net_v2.phase_2[i].inv_res[0][0].weight.data.copy_(mobile_net.features[curr_bl].conv[0][0].weight.data)
            # bn - weight, bias, running_mean, running_var
            mob_net_v2.phase_2[i].inv_res[0][1].weight.data.copy_(mobile_net.features[curr_bl].conv[0][1].weight.data)
            mob_net_v2.phase_2[i].inv_res[0][1].bias.data.copy_(mobile_net.features[curr_bl].conv[0][1].bias.data)
            mob_net_v2.phase_2[i].inv_res[0][1].running_mean.data.copy_(mobile_net.features[curr_bl].conv[0][1].running_mean.data)
            mob_net_v2.phase_2[i].inv_res[0][1].running_var.data.copy_(mobile_net.features[curr_bl].conv[0][1].running_var.data)
            

            # 1 - convbnrelu
            # conv - weight
            mob_net_v2.phase_2[i].inv_res[1][0].weight.data.copy_(mobile_net.features[curr_bl].conv[1][0].weight.data)
            # bn - weight, bias, running_mean, running_var
            mob_net_v2.phase_2[i].inv_res[1][1].weight.data.copy_(mobile_net.features[curr_bl].conv[1][1].weight.data)
            mob_net_v2.phase_2[i].inv_res[1][1].bias.data.copy_(mobile_net.features[curr_bl].conv[1][1].bias.data)
            mob_net_v2.phase_2[i].inv_res[1][1].running_mean.data.copy_(mobile_net.features[curr_bl].conv[1][1].running_mean.data)
            mob_net_v2.phase_2[i].inv_res[1][1].running_var.data.copy_(mobile_net.features[curr_bl].conv[1][1].running_var.data)

            # 2 - conv
            mob_net_v2.phase_2[i].inv_res[2].weight.data.copy_(mobile_net.features[curr_bl].conv[2].weight.data)

            # 3 - bn
            mob_net_v2.phase_2[i].inv_res[3].weight.data.copy_(mobile_net.features[curr_bl].conv[3].weight.data)
            mob_net_v2.phase_2[i].inv_res[3].bias.data.copy_(mobile_net.features[curr_bl].conv[3].bias.data)
            mob_net_v2.phase_2[i].inv_res[3].running_mean.data.copy_(mobile_net.features[curr_bl].conv[3].running_mean.data)
            mob_net_v2.phase_2[i].inv_res[3].running_var.data.copy_(mobile_net.features[curr_bl].conv[3].running_var.data)
        
        curr_bl += 1

    # phase - 3
    for i in range(len(mob_net_v2.phase_3)):
        
        if phase_type == 'inv_res_2':
            # first move to inv_res block
            # 0 is convbnrelu, 1 is convrbnrelu, 2 is conv, 3 is bn
            
            # 0 - convbnrelu
            # conv - weight
            mob_net_v2.phase_3[i].inv_res[0][0].weight.data.copy_(mobile_net.features[curr_bl].conv[0][0].weight.data)
            # bn - weight, bias, running_mean, running_var
            mob_net_v2.phase_3[i].inv_res[0][1].weight.data.copy_(mobile_net.features[curr_bl].conv[0][1].weight.data)
            mob_net_v2.phase_3[i].inv_res[0][1].bias.data.copy_(mobile_net.features[curr_bl].conv[0][1].bias.data)
            mob_net_v2.phase_3[i].inv_res[0][1].running_mean.data.copy_(mobile_net.features[curr_bl].conv[0][1].running_mean.data)
            mob_net_v2.phase_3[i].inv_res[0][1].running_var.data.copy_(mobile_net.features[curr_bl].conv[0][1].running_var.data)
            

            # 1 - convbnrelu
            # conv - weight
            mob_net_v2.phase_3[i].inv_res[1][0].weight.data.copy_(mobile_net.features[curr_bl].conv[1][0].weight.data)
            # bn - weight, bias, running_mean, running_var
            mob_net_v2.phase_3[i].inv_res[1][1].weight.data.copy_(mobile_net.features[curr_bl].conv[1][1].weight.data)
            mob_net_v2.phase_3[i].inv_res[1][1].bias.data.copy_(mobile_net.features[curr_bl].conv[1][1].bias.data)
            mob_net_v2.phase_3[i].inv_res[1][1].running_mean.data.copy_(mobile_net.features[curr_bl].conv[1][1].running_mean.data)
            mob_net_v2.phase_3[i].inv_res[1][1].running_var.data.copy_(mobile_net.features[curr_bl].conv[1][1].running_var.data)

            # 2 - conv
            mob_net_v2.phase_3[i].inv_res[2].weight.data.copy_(mobile_net.features[curr_bl].conv[2].weight.data)

            # 3 - bn
            mob_net_v2.phase_3[i].inv_res[3].weight.data.copy_(mobile_net.features[curr_bl].conv[3].weight.data)
            mob_net_v2.phase_3[i].inv_res[3].bias.data.copy_(mobile_net.features[curr_bl].conv[3].bias.data)
            mob_net_v2.phase_3[i].inv_res[3].running_mean.data.copy_(mobile_net.features[curr_bl].conv[3].running_mean.data)
            mob_net_v2.phase_3[i].inv_res[3].running_var.data.copy_(mobile_net.features[curr_bl].conv[3].running_var.data)
        
        curr_bl += 1
    
    return mob_net_v2

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_size, out_size, kernel_size, stride, padding = 0, groups = 1):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_size, out_size, kernel_size = kernel_size, stride = stride, padding = padding, groups = groups, bias = False),
            nn.BatchNorm2d(out_size),
            nn.ReLU6(inplace = True))

class InvertedResidual_1(nn.Module):
    def __init__(self, in_filters, mid_filters, out_filters, stride = 1, groups = 1):
        super(InvertedResidual_1, self).__init__()
        layers = []
        layers.append(ConvBNReLU(mid_filters,mid_filters, 3, stride, 1, groups))
        layers.extend([nn.Conv2d(mid_filters, out_filters, 1, 1, bias = False),
                       nn.BatchNorm2d(out_filters)])
        self.inv_res = nn.Sequential(*layers)

    def forward(self, x):
        return self.inv_res(x)

class InvertedResidual_2(nn.Module):
    def __init__(self, in_filters, mid_filters, out_filters, stride = 1, groups = 1):
        super(InvertedResidual_2, self).__init__()
        self.stride = stride
        self.in_filters = in_filters
        self.out_filters = out_filters
        layers = []
        # parameters to call ConvBNReLU (in_size, out_size, kernel_size, stride, padding, groups)
        layers.append(ConvBNReLU(in_filters, mid_filters, 1, 1, 0, 1))
        layers.append(ConvBNReLU(mid_filters,mid_filters, 3, stride, 1, groups))
        layers.extend([nn.Conv2d(mid_filters, out_filters, 1, 1, bias = False),
                       nn.BatchNorm2d(out_filters)])
        self.inv_res = nn.Sequential(*layers)

    def forward(self, x):
        if self.stride == 1 and (self.in_filters == self.out_filters):
            return x + self.inv_res(x)
        else:
            return self.inv_res(x)


class Mobile_Net_V2(nn.Module):
    def __init__(self):
        super(Mobile_Net_V2, self).__init__()

        inv_res_bl = InvertedResidual_2

        self.phase_1_type = ['convbnrelu', 'inv_res_1', 'inv_res_2', 'inv_res_2',
                             'inv_res_2', 'inv_res_2', 'inv_res_2']
        p1_blocks = []
        # parameters to call ConvBNReLU (in_size, out_size, kernel_size, stride, padding, groups)
        p1_blocks.append(ConvBNReLU(3, 32, 3, 2, 1)) # 0 (-1, 32, 256, 256)
        # parameters order to call InvertedResidual (in_filters, mid_filters, out_filters, stride, groups)
        p1_blocks.append(InvertedResidual_1(32, 32, 16, 1, 32)) # 1 (-1, 16, 256, 256)
        p1_blocks.append(inv_res_bl(16, 96, 24, 2, 96)) # 2 (-1, 24, 128, 128)
        p1_blocks.append(inv_res_bl(24, 144, 24, 1, 144)) # 3 (-1, 24, 128, 128)
        p1_blocks.append(inv_res_bl(24, 144, 32, 2, 144)) # 4 (-1, 32, 64, 64)
        p1_blocks.append(inv_res_bl(32, 192, 32, 1, 192)) # 5 (-1, 32, 64, 64)
        p1_blocks.append(inv_res_bl(32, 192, 32, 1, 192)) # 6 (-1, 32, 64, 64)
        self.phase_1 = nn.Sequential(*p1_blocks)

        self.phase_2_type = ['inv_res_2', 'inv_res_2', 'inv_res_2', 'inv_res_2',
                             'inv_res_2', 'inv_res_2', 'inv_res_2']
        p2_blocks = []
        p2_blocks.append(inv_res_bl(32, 192, 64, 2, 192)) # 7 (-1, 64, 32, 32)
        p2_blocks.append(inv_res_bl(64, 384, 64, 1, 384)) # 8 (-1, 64, 32, 32)
        p2_blocks.append(inv_res_bl(64, 384, 64, 1, 384)) # 9 (-1, 64, 32, 32)
        p2_blocks.append(inv_res_bl(64, 384, 64, 1, 384)) # 10 (-1, 64, 32, 32)
        p2_blocks.append(inv_res_bl(64, 384, 96, 1, 384)) # 11 (-1, 96, 32, 32)
        p2_blocks.append(inv_res_bl(96, 576, 96, 1, 576)) # 12 (-1, 96, 32, 32)
        p2_blocks.append(inv_res_bl(96, 576, 96, 1, 576)) # 13 (-1, 96, 32, 32)
        self.phase_2 = nn.Sequential(*p2_blocks)

        self.phase_3_type = ['inv_res_2', 'inv_res_2', 'inv_res_2', 'inv_res']
        p3_blocks = []
        p3_blocks.append(inv_res_bl(96, 576, 160, 2, 576)) # 14 (-1, 160, 16, 16)
        p3_blocks.append(inv_res_bl(160, 960, 160, 1, 960)) # 15 (-1, 160, 16, 16)
        p3_blocks.append(inv_res_bl(160, 960, 160, 1, 960)) # 16 (-1, 160, 16, 16)
        p3_blocks.append(inv_res_bl(160, 960, 320, 1, 960)) # 17 (-1, 320, 16, 16)
        self.phase_3 = nn.Sequential(*p3_blocks)

    def forward(self, x):
        x_64 = self.phase_1(x)
        x_32 = self.phase_2(x_64)
        x_16 = self.phase_3(x_32)
        return [x_64, x_32, x_16]

'''
mob_net_v2 = Mobile_Net_V2().to(device)
mob_net_v2 = weight_initialize(mob_net_v2, mobile_net)
torch.save(mob_net_v2, 'mob_net_v2.pth')
'''