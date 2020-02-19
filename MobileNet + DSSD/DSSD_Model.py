import torch
import torch.nn as nn

class Deconvolution_Module(nn.Module):
    def __init__(self, ssd_in_filters):
        super(Deconvolution_Module, self).__init__()
        
        self.deconv = nn.ConvTranspose2d(512, 512, kernel_size = 2, stride = 2)
        self.conv_ssd = nn.Conv2d(ssd_in_filters, 512, 3, padding = 1)
    
        self.conv = nn.Conv2d(512, 512, 3, padding = 1)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace = True)

        # upsampling Deconvolution feature maps
        self.deconv_layer = nn.Sequential(*[self.deconv, self.conv, self.bn])
        # convolutions on SSD feature maps
        self.ssd_layer = nn.Sequential(*[self.conv_ssd, self.bn, self.relu, self.conv, self.bn])

    def forward(self, deconv_features, ssd_features):
        x = self.deconv_layer(deconv_features)
        y = self.ssd_layer(ssd_features)
        return self.relu(x * y)

class DSSD(nn.Module):
    def __init__(self, base):
        super(DSSD, self).__init__()

        self.base = base
        deconv_module = Deconvolution_Module
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # deconv_ssd(channels) 
        self.deconv_32 = deconv_module(32)
        self.deconv_96 = deconv_module(96)
        self.deconv_320 = deconv_module(320)
        self.deconv_512 = deconv_module(512)

        # predict module
        self.conv_c1024 = nn.Conv2d(512, 1024, 1)
        layers = []
        layers.append(nn.Conv2d(512, 256, 1))
        layers.append(nn.Conv2d(256, 256, 1))
        layers.append(nn.Conv2d(256, 1024, 1))
        self.predict_module = nn.Sequential(*layers)

        self.conv_16_8 = nn.Conv2d(320, 512, 3, 2, padding = 1)
        self.conv_8_4 = nn.Conv2d(512, 512, 3, 2, padding = 1)

        self.classes = 2
        self.loc, self.conf = [], []
        default_boxes = [6, 6, 6, 6, 4]
        for num_boxes in default_boxes:
            self.loc.append(nn.Conv2d(1024, 4*num_boxes, 3, padding = 1))
            self.conf.append(nn.Conv2d(1024, num_boxes, 3, padding = 1))
        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)

        for layer in [*self.loc, *self.conf]:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, x):
        x_64, x_32, x_16 = self.base(x)
        # Channels in x_16_16 = 32, x_32_32 = 96, x_64_64 = 320
        x_8 = self.conv_16_8(x_16)
        p_1 = self.predict_module(x_8) + self.conv_c1024(x_8)

        x_4 = self.conv_8_4(x_8)
        p_2 = self.predict_module(x_4) + self.conv_c1024(x_4)

        # deconv_x(deconv_features, ssd_features) returns (deconv_features * 512)
        x_8 = self.deconv_512(x_4, x_8)
        p_3 = self.predict_module(x_8) + self.conv_c1024(x_8)

        x_16 = self.deconv_320(x_8, x_16)
        p_4 = self.predict_module(x_16) + self.conv_c1024(x_16)

        x_32 = self.deconv_96(x_16, x_32)
        p_5 = self.predict_module(x_32) + self.conv_c1024(x_32)

        #x_64 = self.deconv_32(x_32, x_64)
        #p_6 = self.predict_module(x_64) + self.conv_c1024(x_64)

        locs = []
        confs = []
        features_maps = [p_1, p_2, p_3, p_4, p_5]
        for ft, lc, cf in zip(features_maps, self.loc, self.conf):
            locs.append(lc(ft).view(ft.shape[0], 4, -1))
            confs.append(self.sigmoid(cf(ft)).view(ft.shape[0], 1, -1))

        locs = torch.cat(locs, 2).transpose(2, 1).contiguous()
        confs = torch.cat(confs, 2).contiguous().squeeze()
        confs[torch.isnan(confs).nonzero().squeeze()] == 0
        return locs, confs


'''
dsd = DSSD(mob_net_v2).to(device)
l, c = dssd(torch.randn(2,3,512,512).to(device))
print(l.shape, c.shape)'''