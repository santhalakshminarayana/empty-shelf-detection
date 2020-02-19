import torch
import torch.nn as nn
from itertools import product
import math

class Default_Boxes(object):
    def __init__(self):
        super(Default_Boxes, self).__init__()

        self.img_size = 512
        self.s_min = 0.2
        self.s_max = 0.9
        self.num_boxes = [6, 6, 6, 6, 4]
        self.m = len(self.num_boxes)
        self.feature_map_dims = [8, 4, 8, 16 ,32]
        self.shrinkage = [64, 128, 64, 32, 16]
        self.aspect_ratios = [[2, 3], [2, 3], [2, 3],
                              [2, 3], [2], [2]]
        self.s_k = []
        for i in range(self.m + 1):
            s = (self.s_max - self.s_min) / (self.m - 1)
            s = self.s_min + s * (i - 0)
            self.s_k.append(s)
        
    def forward(self):
        boxes = []
        for f, fp_dim in enumerate(self.feature_map_dims):    
            for i, j in product(range(fp_dim), repeat = 2):
                f_k = self.img_size / self.shrinkage[f]
                # centers for bounding box
                c_x = (j + 0.5) / f_k
                c_y = (i + 0.5) / f_k
                # box aspect ratio = 1
                # relative size = sk
                s = self.s_k[f]
                boxes.append([c_x, c_y, s, s])

                # box aspect ratio = 1 
                # relative size = sqrt(sk * sk+1)
                s = math.sqrt(self.s_k[f]*self.s_k[f + 1])
                boxes.append([c_x, c_y, s, s])

                for ratio in self.aspect_ratios[f]:
                    s = self.s_k[f]
                    boxes.append([c_x, c_y, s*math.sqrt(ratio), s/math.sqrt(ratio)])
                    boxes.append([c_x, c_y, s/math.sqrt(ratio), s*math.sqrt(ratio)])
        
        boxes = torch.Tensor(boxes).reshape(-1, 4)
        boxes.clamp_(min = 0.0, max = 1.0)
        return boxes

'''
db = Default_Boxes().forward()
print(db.shape)
print((db < 0).nonzero())'''