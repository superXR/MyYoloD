from time import time
import torch
from torch import tensor
import torch.nn as nn
import sys,os
import math
import sys
sys.path.append(os.getcwd())
#sys.path.append("lib/models")
#sys.path.append("lib/utils")
#sys.path.append("/workspace/wh/projects/DaChuang")
sys.path.append('/mnt/sdb/dpai3/project/YOLOP')
from lib.utils import initialize_weights
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv
from lib.models.transformer_encoding import Reasoning
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized


# The lane line and the driving area segment branches without share information with each other and without link
YOLOP = [
[33, 42, 51],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10   nchw [1, 256, 20, 20]
[ -1, Reasoning, [256, 4, 1, 512]],  # 11 Reasoning Layer
[ [-1, 10], Concat, [1]],     # 12
[ -1, Conv, [512, 256, 1, 1]],     # 13    nchw [1, 256, 20, 20]

[ -1, Upsample, [None, 2, 'nearest']],  #14
[ [-1, 6], Concat, [1]],    #15
[ -1, BottleneckCSP, [512, 256, 1, False]], #16
[ -1, Conv, [256, 128, 1, 1]],  #17   nchw [1, 128, 40, 40]
[ -1, Reasoning, [128, 2, 1, 512]],  # 18 Reasoning Layer
[ [-1, 17], Concat, [1]],     # 19
[ -1, Conv, [256, 128, 1, 1]],     # 20  nchw [1, 128, 40, 40]

[ -1, Upsample, [None, 2, 'nearest']],  #21
[ [-1,4], Concat, [1]],     #22         #Encoder   nchw [1, 256, 80, 80]
[ -1, Reasoning, [256, 4, 1, 512]],     # 23 Reasoning Layer
[ [-1, 22], Concat, [1]],     # 24
[ -1, Conv, [512, 256, 1, 1]],     # 25   Encoder nchw [1, 256, 80, 80]

[ -1, BottleneckCSP, [256, 128, 1, False]],     #26
[ -1, Conv, [128, 128, 3, 2]],      #27
[ [-1, 20], Concat, [1]],       #28
[ -1, BottleneckCSP, [256, 256, 1, False]],     #29
[ -1, Conv, [256, 256, 3, 2]],      #30
[ [-1, 13], Concat, [1]],   #31
[ -1, BottleneckCSP, [512, 512, 1, False]],     #32
[ [26, 29, 32], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detection head 33

[ 25, Conv, [256, 128, 3, 1]],   #34
[ -1, Upsample, [None, 2, 'nearest']],  #35
[ -1, BottleneckCSP, [128, 64, 1, False]],  #36
[ -1, Conv, [64, 32, 3, 1]],    #37
[ -1, Upsample, [None, 2, 'nearest']],  #38
[ -1, Conv, [32, 16, 3, 1]],    #39
[ -1, BottleneckCSP, [16, 8, 1, False]],    #40
[ -1, Upsample, [None, 2, 'nearest']],  #41
[ -1, Conv, [8, 2, 3, 1]], #42 Driving area segmentation head

[ 25, Conv, [256, 128, 3, 1]],   #43
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, BottleneckCSP, [128, 64, 1, False]],  #45
[ -1, Conv, [64, 32, 3, 1]],    #46
[ -1, Upsample, [None, 2, 'nearest']],  #47
[ -1, Conv, [32, 16, 3, 1]],    #48
[ -1, BottleneckCSP, [16, 8, 1, False]],    #49
[ -1, Upsample, [None, 2, 'nearest']],  #50
[ -1, Conv, [8, 2, 3, 1]] #51 Lane line segmentation head
]


class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save= [], []
        self.nc = 10
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]
        

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _, _= model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()
        
        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        Da_fmap = []
        LL_fmap = []
        neck_fmap = []
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            x = block(x)
            if i in self.seg_out_idx:     #save driving area segment result
                m=nn.Sigmoid()
                out.append(m(x))
            if i == self.detector_index:
                det_out = x
            # if i < 11:
            #     # neck_fmap.append(x)
            #     print(str(i) + ' shape:', x.shape)
            cache.append(x if block.index in self.save else None)
        out.insert(0,det_out)
        # out.append(neck_fmap)
        return out
            
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

def get_tf_net(cfg, **kwargs): 
    m_block_cfg = YOLOP
    model = MCnet(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":
    # from torch.utils.tensorboard import SummaryWriter
    model = get_tf_net(False)
    input_ = torch.randn((1, 3, 640, 640))
    # gt_ = torch.rand((1, 2, 256, 256))
    # metric = SegmentationMetric(2)
    t1 = time()
    model_out, Da_fmap, LL_fmap = model(input_)
    t2 = time()
    print('infer time:', t2 - t1)
    print('detect shape:')
    for det in model_out:
        print(det.shape)
    print('Da_fmap shape:', Da_fmap.shape)
    print('LL_fmap shape:', LL_fmap.shape)
    # print('neck_index 1-10 shape:')
    # for neck in neck_fmap:
    #     print(neck.shape)
 
