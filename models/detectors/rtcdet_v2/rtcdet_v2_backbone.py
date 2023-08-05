import torch
import torch.nn as nn
try:
    from .rtcdet_v2_basic import Conv, CSPFasterStage, DSBlock
except:
    from rtcdet_v2_basic import Conv, CSPFasterStage, DSBlock



model_urls = {
    'fasternet_n': None,
    'fasternet_t': None,
    'fasternet_s': None,
    'fasternet_m': None,
    'fasternet_l': None,
    'fasternet_x': None,
}


# ---------------------------- Backbones ----------------------------
# Modified FasterNet
class FasterConvNet(nn.Module):
    def __init__(self, width=1.0, depth=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(FasterConvNet, self).__init__()
        # ------------------ Basic parameters ------------------
        ## scale factor
        self.width = width
        self.depth = depth
        ## pyramid feats
        self.base_dims = [64, 128, 256, 512, 1024]
        self.feat_dims = [round(dim * width) for dim in self.base_dims]
        ## block depth
        self.base_blocks = [3, 9, 9, 3]
        self.feat_blocks = [round(nblock * depth) for nblock in self.base_blocks]
        ## nonlinear
        self.act_type = act_type
        self.norm_type = norm_type
        self.depthwise = depthwise
        
        # ------------------ Network parameters ------------------
        ## P1/2
        self.layer_1 = nn.Sequential(
            Conv(3, self.feat_dims[0], k=6, p=2, s=2, act_type=self.act_type, norm_type=self.norm_type),
            Conv(self.feat_dims[0], self.feat_dims[0], k=3, p=1, act_type=self.act_type, norm_type=self.norm_type, depthwise=self.depthwise),
        )
        ## P2/4
        self.layer_2 = nn.Sequential(   
            Conv(self.feat_dims[0], self.feat_dims[1], k=3, p=1, s=2, act_type=self.act_type, norm_type=self.norm_type, depthwise=self.depthwise),
            CSPFasterStage(self.feat_dims[1], self.feat_dims[1], self.feat_blocks[0], 3, True, self.act_type, self.norm_type)
        )
        ## P3/8
        self.layer_3 = nn.Sequential(
            DSBlock(self.feat_dims[1], self.feat_dims[2], act_type=self.act_type, norm_type=self.norm_type, depthwise=self.depthwise),
            CSPFasterStage(self.feat_dims[2], self.feat_dims[2], self.feat_blocks[1], 3, True, self.act_type, self.norm_type)
        )
        ## P4/16
        self.layer_4 = nn.Sequential(
            DSBlock(self.feat_dims[2], self.feat_dims[3], act_type=self.act_type, norm_type=self.norm_type, depthwise=self.depthwise),
            CSPFasterStage(self.feat_dims[3], self.feat_dims[3], self.feat_blocks[2], 3, True, self.act_type, self.norm_type)
        )
        ## P5/32
        self.layer_5 = nn.Sequential(
            DSBlock(self.feat_dims[3], self.feat_dims[4], act_type=self.act_type, norm_type=self.norm_type, depthwise=self.depthwise),
            CSPFasterStage(self.feat_dims[4], self.feat_dims[4], self.feat_blocks[3], 5, True, self.act_type, self.norm_type)
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]

        return outputs


# ---------------------------- Functions ----------------------------
## load pretrained weight
def load_weight(model, model_name):
    # load weight
    print('Loading pretrained weight ...')
    url = model_urls[model_name]
    if url is not None:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)
    else:
        print('No pretrained for {}'.format(model_name))

    return model


## build MCNet
def build_backbone(cfg, pretrained=False):
    # model
    backbone = FasterConvNet(cfg['width'], cfg['depth'], cfg['bk_act'], cfg['bk_norm'], cfg['bk_depthwise'])

    # check whether to load imagenet pretrained weight
    if pretrained:
        if cfg['width'] == 0.25 and cfg['depth'] == 0.34:
            backbone = load_weight(backbone, model_name='fasternet_n')
        elif cfg['width'] == 0.375 and cfg['depth'] == 0.34:
            backbone = load_weight(backbone, model_name='fasternet_t')
        elif cfg['width'] == 0.5 and cfg['depth'] == 0.34:
            backbone = load_weight(backbone, model_name='fasternet_s')
        elif cfg['width'] == 0.75 and cfg['depth'] == 0.67:
            backbone = load_weight(backbone, model_name='fasternet_m')
        elif cfg['width'] == 1.0 and cfg['depth'] == 1.0:
            backbone = load_weight(backbone, model_name='fasternet_l')
        elif cfg['width'] == 1.25 and cfg['depth'] == 1.34:
            backbone = load_weight(backbone, model_name='fasternet_x')
    feat_dims = backbone.feat_dims[-3:]

    return backbone, feat_dims


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        ## Backbone
        'backbone': 'mcnet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'width': 1.0,
        'depth': 1.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
    }
    model, feats = build_backbone(cfg)
    x = torch.randn(1, 3, 640, 640)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for out in outputs:
        print(out.shape)

    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))