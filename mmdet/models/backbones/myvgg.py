from mmcv.cnn import VGG
import torch
from mmdet.registry import MODELS

@MODELS.register_module()
class MyVGG(VGG):
    def __init__(self,
                 pretrained=None,
                 *args,
                 **kwargs):
        self.pretrained = pretrained
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(*args, **kwargs)
        


    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)
    
    def forward(self, x):
        # outs = []
        # vgg_layers = getattr(self, self.module_name).to(self.device)
        
        # for i in range(len(self.stage_blocks)):
        #     for j in range(*self.range_sub_modules[i]):
        #         vgg_layer = vgg_layers[j].to(self.device)
        #         x = vgg_layer(x.squeeze())
        #         print(x)
        #         exit()
        #     if i in self.out_indices:
        #         outs.append(x)
        # if self.num_classes > 0:
        #     x = x.view(x.size(0), -1)
        #     x = self.classifier(x)
        #     outs.append(x)
        # if len(outs) == 1:
        #     return outs[0]
        # else:
        #     return tuple(outs)
        outs = []
        vgg_layers = getattr(self, self.module_name)
        for i in range(len(self.stage_blocks)):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                x = vgg_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            outs.append(x)
        
        import gc
        torch.cuda.empty_cache()
        gc.collect()

        return tuple(outs)