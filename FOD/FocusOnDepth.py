import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from FOD.Reassemble import Reassemble
from FOD.Fusion import Fusion
from FOD.Head import HeadDepth

class FocusOnDepth(nn.Module):
    def __init__(self,
                 image_size = (3, 384, 384),
                 patch_size = 16,
                 emb_dim = 1024,
                 resample_dim = 256,
                 read = 'projection',
                 num_layers_encoder = 24,
                 hooks = [5, 11, 17, 23],
                 reassemble_s = [4, 8, 16, 32],
                 nhead = 16,
                 transformer_dropout = 0):
        """
        Focus on Depth - Large
        image_size : (c, h, w)
        patch_size : *a square*
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super().__init__()

        #Splitting img into patches
        channels, image_height, image_width = image_size
        assert image_height % patch_size == 0 and image_width % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_size) * (image_width // patch_size)
        patch_dim = channels * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, emb_dim),
        )
        #Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))

        #Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dropout=transformer_dropout, dim_feedforward=emb_dim*4)
        self.transformer_encoders = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_encoder)
        #Register hooks
        self.activation = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)

        #Reassembles Fusion
        self.reassembles = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles.append(Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim))
            self.fusions.append(Fusion(resample_dim))
        self.reassembles = nn.ModuleList(self.reassembles)
        self.fusions = nn.ModuleList(self.fusions)

        #Head
        self.head = HeadDepth(resample_dim)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        t = self.transformer_encoders(x)
        previous_stage = None
        for i in np.arange(len(self.fusions)-1, -1, -1):
            hook_to_take = 't'+str(self.hooks[i])
            activation_result = self.activation[hook_to_take]
            reassemble_result = self.reassembles[i](activation_result)
            fusion_result = self.fusions[i](reassemble_result, previous_stage)
            previous_stage = fusion_result
        out = self.head(previous_stage)
        return out

    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook
        for h in hooks:
            self.transformer_encoders.layers[h].register_forward_hook(get_activation('t'+str(h)))