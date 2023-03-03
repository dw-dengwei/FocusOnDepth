import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
from glob import glob
from PIL import Image

from FOD.FocusOnDepth import FocusOnDepth
from FOD.utils import create_dir, get_normal
from FOD.dataset import show


class Predictor(object):
    def __init__(self, config, input_root, output_root, start, end, gpu):
        # self.input_images = input_images
        self.config = config
        self.type = self.config['General']['type']

        # self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        self.device = torch.device(f'cuda:{gpu}')
        print("device: %s" % self.device)
        resize = config['Dataset']['transforms']['resize']
        self.model = FocusOnDepth(
                    image_size  =   (3,resize,resize),
                    emb_dim     =   config['General']['emb_dim'],
                    resample_dim=   config['General']['resample_dim'],
                    read        =   config['General']['read'],
                    nclasses    =   len(config['Dataset']['classes']) + 1,
                    hooks       =   config['General']['hooks'],
                    model_timm  =   config['General']['model_timm'],
                    type        =   self.type,
                    patch_size  =   config['General']['patch_size'],
        )
        path_model = os.path.join(config['General']['path_model'], 'FocusOnDepth.p')
        self.model.load_state_dict(
            torch.load(path_model, map_location=self.device)['model_state_dict']
        )
        self.model.to(self.device)
        self.model.eval()
        self.transform_image = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # create_dir(self.output_dir)
        self.input_root = input_root
        self.output_root = output_root
        self.start = start
        self.end = end

    def run(self):
        with torch.no_grad():
            for idx in range(self.start, 1 + self.end):
                input_images = glob(self.input_root + f'{idx}/*.png')
                output_dir = self.output_root + f'{idx}'
                create_dir(output_dir)
                # input_images = glob(input_dir + '*.png')
                for images in tqdm(input_images, desc=f'{idx}'):
                    pil_im = Image.open(images).convert('RGB')
                    original_size = pil_im.size
                    if original_size[0] != 128 or original_size[1] != 128:
                        continue

                    tensor_im = self.transform_image(pil_im).unsqueeze(0).to(self.device)
                    depth, _ = self.model(tensor_im)
                    depth = depth.squeeze().float().detach().cpu().numpy()
                    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                    normal = get_normal(depth)
                    depth = cv2.resize(depth, original_size)
                    normal = cv2.resize(normal, original_size)
                    # normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
                    dzyx = np.concatenate([depth.reshape(128, 128, 1), normal], axis=2)
                    # dzyx = cv2.cvtColor(dzyx, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(
                        output_dir, os.path.basename(images).split('.')[0] + '_dzyx.png'
                    ), dzyx)

                    # depth = transforms.ToPILImage()(depth.squeeze(0).float()).resize(original_size, resample=Image.BICUBIC)

                    # path_dir_segmentation = os.path.join(self.output_dir, 'segmentations')
                    # path_dir_depths = os.path.join(output_dir, 'depths')
                    # create_dir(path_dir_segmentation)
                    # output_segmentation.save(os.path.join(path_dir_segmentation, os.path.basename(images)))

                    # path_dir_depths = os.path.join(output_dir, 'depths')
                    # create_dir(outputd)
                    # depth.save(os.path.join(output_dir, os.path.basename(images)))