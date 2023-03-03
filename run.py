import json
from glob import glob
from FOD.Predictor import Predictor

with open('config.json', 'r') as f:
    config = json.load(f)

input_root = f'/home/dw/data/vgg/face/'
output_root = f'/home/dw/data/vgg/trans_depth_normal/' 
start = 2301 
end = 2361
gpu = 2
# input_images = glob(input_dir + '*.png')
predictor = Predictor(config, input_root, output_root, start, end, gpu)
predictor.run()
