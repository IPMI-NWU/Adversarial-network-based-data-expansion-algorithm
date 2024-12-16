"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04
from trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image

# 获取一个文件夹下的文件名
def get_filePath(path):
    pathDir = os.listdir(path)

    filePaths = []

    for file in pathDir:
        filePaths.append(file)
    return filePaths

# 创建文件夹
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder_origin.yaml',help="net configuration")
parser.add_argument('--input', default='/home/zhangdandan/project/part_data/dataset_v1/VinDr_RibCXR_square/train/img', help="input image path")
parser.add_argument('--output_folder', default='./results/test_vindr_ribcxr_4', help="output image path")
parser.add_argument('--checkpoint', default='./output_pth/outputs/edges2handbags_folder/checkpoints/gen_00900000.pt', help="checkpoint of autoencoders")
parser.add_argument('--style', type=str, default='', help="style image path")  # ./datasets/demo_edges2handbags/abnormal/CHNCXR_0425_1.png  ./datasets/demo_edges2handbags/abnormal/00002477_001.png
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")  # a 为正常的， b为不正常的
parser.add_argument('--seed', type=int, default=16, help="random seed")
parser.add_argument('--num_style',type=int, default=4, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()



torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

# Setup model and data loader
config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

try:
    # f = open(opts.checkpoint, 'rb')
    # data = torch.load(f, map_location='cpu')
    # state_dict = torch.load(opts.checkpoint, map_location='gpu')
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function

if 'new_size' in config:
    new_size = config['new_size']
else:
    if opts.a2b==1:
        new_size = config['new_size_a']
    else:
        new_size = config['new_size_b']

with torch.no_grad():
    file_list = get_filePath(opts.input)

    for img_name in file_list:
        img_path = os.path.join(opts.input, img_name)
        img_output_dir = os.path.join(opts.output_folder, img_name[:-4])
        mkdir(img_output_dir)

        transform = transforms.Compose([transforms.Resize(new_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])  #
        image = Variable(transform(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda())
        style_image = Variable(transform(Image.open(opts.style).convert('RGB')).unsqueeze(0).cuda()) if opts.style != '' else None

        # Start testing
        content, _ = encode(image)

        if opts.trainer == 'MUNIT':
            style_rand = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda())
            if opts.style != '':
                _, style = style_encode(style_image)
            else:
                style = style_rand
            for j in range(opts.num_style):
                s = style[j].unsqueeze(0)
                outputs = decode(content, s)
                outputs = (outputs + 1) / 2.
                path = os.path.join(img_output_dir, 'output{:03d}.jpg'.format(j))  # opts.output_folder
                vutils.save_image(outputs.data, path, padding=0, normalize=True)
        elif opts.trainer == 'UNIT':
            outputs = decode(content)
            outputs = (outputs + 1) / 2.
            path = os.path.join(opts.output_folder, 'output.jpg')
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
        else:
            pass

        if not opts.output_only:
            # also save input images
            vutils.save_image(image.data, os.path.join(img_output_dir, 'input.jpg'), padding=0, normalize=True)

