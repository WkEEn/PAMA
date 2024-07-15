import math
from PIL import Image
import torchvision
import torch
import torch.utils.data as data
import torch.distributed as dist
from torch.utils.data import Sampler
import csv
import os
import numpy as np
import random
import cv2

MAX_LENGTH = 500


def extract_tile(image_dir, tile_size, x, y, width, height):
    x_start_tile = x // tile_size
    y_start_tile = y // tile_size
    x_end_tile = (x + width) // tile_size
    y_end_tile = (y + height) // tile_size

    tmp_image = np.ones(
        ((y_end_tile - y_start_tile + 1) * tile_size, (x_end_tile - x_start_tile + 1) * tile_size, 3),
        np.uint8) * 240

    for y_id, col in enumerate(range(x_start_tile, x_end_tile + 1)):
        for x_id, row in enumerate(range(y_start_tile, y_end_tile + 1)):
            img_path = os.path.join(image_dir, '{:04d}_{:04d}.jpg'.format(row, col))
            # tcga
            # img_path = os.path.join(image_dir, '{}_{}.png'.format(int(col), int(row)))
            if not os.path.exists(img_path):
                # print(f'{img_path} not exists!')
                continue
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            if h != tile_size or w != tile_size:
                continue
            tmp_image[(x_id * tile_size):(x_id * tile_size + h), (y_id * tile_size):(y_id * tile_size + w), :] = img

    x_off = x % tile_size
    y_off = y % tile_size
    output = tmp_image[y_off:y_off + height, x_off:x_off + width]

    return output


class SlideLocalTileDataset(data.Dataset):
    def __init__(self, image_dir, position_list, transform,
                 tile_size=512, imsize=224, od_mode=False, invert_rgb=False):
        self.transform = transform

        self.im_dir = image_dir
        self.pos = position_list
        self.od = od_mode
        self.ts = tile_size
        self.imsize = imsize
        self.inv_rgb = invert_rgb

    def __getitem__(self, index):
        img = extract_tile(self.im_dir, self.ts, self.pos[index][1], self.pos[index][0], self.imsize, self.imsize)
        if len(img) == 0:
            img = np.ones((self.imsize, self.imsize, 3), np.uint8) * 240
        # The default format of opencv is BGR but PIL.Image is RGB.
        # So, a cvtColor is required here, to make sure the color
        # channels are consistent with the trained model.
        if not self.inv_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')
        img = self.transform(img)

        if self.od:
            img = -torch.log(img + 1.0 / 255.0)

        return img

    def __len__(self):
        return self.pos.shape[0]
