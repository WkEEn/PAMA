
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
torch.autograd.set_detect_anomaly(True)

import models_posemb



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PAMA pre-training')

# additional configs:
parser.add_argument('--global_pool', action='store_true')
parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                    help='Use class token instead of global pool for classification')

# pama specific configs:
parser.add_argument('--max-size', default=8, type=int,
                    help='number of classes (default: 5)')
parser.add_argument('--max-kernel-num', default=128, type=int,
                    help='images input size')
parser.add_argument('--patch-per-kernel', default=18, type=int,
                    help='images input size')
parser.add_argument('--model', default='pama_vit_base', type=str, metavar='MODEL',
                    help='Name of model to train')
parser.add_argument('--polar-bins', default=8, type=int,
                    help='in_chans')
parser.add_argument('--kernel-drop', default=0.2, type=float,
                    help='kernel dropout rate.')
parser.add_argument('--input_size', default=2048, type=int,
                    help='images input size')

parser.add_argument('--in-chans', default=512, type=int,
                    help='in_chans')



def main():
    args = parser.parse_args()
    model = models_posemb.__dict__[args.model](num_kernel=args.max_kernel_num,
                                               in_chans=args.in_chans,
                                               embed_dim=1024, depth=4, num_heads=8,
                                               polar_bins=args.polar_bins)
    
    pretrain_model = './checkpoints/multi_organ_pretrain.pth.tar'
    if os.path.isfile(pretrain_model):
        print("=> loading checkpoint '{}'".format(pretrain_model))
        checkpoint = torch.load(pretrain_model, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # rename byol pre-trained keys
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        new_state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in state_dict and state_dict[k].shape != new_state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
        print("=> missing_keys\n", msg.missing_keys)

        print("=> loaded pre-trained model '{}'".format(pretrain_model))
    else:
        print("=> no checkpoint found at '{}'".format(pretrain_model))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    slide_pth = './checkpoints/23515CDA-1131-4847-85E9-E42C1A50CFFF.pth'
    wsidata = load_slide(slide_pth, args)
    if torch.cuda.is_available():
        wsi_feat = torch.tensor(wsidata[0]).float().to(device).unsqueeze(0)
        wsi_rd = torch.tensor(wsidata[1]).int().to(device).unsqueeze(0)
        wsi_polar = torch.tensor(wsidata[2]).int().to(device).unsqueeze(0)
        token_mask = torch.tensor(wsidata[3]).int().to(device).unsqueeze(0)
        kernel_mask = torch.tensor(wsidata[4]).int().to(device).unsqueeze(0)
    slide_embedding, kernel_tokens = model(wsi_feat, wsi_rd, wsi_polar, token_mask, kernel_mask, device, extract=True)
    print(slide_embedding.shape)
    print(kernel_tokens.shape)



def load_slide(slide_pth, args):
    features_dict = torch.load(slide_pth, map_location='cpu')

    num_node = min(features_dict['feature_array'].shape[0], args.max_size)
    features = features_dict['feature_array'][:num_node]
    nk_lvl = np.where(np.asarray(features_dict['npks']) == args.patch_per_kernel)[0][0]

    anchor_num = min(features_dict['kns'][nk_lvl], args.max_kernel_num)

    k_index_min = features_dict['k_index'][nk_lvl][:anchor_num]
    k_len = len(features_dict['k_index'][nk_lvl])
    polar_pos = features_dict['polar_pos'][args.polar_bins][k_len][:anchor_num, :][:, :num_node]
    re_dist = features_dict['re_dist'][k_index_min, :][:, :num_node]
    wsidata = pack_data(features, re_dist, polar_pos, num_node, args)
    return wsidata


def pack_data(feat, rd, polar, num_node, args):
    num_anchor = rd.shape[0]

    wsi_feat = np.zeros((args.max_size, feat.shape[-1]))
    wsi_rd = np.zeros((args.max_kernel_num, args.max_size))
    wsi_polar = np.zeros((args.max_kernel_num, args.max_size))

    wsi_feat[:num_node] = np.squeeze(feat)
    wsi_rd[:num_anchor, :num_node] = rd
    wsi_polar[:num_anchor, :num_node] = polar[:, :, 0]
    wsi_polar[wsi_polar > int(args.polar_bins - 1)] = int(args.polar_bins - 1)

    token_mask = np.zeros((args.max_size, 1), int)
    token_mask[:num_node] = 1
    kernel_mask = np.zeros((args.max_kernel_num, 1), int)
    kernel_mask[:num_anchor] = 1

    return wsi_feat, wsi_rd, wsi_polar, token_mask, kernel_mask



if __name__ == '__main__':
    main()
