import torch
import torch.utils.data as data
import csv
import random
import numpy as np

MAX_LENGTH = 500
MAX_LENGTH_CASE = 2000



class TCGALungKDataset(data.Dataset):
    def __init__(self, root, data_path, fold, args, set="test",
                 shuffle=True, max_size=MAX_LENGTH, max_kernel_num=64, patch_per_kernel=36, polar_bins=8):
        self.root = root
        self.labels = []
        self.slide_list = []
        self.shuffle = shuffle
        self.max_size = max_size
        self.set = set
        self.fold = fold
        self.args = args
        self.nk = max_kernel_num
        self.patch_per_kernel = patch_per_kernel
        self.polar_bins=polar_bins
        try:
            with open(data_path) as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    slide_id = row[0]
                    slide_label = int(row[2])

                    self.slide_list.append(slide_id)
                    self.labels.append(slide_label)
        finally:
            pass

    def __getitem__(self, ind):
        slide_id = self.slide_list[ind]
        label = int(self.labels[ind])
        full_path = self.root + '/' + f'{slide_id}.pth'
        features_dict = torch.load(full_path, map_location='cpu')

        num_node = min(features_dict['feature_array'].shape[0], self.max_size)
        features = features_dict['feature_array'][:num_node]
        nk_lvl = np.where(np.asarray(features_dict['npks']) == self.patch_per_kernel)[0][0]

        anchor_num = min(features_dict['kns'][nk_lvl], self.nk)

        k_index_min = features_dict['k_index'][nk_lvl][:anchor_num]
        k_len = len(features_dict['k_index'][nk_lvl])
        polar_pos = features_dict['polar_pos'][self.polar_bins][k_len][:anchor_num, :][:, :num_node]
        re_dist = features_dict['re_dist'][k_index_min, :][:, :num_node]
        wsidata = self.pack_data(features, re_dist, polar_pos, num_node)
        return wsidata, label, slide_id

    def __len__(self):
        return len(self.slide_list)

    def pack_data(self, feat, rd, polar, num_node):
        num_anchor = rd.shape[0]

        wsi_feat = np.zeros((self.max_size, feat.shape[-1]))
        wsi_rd = np.zeros((self.nk, self.max_size))
        wsi_polar = np.zeros((self.nk, self.max_size))

        wsi_feat[:num_node] = np.squeeze(feat)
        wsi_rd[:num_anchor, :num_node] = rd
        wsi_polar[:num_anchor, :num_node] = polar[:, :, 0]
        wsi_polar[wsi_polar > int(self.polar_bins - 1)] = int(self.polar_bins - 1)

        token_mask = np.zeros((self.max_size, 1), int)
        token_mask[:num_node] = 1
        kernel_mask = np.zeros((self.nk, 1), int)
        kernel_mask[:num_anchor] = 1

        return wsi_feat, wsi_rd, wsi_polar, token_mask, kernel_mask

    def get_weights(self):
        labels = np.asarray(self.labels)
        tmp = np.bincount(labels)
        weights = 1 / np.asarray(tmp[labels], np.float)

        return weights

class DistributedWeightedSampler(data.DistributedSampler):
    def __init__(self, dataset, weights, num_replicas=None, rank=None, replacement=True):

        super(DistributedWeightedSampler, self).__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=False
            )
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, self.replacement).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)