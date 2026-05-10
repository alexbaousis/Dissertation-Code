import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

import model.config_task as config_task


class conv_task(nn.Module):
    def __init__(self, in_planes, planes, stride=1, kernel_size=3, padding=1, num_tasks=2):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gamma = nn.Parameter(torch.ones(planes, num_tasks*(num_tasks - 1)))
        self.beta = nn.Parameter(torch.zeros(planes, num_tasks*(num_tasks - 1)))
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # first, get the taskpair information: compute A
        A_taskpair = config_task.A_taskpair

        x = self.conv(x)

        # generate taskpair-specific FiLM parameters
        gamma = torch.mm(A_taskpair, self.gamma.t())
        beta = torch.mm(A_taskpair, self.beta.t())
        gamma = gamma.view(1, x.size(1), 1, 1)
        beta = beta.view(1, x.size(1), 1, 1)

        x = self.bn(x)

        # taskpair-specific transformation
        x = x * gamma + beta
        x = self.relu(x)


        return x

class Region_Contra_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
        self.min_region = 120
        self.max_regions = 5
        self.max_bank_samples = 10
        self.temp = 5.0
        self.gmm_bank = deque(maxlen=1000)

    def forward(self, map_s, map_t, mask=None, index=None):
        losses, new_bank = [], []

        for b in range(map_s.size(0)):
            gx, gy = self.get_region_gmms(map_s[b], map_t[b], mask[b])
            
            for i in range(len(gx)):
                losses.append(self.comp_constra_loss_gmm(gx[i], gy[i], gy, i))

            new_bank.extend(gy)

        for gmm in new_bank:
            copied_gmm = {}

            for key, value in gmm.items():
                copied_gmm[key] = np.array(value, dtype=np.float32, copy=True)

            self.gmm_bank.append(copied_gmm)

        return map_s.new_tensor(sum(losses) / len(losses) if losses else 0.0)

    def compute_gmm_distance(self, g1, g2, eps=1e-8):

        w1 = np.asarray(g1["weights"], np.float32)[:, None]
        m1 = np.asarray(g1["means"], np.float32)
        c1 = np.asarray(g1["covariances"], np.float32)

        w2 = np.asarray(g2["weights"], np.float32)[None, :]
        m2 = np.asarray(g2["means"], np.float32)
        c2 = np.asarray(g2["covariances"], np.float32)

        num_components_1 = m1.shape[0]
        num_components_2 = m2.shape[0]

        mean_term = np.zeros((num_components_1, num_components_2), dtype=np.float32)
        cov_term = np.zeros((num_components_1, num_components_2), dtype=np.float32)

        for i in range(num_components_1):
            for j in range(num_components_2):
                mean_difference = m1[i] - m2[j]
                mean_term[i, j] = np.sum(mean_difference ** 2)

                covariance_difference =  c1[i] + c2[j] - 2 * np.sqrt(np.maximum(c1[i] * c2[j], eps))

                cov_term[i, j] = np.sum(covariance_difference)

        return float((w1 * w2 * (mean_term + cov_term)).sum())




    def fit_region_gmm(self, region_tensor):
        samples = region_tensor.detach().cpu().numpy().T.astype(np.float32, copy=False)
        n_samples = samples.shape[0]

        if n_samples < 2:
            return None
        
        k = min(2, n_samples)

        gmm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-4, random_state=0)
        gmm.fit(samples)


        return {"weights": gmm.weights_, "means": gmm.means_, "covariances": gmm.covariances_}
  

    def get_region_gmms(self, x, y, sam):
        region_ids = torch.unique(sam)

        gx, gy = [], []

        for region_id in region_ids:
            idx = (sam == region_id).nonzero(as_tuple=True)

            gmm_x = self.fit_region_gmm(x[:, idx[0], idx[1]])
            gmm_y = self.fit_region_gmm(y[:, idx[0], idx[1]])

            if gmm_x is not None and gmm_y is not None:
                gx.append(gmm_x)
                gy.append(gmm_y)

        return gx, gy


    def sample_bank_negatives(self):
        bank = list(self.gmm_bank)

        if len(bank) <= self.max_bank_samples:
            return bank

        sampled_negatives = random.sample(bank, self.max_bank_samples)
        
        return sampled_negatives


    def comp_constra_loss_gmm(self, gmm_anchor, gmm_pos, all_pos_gmms, anchor_index):
        pos = np.exp(-self.compute_gmm_distance(gmm_anchor, gmm_pos) / self.temp)
        negative_gmms = []

        for region_index, gmm in enumerate(all_pos_gmms):
            if region_index != anchor_index:
                negative_gmms.append(gmm)

        if len(self.gmm_bank) > 0:
            negative_gmms += self.sample_bank_negatives()
        


        negative_similarities = []

        for negative_gmm in negative_gmms:
            distance = self.compute_gmm_distance(gmm_anchor, negative_gmm)
            similarity = np.exp(-distance / self.temp)
            negative_similarities.append(similarity)

        neg = sum(negative_similarities) / len(negative_similarities)
                
        return float(-np.log(pos / (neg + self.eps) + self.eps))


class RegionConsistency(nn.Module):
    def __init__(self, tasks=("semantic", "depth", "normal"), last_inp_channels=512):
        super().__init__()

        n = len(tasks)

        self.last_layer = nn.Sequential(conv_task(last_inp_channels, last_inp_channels, kernel_size=1, padding=0, num_tasks=n), conv_task(last_inp_channels, 32, kernel_size=1, padding=0, num_tasks=n),)
        self.comp_contrastive_loss = Region_Contra_Loss()

    def forward(self, map_s, map_t, img_size, mask=None, index=None):
        map_s = F.interpolate(map_s, img_size, mode="bilinear")
        map_t = F.interpolate(map_t, img_size, mode="bilinear")
        mask = F.interpolate(mask.float().unsqueeze(0), img_size, mode="nearest").squeeze(0) 

        return self.comp_contrastive_loss(map_s, map_t, mask, index)