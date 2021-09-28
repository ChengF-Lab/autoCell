import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn import cluster, mixture
from collections import Counter

class Block(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=tuple(), headers=None, reverse=False, dropout=0.1, batch_norm=True):
        super(Block, self).__init__()
        assert isinstance(hidden_dims, Sequence)
        if reverse:
            out_dim, in_dim = in_dim, out_dim
            hidden_dims = list(reversed(hidden_dims))
        self.dims = [in_dim]+list(hidden_dims)
        if len(hidden_dims)==0:
            self.base = nn.Identity()
        else:
            model = []
            for dim_in, dim_out in zip(self.dims, self.dims[1:]):
                model.append(nn.Linear(dim_in, dim_out))
                if batch_norm:
                    model.append(nn.BatchNorm1d(dim_out))
                model.append(nn.ReLU(inplace=True))
                if dropout>0:
                    model.append(nn.Dropout(dropout))
            self.base = nn.Sequential(*model)
        if headers is None:
            headers = [nn.Linear(self.dims[-1], out_dim)]
        self.headers = nn.ModuleList(headers)

    def forward(self, x):
        x = self.base(x)
        output = [header(x) for header in self.headers]
        return output[0] if len(output)==1 else output


class ReconstructionLoss(nn.Module):
    def __init__(self, distribution="gaussian"):
        super(ReconstructionLoss, self).__init__()
        self.distribution = distribution
        func = {"gaussian":self.gaussian_distribution_loss_fn,
                "bernoulli":self.bernoulli_distribution_loss_fn,
                "possion":self.possion_distribution_loss_fn,
                "nb":self.nb_distribution_loss_fn,
                "zinb":self.zinb_distribution_loss_fn}
        sample_fn = {"gaussian":self.gaussian_sample_fn,
                     "bernoulli": self.bernoulli_sample_fn,
                     "possion": self.possion_sample_fn,
                     "nb": self.nb_sample_fn,
                     "zinb": self.nb_sample_fn,
                     }
        func = func.get(distribution, None)
        if func:
            self.forward = func
            self.sample = sample_fn.get(distribution, None)
        else:
            raise NotImplementedError
        self.register_buffer("eps", torch.tensor(1e-10))
        self.register_buffer("one", torch.tensor(1.0))

    def bernoulli_sample_fn(self, mu, theta=None, drop_prob=None, sample_shape=torch.Size()):
        return torch.distributions.Bernoulli(mu).sample(sample_shape)

    def bernoulli_distribution_loss_fn(self, mu, y, theta=None, drop_prob=None):
        return F.binary_cross_entropy(input=mu, target=y, reduction="none")

    def gaussian_sample_fn(self, mu, theta=None, drop_prob=None, sample_shape=torch.Size()):
        return torch.distributions.Normal(loc=mu, scale=torch.ones_like(mu)).sample(sample_shape)

    def gaussian_distribution_loss_fn(self, mu, y, theta=None, drop_prob=None):
        return F.mse_loss(mu, y, reduction="none")

    def possion_sample_fn(self, mu, theta=None, drop_prob=None, sample_shape=torch.Size()):
        return torch.distributions.Poisson(rate=mu).sample(sample_shape)

    def possion_distribution_loss_fn(self, mu, y, theta=None, drop_prob=None):
        return mu - y * torch.log(mu + self.eps) + torch.lgamma(y + self.one)

    def nb_sample_fn(self, mu, theta, drop_prob=None, sample_shape=torch.Size()):
        return torch.distributions.Gamma(theta, theta/mu).sample(sample_shape)

    def nb_distribution_loss_fn(self, mu, y, theta, drop_prob=None):
        t1 = torch.lgamma(theta+1e-10)+torch.lgamma(y+1.0)-torch.lgamma(y+theta+1e-10)
        t2 = (theta+y)*torch.log(1.0+(mu/(theta+1e-10)))+y*(torch.log(theta+1e-10)-torch.log(mu+1e-10))
        loss = t1+t2
        assert loss.sum(dim=-1).min()>-self.eps
        return loss

    def zinb_distribution_loss_fn2(self, mu, y, theta, drop_prob):
        nb_case = self.nb_distribution_loss_fn(mu, y, theta) - torch.log(self.one - drop_prob + self.eps)
        zero_nb = torch.pow(theta / (theta + mu + self.eps), theta)
        zero_case = -torch.log(drop_prob + ((self.one - drop_prob) * zero_nb) + self.eps)
        result = torch.where(y < self.eps, zero_case, nb_case)
        # result = result.sum(dim=-1)
        assert result.sum(dim=-1).min() > -self.eps
        return result

    def zinb_distribution_loss_fn(self, mu, y, theta, drop_prob):
        x, pi = y, drop_prob
        eps = 1e-10
        softplus_pi = F.softplus(-pi)  #  uses log(sigmoid(x)) = -softplus(-x)
        log_theta_eps = torch.log(theta + eps)
        log_theta_mu_eps = torch.log(theta + mu + eps)
        pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

        case_zero = F.softplus(pi_theta_log) - softplus_pi
        mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

        case_non_zero = (
                -softplus_pi
                + pi_theta_log
                + x * (torch.log(mu + eps) - log_theta_mu_eps)
                + torch.lgamma(x + theta)
                - torch.lgamma(theta)
                - torch.lgamma(x + 1)
        )
        mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)
        result = -(mul_case_zero + mul_case_non_zero)
        # assert result.sum(dim=-1).min() > -eps
        return result

    def __str__(self):
        return f"{self.distribution.title()}DistributionLoss()"

    def __repr__(self):
        return self.__str__()



class NormalSampler(nn.Module):
    """p(z)"""
    def __init__(self):
        super(NormalSampler, self).__init__()
        self.register_buffer("eps", torch.tensor(1e-10))

    def forward(self, mean, log_var):
        epsilon = torch.randn(mean.size(), requires_grad=False, device=mean.device)
        std = log_var.mul(0.5).exp_()
        z = mean.addcmul(std, epsilon)
        return z

    def kl_divergence(self, mean, log_var, z):
        """
        L elbo(x) = Eq(z|x)[log p(x|z)] - KL(q(z|x)||p(z))
        D_{KL}(q(z|x)||p(z))
        """
        return -0.5*torch.sum(1+log_var-mean.pow(2)-log_var.exp(), dim=1)



class GMMSampler(NormalSampler):
    def __init__(self, latent_dim, n_centroid, trainable=False):
        super(GMMSampler, self).__init__()
        self.n_centroid = n_centroid
        self.trainable = trainable
        self.gmm = mixture.GaussianMixture(n_components=n_centroid, random_state=0,
                                           warm_start=True, covariance_type="diag")
        if trainable:
            self._pi = nn.Parameter(torch.ones(1, n_centroid)/n_centroid) #Bxk
            self.mu_c = nn.Parameter(torch.zeros(1, latent_dim, n_centroid)) #BxDxk
            self._var_c = nn.Parameter(torch.ones(1, latent_dim, n_centroid)) #BxDxk
        else:
            # self.gmm = CenteringMethod(method="kmeans")
            self.register_buffer("_pi", torch.ones(1, n_centroid)/n_centroid)
            self.register_buffer("mu_c", torch.zeros(1, latent_dim, n_centroid))
            self.register_buffer("_var_c", torch.ones(1, latent_dim, n_centroid))


    def fit(self, z):
        self.gmm.fit(z)
        pi = torch.from_numpy(self.gmm.weights_).float()
        mu_c = torch.from_numpy(self.gmm.means_.T).float()
        var_c = torch.from_numpy(self.gmm.covariances_.T).float()
        if not self.trainable:
            # label, pi, mean, var = self.gmm(z, n_clusters=self.n_centroid)
            # mu_c = torch.from_numpy(mean).float()
            # var_c = torch.from_numpy(var).float()
            # pi = torch.from_numpy(pi).float()
            self.init_parameter(mu_c=mu_c, var_c=var_c, pi=pi)
        return pi.unsqueeze(0).to(device=self.pi.device), \
               mu_c.unsqueeze(0).to(device=self.mu_c.device), \
               var_c.unsqueeze(0).to(device=self.var_c.device)

    @torch.no_grad()
    def update(self, z):
        prob = self.get_gamma(z, update_pi=False)
        pi = prob.mean(dim=0)
        mu = (prob.unsqueeze(1)*z.unsqueeze(-1)).sum(dim=0)/prob.sum(dim=0, keepdim=True)
        var = (prob.unsqueeze(1) * (z.unsqueeze(-1)-mu.unsqueeze(0)).square()).sum(dim=0) / prob.sum(dim=0, keepdim=True)
        return pi, mu, var

    @property
    def var_c(self):
        return torch.relu(self._var_c)+self.eps

    @property
    def pi(self):
        return self._pi/self._pi.sum(dim=-1, keepdim=True)

    def init_parameter(self, mu_c=None, var_c=None, pi=None):
        # var_c = None
        if mu_c is None:
            mu_c = torch.zeros(*self.mu_c.shape[1:])
        if var_c is None:
            var_c = torch.ones(*self.var_c.shape[1:])
        if pi is None:
            pi = torch.ones(self.n_centroid)/self.n_centroid
        self.mu_c.data = mu_c.unsqueeze(0).to(device=self.mu_c.device)
        self._var_c.data = var_c.unsqueeze(0).to(device=self.var_c.device)
        self._pi.data = pi.unsqueeze(0).to(device=self.pi.device)

    def get_gamma(self, z, update_pi=False):
        """
        Inference c from z
        gamma is q(c|x)
        q(c|x) = p(c|z) = p(c,z)/p(z) =p(c)p(c|z)/p(z)
        z: [B, D]
        """
        B, D = z.shape
        z = z.unsqueeze(2).expand(B, D, self.n_centroid)
        # p(c,z) = p(c)*p(z|c) as p_c_z
        pi = self.pi
        p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5*torch.log(2*math.pi*self.var_c) + (z-self.mu_c)**2/(2*self.var_c), dim=1))+self.eps
        gamma = p_c_z/torch.sum(p_c_z, dim=1, keepdim=True)
        return gamma

    def kl_divergence(self, mean, log_var, z):
        """
        L elbo(x) = Eq(z,c|x)[ log p(x|z) ] - KL(q(z,c|x)||p(z,c))
                  = Eq(z,c|x)[ log p(x|z) + log p(z|c) + log p(c) - log q(z|x) - log q(c|x) ]
        D_{KL}( q(z,c|x) || p(z,c) ) = Eq(z,c|x)[ -log p(z|c) -log p(c) +log q(z|x) +log q(c|x) ]
        """
        mu_c, var_c = self.mu_c, self.var_c
        n_centroids = self.n_centroid
        B, D = mean.shape
        mu_expand = mean.unsqueeze(2).expand(B, D, n_centroids)
        logvar_expand = log_var.unsqueeze(2).expand(B, D, n_centroids)

        # q(c|x)
        gamma = self.get_gamma(z, update_pi=True)

        # E log p(z|c)
        logpzc = -0.5*torch.sum(gamma*torch.sum(math.log(2*math.pi) + \
                                               torch.log(var_c) + \
                                               torch.exp(logvar_expand)/var_c + \
                                               (mu_expand-mu_c)**2/var_c, dim=1), dim=1)
        # E log p(c)
        pi = self.pi
        logpc = torch.sum(gamma*torch.log(pi), 1)
        # E log q(z|x) or q entropy
        qentropy = -0.5*torch.sum(1+log_var+math.log(2*math.pi), 1)
        # E log q(c|x)
        logqcx = torch.sum(gamma*torch.log(gamma), 1)
        log_qcx_pc = torch.sum(gamma*torch.log(gamma/(self.eps+pi)), 1)
        # kld = -logpzc - logpc + qentropy + logqcx
        kld = -logpzc+qentropy+log_qcx_pc
        # assert kld.min()>-self.eps
        return kld, {"loss_logqzx":qentropy.mean(),
                     "loss_logqcx":logqcx.mean(),
                     "loss_logpzc":logpzc.mean(),
                     "loss_log_qcx_pc":log_qcx_pc.mean(),
                     "loss_logpc":logpc.mean(),
                     "prob":gamma.detach()}

class DGGSampler(GMMSampler):
    def __init__(self, latent_dim, n_centroid, neighbor_num, trainable=False):
        super(DGGSampler, self).__init__(latent_dim, n_centroid, trainable=trainable)
        self.neighbor_num = neighbor_num
        self._sim = None

    def get_gamma(self, feature, update_pi=True):
        output = super(DGGSampler, self).get_gamma(feature, update_pi=False)
        # if update_pi and self._sim is not None:
        #     self._pi.data = self.compute_pi(output, self._sim)
        return output

    def kl_divergence(self, mean, log_var, z):
        x = mean
        neighbor = x.unsqueeze(0)
        sim = self.compute_similary(x=x, neighbor=neighbor)
        sim, index = torch.topk(sim, dim=1, k=min(self.neighbor_num, sim.shape[1]))
        sim = sim/sim.sum(dim=-1, keepdims=True)
        neighbor_mean = x[index]
        neighbor_log_var = log_var[index]
        neighbor_z = z[index]
        B, N, D = neighbor_mean.shape
        neighbor_log_var = neighbor_log_var.reshape(-1, D)
        neighbor_mean = neighbor_mean.reshape(-1, D)
        neighbor_z = neighbor_z.reshape(-1, D)
        sim[:, 0] = sim[:, 1:].sum(dim=-1)+1
        self._sim = sim
        loss, info = super(DGGSampler, self).kl_divergence(mean=neighbor_mean,
                                                           log_var=neighbor_log_var,
                                                           z=neighbor_z)
        info["prob"] = info["prob"].view(B, N, -1)[:,0,:]
        loss = loss.view(B, N) * sim.view(-1, N)
        info["sim"] = sim
        return loss.sum(dim=-1), info

    @torch.no_grad()
    def compute_similary(self, x, neighbor, sigma=10):
        """
        :param x: [B, D]
        :param neighbor: [B, N, D]
        :param similarity_type:
        :return: sim: [B, N]
        """
        B, D = x.shape
        dist = (x.reshape(B, 1, D) - neighbor).square().sum(dim=-1)
        dist = dist / sigma
        Gauss_simi = torch.softmax(-dist, dim=1)
        return Gauss_simi

    @torch.no_grad()
    def compute_similary2(self, x, neighbor):
        prob = self.get_gamma(x)
        return self.compute_similary(prob, prob.unsqueeze(0))

    @torch.no_grad()
    def compute_pi(self, gamma_n, sim_n):
        """
        :param gamma: [B*N, K]
        :param sim_n: [B, N]
        :return:
        """
        B = sim_n.shape[0]
        N = gamma_n.shape[0] // sim_n.shape[0]
        K = self.n_centroid
        pc = gamma_n.view(-1, N, K)
        pc = pc * sim_n.unsqueeze(-1)
        pi = torch.sum(pc, dim=1)
        pi = pc[:, 0, :] / pi
        pi = pi.unsqueeze(1).expand(B, N, K).reshape(-1, K)
        return pi

class CenteringMethod():
    def __init__(self, method="spec"):
        self.method = method
        if method=="spec":
            self.fn = self.spec_fn
        elif method=="kmeans":
            self.fn = self.kmeans_fn
        elif method=="gmm":
            self.fn = self.gmm_fn
        else:
            raise NotImplementedError

    def kmeans_fn(self, feature, n_clusters):
        kmeans = cluster.KMeans(n_clusters=n_clusters)
        label = kmeans.fit_predict(feature)
        count = Counter(label)
        pi = np.array([count[i] for i in range(n_clusters)])
        mean = kmeans.cluster_centers_
        return label

    def gmm_fn(self, feature, n_clusters):
        gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='diag')
        label = gmm.fit_predict(feature)
        mean = gmm.means_.T
        var = gmm.covariances_.T
        pi = gmm.weights_
        return label, pi, mean, var

    def spec_fn(self, feature, n_clusters):
        spec = cluster.SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0)
        label = spec.fit_predict(feature)
        return label

    def __call__(self, feature, n_clusters):
        assert not np.any(np.isnan(feature)) and not np.any(np.isinf(feature))
        label = self.fn(feature, n_clusters)
        mean = []
        var = []
        count = Counter(label)
        for i in range(n_clusters):
            sub_feature = feature[label==i]
            sub_mean = sub_feature.mean(axis=0)
            sub_var = sub_feature.var(axis=0)*len(sub_feature)/(len(sub_feature)-1)
            mean.append(sub_mean)
            var.append(sub_var)
        mean = np.stack(mean, axis=1)
        var = np.stack(var, axis=1)
        pi = np.array([count[i] for i in range(n_clusters)])
        return label, pi, mean, var

class WarmUpScheduler():
    def __init__(self, max_step, min_value=0.1, max_value=1.0, frozen_step=0):
        self.max_step = max_step
        self.min_value = min_value
        self.max_value = max_value
        self.cnt = 0
        self.frozen_step = frozen_step

    def step(self):
        ans = self.compute()
        self.cnt += 1
        return ans

    def init(self):
        self.cnt = 0

    def compute(self):
        ans =  min(max(self.cnt / self.max_step, self.min_value), self.max_value)
        if self.cnt<self.frozen_step:
            return self.min_value
        return ans


class EMA():
    def __init__(self, decay=0.9):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.decay) * x + self.decay * self.shadow[name]
        self.shadow[name] = new_average.clone()