import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import numpy as np

import traceback

class LDVAE(nn.Module):
    """
    This model is an implementation of the following paper.
    
    Attributes
    ----------

    """
    def __init__(
        self,
        genes_cnt: int, 
        hidden_dims: tuple,
        hidden_l_dims: tuple= (128,),
        latent_dim: int = 20,
        drop_use: bool = True,
        drop_rate: float = 0.1,
        norm_use: bool = True,
        norm_momentum: float = 0.01,
        eps: float = 1e-8,
    ):
        """
        Parameters
        ----------
        genes_cnt: int
            Number(int) of input genes
        hidden_dims: tuple of int
            tuple of the latent(z) encoder's hidden dimensions
        hidden_l_dims: tuple of int, default (128,)
            tuple of the library(l) encoder's hidden dimensions
        latent_dim: int, default 20
            Dimensionality of the latent space
        drop_use: boolean, default True
            Set drop_use to True if you use a dropout layer.
        drop_rate: float, default 0.1
            Add nn.Dropout(drop_rate). Ignore this if set drop_use to False.
        norm_use: boolean, default True
            Set norm_use to True if you use a batch norm layer.
        norm_momentum: float, default 0.01
            Add nn.BatchNorm1d(momentum=norm_momentum). Ignore this if set 
            norm_momentum to False.
        eps: float, default 1e-8
        """
        super(LDVAE,self).__init__()
        self.local_l_mean = None
        self.local_l_var = None
        self.hidden_dims = hidden_dims
        self.eps = eps

        # log_theta use reconst error
        self.log_theta = nn.Parameter(torch.randn(genes_cnt))

        # encoder_z, encoder_z_var, encoder_z_mean
        encoder_layers = []
        old_dim = genes_cnt
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(old_dim, dim))
            if norm_use:
                encoder_layers.append(nn.BatchNorm1d(dim,
                                                    eps=eps, 
                                                    momentum=norm_momentum))
            encoder_layers.append(nn.ReLU())
            if drop_use:
                encoder_layers.append(nn.Dropout(drop_rate))
            old_dim = dim
        self.encoder_z = nn.Sequential(*encoder_layers)
        self.encoder_z_mean = nn.Linear(hidden_dims[-1],latent_dim)
        self.encoder_z_var = nn.Linear(hidden_dims[-1],latent_dim)

        # encoder_l, encoder_l_var, encoder_l_mean
        encoder_layers = []
        in_dim = genes_cnt
        for dim in hidden_l_dims:
            encoder_layers.append(nn.Linear(in_dim, dim))
            if norm_use:
                encoder_layers.append(nn.BatchNorm1d(dim,
                                                    eps=eps, 
                                                    momentum=norm_momentum))
            encoder_layers.append(nn.ReLU())
            if drop_use:
                encoder_layers.append(nn.Dropout(drop_rate))
            in_dim = dim
        self.encoder_l = nn.Sequential(*encoder_layers)
        self.encoder_l_mean = nn.Linear(hidden_l_dims[-1],1)
        self.encoder_l_var = nn.Linear(hidden_l_dims[-1],1)

        # decoder
        decoder = []
        decoder.append(nn.Linear(latent_dim, genes_cnt,bias=False))
        if norm_use:
            decoder.append(nn.BatchNorm1d(genes_cnt,
                            eps=eps, 
                            momentum=norm_momentum))
        self.decoder = nn.Sequential(*decoder)


    def forward(self,x: torch.Tensor):
        """
        forward

        Parameters
        ----------
        x: torch.Tensor
            input tensor (genes * cells)

        Return
        ------
        [z_mean, z_var, z]: list of tensor
            z_mean: latent variables(z)' mean
            z_var:  latent variables (z)' variance
            z: resample latent varuables(z) 
        [l_mean, l_var, library]: list of tensor
            l_mean: library mean
            l_var: library variance
            llbrary: resample library from l_mean and l_var
        y: tensor

        Raises
        -------
        self.local_l_mean is None
        self.local_l_var is 

        See Also
        --------
        loss: get loss
        """

        # create latent variables(z) 
        x_z = self.encoder_z(x)
        z_mean = self.encoder_z_mean(x_z)
        # z_logvar = self.encoder_z_std
        z_var = torch.exp(torch.clip(self.encoder_z_var(x_z),max=10)) 
        z_eps = torch.randn(z_mean.shape).to('cuda' if next(self.parameters()).is_cuda else 'cpu')
        z = z_mean + z_var * z_eps

        # create library
        x_l = self.encoder_l(x)
        l_mean = self.encoder_l_mean(x_l)
        l_var = torch.exp(torch.clip(self.encoder_l_var(x_l),max=10))
        l_eps = torch.randn(l_mean.shape).to('cuda' if next(self.parameters()).is_cuda else 'cpu')
        library = (l_mean + l_var * l_eps)

        # decoder
        y = self.decoder(z)
        y = torch.exp(torch.clip(library,max=10))*torch.softmax(y, dim=-1)
        return [z_mean, z_var, z], [l_mean, l_var, library], y


    def set_local_l_mean_and_var(self, data):
        """
        set local_l_mean and local_l_var
        l_mean get closer local_l_mean and l_var get closer local_l_var.
        """
        masked_log_sum =np.ma.log(data.sum(axis=1))
        log_counts = masked_log_sum.filled(0)
        self.local_l_mean = (np.mean(log_counts).reshape(-1, 1)).astype(np.float32)[0][0]
        self.local_l_var = (np.var(log_counts).reshape(-1, 1)).astype(np.float32)[0][0]
        return self.local_l_mean, self.local_l_var


    def reconst_error(self,x, mu, theta):
        """
        get reconst error
        """
        eps = self.eps
        log_theta_mu_eps = torch.log(theta + mu + eps)

        res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
        return res


    def loss(self,x):
        """
        get loss

        Parameters
        ----------
        x: torch.Tensor
            input data (genes*cell)

        Returns
        -------
        reconst: torch.Tensor
            reconst error
        kl_l: torch.Tensor
            kl divergence of library
        kl_z: torch.Tensor
             kl divergence of latent variables(z)

        Examples
        --------
        reconst, kl_l ,kl_z = model.loss(x)
        loss = torch.mean(-reconst+kl_l +kl_z)
        """
        zs,ls,y = self.forward(x)
        z_mean, z_var, z = zs
        l_mean, l_var, library = ls

        mean, std = torch.zeros_like(z_mean), torch.ones_like(z_var)
        kl_z = kl_divergence(Normal(z_mean,torch.sqrt(z_var)), Normal(mean, std)).sum(dim=1)

        try:
            if (self.local_l_mean is None) or (self.local_l_var is None):
                raise ValueError('please use loss() after set_local_l_mean_and_var()')
        except :
            traceback.print_exc()
            
        mean, var = self.local_l_mean*torch.ones_like(l_mean), self.local_l_var*torch.ones_like(l_var)
        kl_l = kl_divergence(Normal(l_mean,torch.sqrt(l_var)), Normal(mean, torch.sqrt(var))).sum(dim=1)

        reconst = self.reconst_error(x, mu=y, theta=torch.exp(self.log_theta)).sum(dim=-1)        
        return reconst, kl_l ,kl_z
