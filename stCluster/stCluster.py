import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import GATConv


class stCluster(nn.Module):

    def __init__(self, layers, dec_alpha, init_centroids, rates, temp) -> None:
        super().__init__()

        # encoder
        self.encoder_gat_1 = GATConv(in_feats=layers[0], out_feats=layers[1], num_heads=1)
        self.encoder_linear = nn.Linear(in_features=layers[-2], out_features=layers[-1], bias=False)

        # optimize model parameters
        # loss weight 
        self.gene_rate = rates['gene_rate']
        self.adj_rate = rates['adj_rate']
        self.pred_rate = rates['pred_rate']

        self.temp = temp                                                                        # temperature parameter
        self.dec_alpha = dec_alpha                                                              # DEC degree of freedom
        self.embed_clust = nn.Parameter(init_centroids)                                         # DEC centroid
        self.decoder = nn.Linear(in_features=layers[-1], out_features=layers[0], bias=False)    # decoder
        self.activation = nn.ELU()                                                              # activation function

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_normal_(self.encoder_linear.weight)
        nn.init.xavier_normal_(self.decoder.weight)


    def forward(self, g, x, train_CL=False):
        # learning embedding
        embed = self.activation(self.encoder_gat_1(feat=x, graph=g)).squeeze()
        embed = self.encoder_linear(embed)

        if (train_CL):
            return embed
        else:
            # gene reconstruction
            pred_gene = self.activation(self.decoder(embed))

            # calculate clustering distribution q
            q = 1.0 / ((1.0 + torch.sum((embed.unsqueeze(1) - self.embed_clust).pow(2), dim=2) / self.dec_alpha) + 1e-8)
            q = q.pow((self.dec_alpha + 1.0) / 2.0)
            q = q / torch.sum(q, dim=1, keepdim=True)

            return embed, pred_gene, q
    

    def _cal_CL_loss(self, embed_1, embed_2):
        # normalize embedding 
        norm_1 = F.normalize(embed_1, dim=1)
        norm_2 = F.normalize(embed_2, dim=1)

        # calculate similarity
        cross_sim = torch.exp(torch.mm(norm_1, norm_2.t()) / self.temp) # Cross-view similarity (similarity between two views)
        inter_sim = torch.exp(torch.mm(norm_1, norm_1.t()) / self.temp) # internal similarity in view 1

        return - torch.log(cross_sim.diag() / (cross_sim.sum(dim=1) + inter_sim.sum(dim=1) - cross_sim.diag()))
    

    def CL_loss(self, embed_1, embed_2):
        CL_loss_1 = self._cal_CL_loss(embed_1, embed_2)
        CL_loss_2 = self._cal_CL_loss(embed_2, embed_1)

        return (CL_loss_1 + CL_loss_2).mean() / 2


    def multitask_loss(self, embed, pred_gene, x, q, adj):
        # gene reconstruction loss
        gene_loss = F.mse_loss(pred_gene, F.normalize(x, dim=1))

        # adjacency matrix reconstruction loss
        norm_embed = F.normalize(embed, dim=1)
        pred_adj = torch.mm(norm_embed, norm_embed.t())
        adj_loss = F.mse_loss(pred_adj, adj)
        
        # spatial domain prediction loss
        p = q.pow(2) / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        pred_loss = torch.mean(torch.sum(p * torch.log(p / (q + 1e-6)), dim=1))

        return self.gene_rate * gene_loss + self.adj_rate * adj_loss + self.pred_rate * pred_loss
    

class Denoising(nn.Module):

    def __init__(self, layers) -> None:
        super().__init__()

        # decoder
        self.encoder_gat_1 = GATConv(in_feats=layers[0], out_feats=layers[1], num_heads=1)
        self.encoder_linear = nn.Linear(in_features=layers[-2], out_features=layers[-1], bias=False)

        self.activation = nn.LeakyReLU()  # activation function

        self.reset_parameters()


    def reset_parameters(self):
        # nn.init.xavier_normal_(self.encoder_gat_1.weight)
        nn.init.xavier_normal_(self.encoder_linear.weight)


    def forward(self, g, x,):
        # learning embedding
        gene = self.activation(self.encoder_gat_1(feat=x, graph=g)).squeeze()
        gene = self.encoder_linear(gene)

        return gene
