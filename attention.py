import torch
import torch.nn as nn


def self_attention(embedding, n_dim, qkv_dim = 64, mask=None):
    """performs scaled dot product attention

        attention = softmax(Q@K.T/sqrt(n_dim)) @ V
    
    """
    Q,K,V = [embedding]*3
    
    proj_q = nn.Linear(n_dim, qkv_dim)
    proj_k = nn.Linear(n_dim, qkv_dim)
    proj_v = nn.Linear(n_dim, qkv_dim)
    
    Q = proj_q(Q)
    K = proj_k(K)
    V = proj_v(V)

    scores = torch.bmm(Q, torch.transpose(K, 1,2))/n_dim**0.5

    return scores
    # if mask is None:
    #     scores = torch.softmax(scores, dim=-1)
    # else:
    #     scores = torch.softmax(scores+(1-mask)*-torch.inf,dim=-1)

    # return torch.bmm(scores, V)


