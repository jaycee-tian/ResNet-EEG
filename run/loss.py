import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent_loss(emb1, emb2, temperature=0.5):
    batch_size = emb1.size(0)
    emb = torch.cat([emb1, emb2], dim=0)
    emb_norm = F.normalize(emb, dim=1)

    sim_matrix = torch.mm(emb_norm, emb_norm.t().contiguous())
    sim_matrix.fill_diagonal_(-float('inf'))  # 将对角线填充为负无穷，以避免计算相似度时取到自身

    # 计算相似度矩阵中的最大值，用于数值稳定性
    max_sim = torch.max(sim_matrix, dim=1).values.view(-1, 1)

    # 计算分母中的e^(s_ij / t)，并减去最大相似度，以防止数值溢出
    sim_matrix = torch.exp((sim_matrix - max_sim) / temperature)


    pos_sim = torch.cat([sim_matrix[i, i + batch_size].unsqueeze(0) for i in range(batch_size)])
    neg_sim = torch.cat([sim_matrix[i, i - batch_size].unsqueeze(0) for i in range(batch_size, 2 * batch_size)])

    loss_matrix = -torch.log(pos_sim / (pos_sim + neg_sim))

    loss = loss_matrix.sum() / (2 * batch_size)

    return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, x3, y):
        # 计算正样本损失（x1与x2之间的欧几里得距离）
        positive_distance = torch.norm(x1 - x2, p=2, dim=1)

        # 计算负样本损失（x1与x3之间的欧几里得距离）
        negative_distance = torch.norm(x1 - x3, p=2, dim=1)

        # 计算对比损失
        loss = 0.5 * (1 - y) * torch.pow(positive_distance, 2) + 0.5 * y * torch.pow(torch.clamp(self.margin - negative_distance, min=0.0), 2)
        return torch.sum(loss)
