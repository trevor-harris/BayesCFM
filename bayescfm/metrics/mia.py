import torch

@torch.no_grad()
def _roc_auc_from_scores(scores, labels):
    scores = scores.detach().cpu(); labels = labels.detach().cpu().long()
    sorted_scores, indices = torch.sort(scores)
    ranks = torch.empty_like(indices, dtype=torch.float)
    i=0; n=scores.numel()
    while i<n:
        j=i+1
        while j<n and sorted_scores[j]==sorted_scores[i]: j+=1
        rank = 0.5*(i+j-1)+1.0
        ranks[indices[i:j]] = rank
        i=j
    pos = labels.sum().item(); neg = labels.numel() - pos
    if pos==0 or neg==0: return float("nan")
    U = ranks[labels==1].sum().item() - pos*(pos+1)/2.0
    return float(U / (pos*neg))

@torch.no_grad()
def mia_auc_nn_distance(member_feats, nonmember_feats, reference_feats, chunk=4096, invert=True):
    def min_dists(A,B):
        mins = torch.empty(A.size(0), device=A.device, dtype=A.dtype)
        for i in range(0, A.size(0), chunk):
            d = torch.cdist(A[i:i+chunk], B)
            mins[i:i+chunk], _ = d.min(1)
        return mins
    m = min_dists(member_feats, reference_feats)
    u = min_dists(nonmember_feats, reference_feats)
    scores = torch.cat([m,u],0)
    if invert: scores = -scores
    labels = torch.cat([torch.ones_like(m, dtype=torch.long), torch.zeros_like(u, dtype=torch.long)],0)
    auc = _roc_auc_from_scores(scores, labels)
    return auc, scores
