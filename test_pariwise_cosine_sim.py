import torch
from torch.nn import CosineSimilarity
import time

# naive pairwise cosine similarity
import pdb


n = 64
d = 1024
# x0 = torch.randn(100, 128)
x0 = torch.randn(n, d)
# x0 = torch.arange(10).reshape(-1,2).double()
# x0 = torch.arange(10)
# x1 = torch.randn(100, 128)

start = time.time()

# a = x0.view(-1,1).expand(10,10).reshape(-1)
a = x0.view(-1,1,d).expand(n,n,d).reshape(-1,d)
# b = x0.repeat(10)
b = x0.repeat(n,1)

cos = CosineSimilarity()
out = cos(a,b).reshape(n,n)

end = time.time()

print("Seth time to compute: " + str(end-start))


start = time.time()

proj_sent_emb = x0
norm_proj_sent_emb = proj_sent_emb / torch.norm(proj_sent_emb, dim = 1).unsqueeze(1)
cos_sim = torch.matmul(norm_proj_sent_emb, norm_proj_sent_emb.T)

end = time.time()

print("Mitch time to compute: " + str(end-start))

pdb.set_trace()
print((out == cos_sim).all())


# print('hi')