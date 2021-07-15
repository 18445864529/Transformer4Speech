import torch
from transformer import Transformer

B, T, F = 3, 30, 12

x = torch.rand(B, T, F)
x_len = torch.tensor([30, 16, 8])
y = torch.tensor([[2, 3, 3, 3, 3, 3, 2, 2, 1, 0],
                  [2, 3, 3, 3, 3, 3, 2, 1, 2, 0],
                  [2, 3, 3, 3, 3, 3, 2, 2, 0, 1]])
model = Transformer(idim=12, odim=5, nlayer=1, ahead=2, adim=10).cuda()
optim = torch.optim.Adam(model.parameters(), lr=0.003)
model.train()
for i in range(100):
    optim.zero_grad()
    loss = model(x, x_len, y)
    loss.backward()
    optim.step()
    print(f'{loss.item():.2f}')
