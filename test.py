import torch
from transformer import Transformer
BATCH_SIZE, SEQ_LENGTH, DIM, NUM_CLASSES = 3, 30, 12, 4

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.rand(BATCH_SIZE, SEQ_LENGTH, DIM).to(device)
input_lengths = torch.IntTensor([30, 16, 8])
targets = torch.LongTensor([[2, 3, 3, 3, 3, 3, 2, 2, 1, 0],
                            [2, 3, 3, 3, 3, 3, 2, 1, 2, 0],
                            [2, 3, 3, 3, 3, 3, 2, 2, 0, 1]]).to(device)  # 1 means <eos_token>
target_lengths = torch.IntTensor([10, 9, 8])
model = Transformer(idim=12, odim=5, nlayer=1, ahead=2, adim=10).cuda()

loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.003)
model.train()
for i in range(100):
    optim.zero_grad()
    loss, out = model(inputs, input_lengths, targets)
    loss.backward()
    optim.step()
    print(f'{loss.item():.2f}')
print(out)