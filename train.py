from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.functional as F
from .gpt_model import GPTModel
from .hypers import num_layers, d_model, num_heads, dim_feedforward, vocab_size, block_size, batch_size, seq_len, learning_rate
from torch import optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = torch.load('X.pt')
Y = torch.load('Y.pt')
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
del X
del Y
model = GPTModel(num_layers, d_model, num_heads, dim_feedforward, vocab_size, block_size, batch_size, seq_len)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, betas=(0.9, 0.98), fused=True)
counter = 0
model.train()
scaler = torch.cuda.amp.GradScaler()
losse4ss = []
los = None
for epoch in range(1):
    total_loss = 0.0
    for batch_X, batch_Y in (pbar := tqdm(dataloader, total=len(dataloader))):
        counter+=1
        optimizer.zero_grad()
        batch_X = batch_X.to(torch.long).to(device)
        batch_Y = batch_Y.to(torch.long).to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model.forward(batch_X)
            loss = criterion(logits.view(-1, logits.size(-1)), batch_Y.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pbar.set_postfix_str(str(loss.item()))
        losse4ss.append(loss.item())
        if counter%120 == 0:
            model.eval()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                pred = torch.argmax(F.softmax(model(batch_X), dim = -1), dim=-1)
            plt.plot(losse4ss)
            plt.show()
            print(f'batch {counter} loss: {loss}\npred{pred[0][:10]}\ntrue{batch_Y[0][-10:]}')
            with open('gpt2_updated_losses.pkl', 'wb') as f:
                pickle.dump(losse4ss, f)
            model.train()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")
torch.save(model.state_dict(), "gpt_trained_state_dict.pt")

