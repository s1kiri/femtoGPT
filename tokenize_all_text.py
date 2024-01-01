import torch
from tqdm import tqdm
import sentencepiece as spm

new = spm.SentencePieceProcessor()
new.Load('tokenizer/russian8000.model')

with open('all_prose.txt', 'r', encoding='utf-8') as f:
        text = f.read()
tokens = new.encode_as_ids(text)
data = tokens[:5_000_000]
tokens = torch.tensor(tokens)
X = []
Y = []
segment_size = 256
for i in tqdm(range(0, len(data)//2 - segment_size, 1)):
    x_segment = data[i:i+segment_size]
    y_segment = data[i+1:i+segment_size+1]
    X.append(torch.tensor(x_segment).to(torch.long))
    Y.append(torch.tensor(y_segment).to(torch.long))
X = torch.stack(X)
Y = torch.stack(Y)
torch.save(X.to(torch.long), "X.pt")
torch.save(Y.to(torch.long), "Y.pt")
