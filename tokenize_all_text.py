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
for i in tqdm(range(0, len(data) - segment_size, 1)):
    x_segment = data[i:i+segment_size]
    y_segment = data[i+segment_size]
    X.append(torch.tensor(x_segment))
    Y.append(torch.tensor(y_segment))
X = torch.stack(X)
Y = torch.stack(Y)
torch.save(X, "X.pt")
torch.save(Y, "Y.pt")
