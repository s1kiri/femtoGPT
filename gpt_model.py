import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from .hypers import num_layers, d_model, num_heads, dim_feedforward, vocab_size, block_size, batch_size, seq_len
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_decoder_mask(seq_length):
    mask = torch.triu(torch.full((seq_length, seq_length), float('-inf')), diagonal=1)
    return mask

class GPTModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward, vocab_size, block_size, batch_size, seq_len):
        super(GPTModel, self).__init__()
        self.block_size = block_size
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)
        self.decoder_blocks = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward, batch_first=True, bias=False, activation='gelu', layer_norm_eps=1e-8, device=device) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        self.indicies = torch.arange(0, seq_len).repeat(batch_size, 1).to(device)
        self.dummy_memory = torch.zeros(batch_size, block_size, d_model).to(device)
        self.mask = create_decoder_mask(seq_len).to(device)

    def forward(self, x):
        word_emb = self.word_embedding(x).to(device)
        pos_emb = self.pos_embedding(self.indicies).to(device)
        output = word_emb + pos_emb
        output = output.to(device)
        for decoder_block in self.decoder_blocks:
            output = decoder_block(output, tgt_mask=self.mask, memory=self.dummy_memory)  
        logits = self.linear(output).to(device)
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        for _ in tqdm(range(max_new_tokens)):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

model = GPTModel(num_layers, d_model, num_heads, dim_feedforward, vocab_size, block_size, batch_size, seq_len)
model = model.to(device)