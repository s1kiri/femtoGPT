import torch
import sentencepiece as spm
from .hypers import num_layers, d_model, num_heads, dim_feedforward, vocab_size, block_size, batch_size, seq_len
from .gpt_model import GPTModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'
new = spm.SentencePieceProcessor()
new.Load('tokenizer/russian8000.model')
model = GPTModel(num_layers, d_model, num_heads, dim_feedforward, vocab_size, block_size, batch_size, seq_len)
model.load_state_dict(torch.load('gpt_trained_state_dict.pt'))
model.eval()
text = 'start sequence @insert yours@'
tokens = new.encode_as_ids(text)
tokens = tokens[:256] # cut for context size
vectors = torch.tensor([tokens for i in range(128)])
generated = model.generate(vectors.to(device), 100, do_sample=True, top_k=3)
generated = generated.tolist()
generated = new.decode_ids(generated) # your context + generated
print(generated)