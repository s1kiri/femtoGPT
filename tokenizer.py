import sentencepiece as spm
corpus_file = "all_prose.txt"
def train_sentencepiece(corpus_file, model_prefix, vocab_size=8000, character_coverage=0.99995):
    
    spm.SentencePieceTrainer.Train(f'--input={corpus_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage={character_coverage}')
    
train_sentencepiece(corpus_file=corpus_file, model_prefix = 'russian8000')
sp_model = spm.SentencePieceProcessor(model_file="russian8000.model")