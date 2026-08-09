import pandas as pd
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tqdm.auto import tqdm

"""
Dataset notes

The tokenizer turns text into token IDs:

    "Once upon a time"
        -> [431, 447, 259, 396]

For causal language modeling, the task is simple:

    given tokens up to position t, predict token t + 1

So if we have:

    [10, 20, 30, 40, 50]

we train with:

    x = [10, 20, 30, 40]
    y = [20, 30, 40, 50]

y is just x shifted by one token.

The Transformer can train all positions in parallel because causal
attention prevents each position from seeing future tokens.

We also append an <|endoftext|> token after every story so the model
can learn where a story ends.

Example:

    story 1 -> [10, 31, 82, 19, 7, EOS]
    story 2 -> [42, 55, 91, 22, 7, EOS]

These stories can then be combined into one long token stream.

For a context length T, each training sample is:

    x = tokens[i : i + T]
    y = tokens[i + 1 : i + T + 1]

We generate these windows inside Dataset.__getitem__ instead of storing
all of them beforehand, since neighboring windows overlap heavily.

Final batch shapes:

    x: [batch_size, context_length]
    y: [batch_size, context_length]

These token IDs will later be passed through the embedding layer before
going into the Transformer.
"""

class TinyDataset(Dataset):
    def __init__(self, all_tokens, tokens_y, context_length):
        self.all_tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return len(self.all_tokens) - self.context_length

    def __getitem__(self, idx):
        x = self.all_tokens[idx: idx+context_length]
        y = self.tokens_y[idx+1: idx+self.context_length + 1] 
        return x, y


# Load the training stories and tokenizer.
train_stories = pd.read_csv("dataset/train.csv")
tokenizer_out_path = "artifacts/tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_out_path)
eos_token_id = tokenizer.token_to_id("<|endoftext|>")

# Keep only valid story strings.
stories = train_stories["text"].dropna()
stories = stories[stories.map(type).eq(str)].tolist()

# Tokenize the stories in batches and separate them with an end-of-text token.
all_tokens = []
batch_size = 10000
for i in tqdm(range(0, len(stories), batch_size)):
    batch = stories[i:i + batch_size]
    encodings = tokenizer.encode_batch(batch)

    for encoding in encodings:
        all_tokens.extend(encoding.ids)
        all_tokens.append(eos_token_id)

context_length = 8

# Build the dataset and shuffled data loader.
dataset = TinyDataset(all_tokens, context_length)
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=128)

x,y = next(iter(dataloader))

print(f"X, Y shapes: {x.shape} {y.shape}")
print(f" X: {x[0]}, Y: {y[0]}")

print("\nDecoded x:")
print(tokenizer.decode(x[0].tolist()))

print("\nDecoded y:")
print(tokenizer.decode(y[0].tolist()))