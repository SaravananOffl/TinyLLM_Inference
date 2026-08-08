import pandas as pd

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import os 

# -----------------------
# Load stories
# -----------------------
dataset = "dataset"
tokenizer_out_path = "artifacts/tokenizer.json"

train_stories_path =os.path.join(dataset, 'train.csv')
train_stories = pd.read_csv(train_stories_path)


def story_iterator(stories_df):
    for story in stories_df["text"]:
        if isinstance(story, str):
            yield story


# -----------------------
# Build tokenizer
# -----------------------

tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False
)

tokenizer.decoder = ByteLevelDecoder()


# -----------------------
# BPE trainer
# -----------------------

trainer = BpeTrainer(
    vocab_size=8192,
    min_frequency=2,
    special_tokens=[
        "<|endoftext|>",
    ],
    initial_alphabet=ByteLevel.alphabet(),
    show_progress=True,
)


# -----------------------
# Train
# -----------------------
print("[Tokenizer] Tokenizer train: start") 
tokenizer.train_from_iterator(
    story_iterator(train_stories),
    trainer=trainer,
    length=len(train_stories),
)


# -----------------------
# Save
# -----------------------
tokenizer.save(tokenizer_out_path) 

print("[Training] Tokenizer train: Completed")