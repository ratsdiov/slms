# imports
import pprint
import torch
from tokenizers import Tokenizer
from nano_gpt_model import NanoGPT

# -----------------------------

# setup cuda
torch.cuda.memory._record_memory_history(enabled=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device("cpu")
print(f"using {device}")
# -----------------------------

# load tokenizer
tokenizer_file = "data/TinyStories-tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_file)
# -----------------------------

# load model
# path = r"checkpoints/simple-stories-4M.pt"
# path = r"checkpoints/3-epoch-4.484M-checkpoint-2.pt"
path = r"checkpoints\3-epoch-7.640M-checkpoint-1.pt"
path = r"checkpoints\2024-01-31-3-epoch-7.607M-checkpoint-4.pt"
path = r"checkpoints\2024-02-07-3-epoch-7.640M-checkpoint-1.pt"
path = r"checkpoints\3-epoch-4.484M-checkpoint-0.pt"
path = r"checkpoints\2024-02-14-3-epoch-6.062M-checkpoint-1.pt"
checkpoint = torch.load(path)
hyperparameters = checkpoint["hyperparameters"]
pprint.pprint(hyperparameters)
model = NanoGPT(hyperparameters, device).to(device)
model.load_state_dict(checkpoint["model"])
# -----------------------------

# generate text
prompt = "Once upon a time "
context = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
# context = torch.tensor([[314, 324, 66, 283, 14]], dtype=torch.long, device=device)
print(tokenizer.decode(model.generate(context, max_new_tokens=256)[0].tolist()))
# -----------------------------
