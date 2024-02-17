# imports
from datetime import datetime
import os
import pprint

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler

from datasets import load_dataset, Dataset
from tokenizers import Tokenizer

from tqdm.auto import tqdm
from nano_gpt_model import NanoGPT


# -----------------------------
def print_cuda_memory_stats():
    print(
        "torch.cuda.memory_allocated: %fGB"
        % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    )
    print(
        "torch.cuda.memory_reserved: %fGB"
        % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
    )
    print(
        "torch.cuda.max_memory_reserved: %fGB"
        % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)
    )


# -----------------------------
# setup cuda
torch.cuda.memory._record_memory_history(enabled=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device("cpu")
print(f"using {device}")
# -----------------------------

# Create checkpoint directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
checkpoint_dir = f"checkpoints/{timestamp}"
os.mkdir(checkpoint_dir)

# load dataset
dataset = load_dataset("roneneldan/TinyStories")
dataset_local = load_dataset(
    "text",
    data_files={
        # "train": r"..\..\datasets\TinyStories\TinyStoriesV2-GPT4-train-100k-lines.txt",
        "train": r"..\..\datasets\TinyStories\TinyStoriesV2-GPT4-train.txt",  # Full original
        "validation": r"..\..\datasets\TinyStories\TinyStoriesV2-GPT4-valid.txt",
    },
)
tokenizer_file = "data/TinyStories-tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_file)
# -----------------------------

# hyperparameters
hyperparameters = {
    "n_epochs": 3,
    "vocab_size": tokenizer.get_vocab_size(),
    "batch_size": 8,
    "block_size": 1080,
    "learning_rate": 5e-4,
    "n_embed": 256,
    "n_heads": 8,  # Must be an even divisor of n_embed
    "n_layers": 8,
    "dropout": 0.1,
}

n_epochs = hyperparameters["n_epochs"]
vocab_size = hyperparameters["vocab_size"]
batch_size = hyperparameters["batch_size"]
block_size = hyperparameters["block_size"]
learning_rate = hyperparameters["learning_rate"]
n_embed = hyperparameters["n_embed"]
n_heads = hyperparameters["n_heads"]
n_layers = hyperparameters["n_layers"]
dropout = hyperparameters["dropout"]

# Training parameters (Must be an even multiple of n_lossi_bins * batch_size for lossi plotting)
n_train = 80000
n_lossi_bins = 25
assert n_train % (n_lossi_bins * batch_size) == 0
# -----------------------------

# tokenize dataset
tokenizer.enable_padding(pad_id=2, pad_token="<|im_end|>", length=block_size)
tokenizer.enable_truncation(max_length=block_size)
tokenized_data = dataset.map(
    lambda x: {"input_ids": [elem.ids for elem in tokenizer.encode_batch(x["text"])]},
    batched=True,
)
tokenized_data = tokenized_data.with_format("torch")

train_ids = tokenized_data["train"].remove_columns(["text"])
train_ids = train_ids.shuffle().select(range(n_train))

val_ids = tokenized_data["validation"].remove_columns(["text"])
val_ids = val_ids.shuffle().select(range(int(n_train / 10)))
# -----------------------------

# setup model and trainer
model = NanoGPT(hyperparameters, device).to(device)
train_dataloader = DataLoader(train_ids, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_ids, batch_size=batch_size, shuffle=True)

optimizer = AdamW(model.parameters(), lr=learning_rate)
num_params = sum(p.numel() for p in model.parameters()) / 1e6

num_training_steps = n_epochs * len(train_dataloader)
scheduler = lr_scheduler.OneCycleLR(
    optimizer=optimizer, max_lr=learning_rate, total_steps=num_training_steps
)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
print(f"{num_params:.3f}M parameters")
# -----------------------------

# load checkpoint
# checkpoint = torch.load("checkpoints/6head-1.452M-checkpoint-0.pt")
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# scheduler.load_state_dict(checkpoint['scheduler'])
# saved_epoch = checkpoint['epoch']

# num_training_steps = (n_epochs - (saved_epoch + 1)) * len(train_dataloader)

saved_epoch = None
# -----------------------------

# train model
lossi = []
lri = []
progress_bar = tqdm(range(num_training_steps))

print_cuda_memory_stats()
start_time = datetime.now()

for epoch in range(n_epochs):
    model.train()  # switch model to training mode
    if saved_epoch != None and epoch <= saved_epoch:
        continue

    for batch in train_dataloader:
        batch = batch["input_ids"].to(device)
        targets = torch.concat(
            (batch[:, 1:], 2 * torch.ones([batch.shape[0], 1]).to(device)), dim=1
        ).long()
        logits, loss = model(batch, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()
        progress_bar.update(1)

        if progress_bar.n % 500 == 0:
            print(
                f"erratic train loss: {loss.item()} lr: {optimizer.param_groups[0]['lr']}"
            )
        lossi.append(loss.log10().item())
        lri.append(optimizer.param_groups[0]["lr"])

    with torch.no_grad():
        # evaluate validation loss
        model.eval()  # switch model to evaluation mode
        losses = torch.zeros(len(val_dataloader), device=device)
        k = 0
        for batch in val_dataloader:
            batch = batch["input_ids"].to(device)
            targets = torch.concat(
                (batch[:, 1:], 2 * torch.ones([batch.shape[0], 1]).to(device)), dim=1
            ).long()
            logits, loss = model(batch, targets)

            losses[k] = loss.item()
            predictions = torch.argmax(logits, dim=-1)
            k += 1

        avg_val_loss = losses.mean()
        print(f"val loss: {avg_val_loss}")
        # -----------------------------

        # evaluate training loss
        losses = torch.zeros(len(val_dataloader), device=device)
        k = 0
        for batch in train_dataloader:
            batch = batch["input_ids"].to(device)
            targets = torch.concat(
                (batch[:, 1:], 2 * torch.ones([batch.shape[0], 1]).to(device)), dim=1
            ).long()
            logits, loss = model(batch, targets)

            losses[k] = loss.item()
            predictions = torch.argmax(logits, dim=-1)
            k += 1

            if k == len(val_dataloader):
                break

        avg_train_loss = losses.mean()
        print(f"train loss: {avg_train_loss}")
        # print(torch.cuda.memory_summary())
        # -----------------------------

        # save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "hyperparameters": hyperparameters,
            "val_loss": avg_val_loss,
            "train_loss": avg_train_loss.item(),
        }
        checkpoint_name = (
            f"{checkpoint_dir}/{n_epochs}-epoch-{num_params:.3f}M-checkpoint-{epoch}"
        )
        print(f"saving {checkpoint_name}")
        torch.save(checkpoint, f"{checkpoint_name}.pt")
        # -----------------------------
# -----------------------------
torch.cuda.memory._dump_snapshot(f"{checkpoint_name}-cuda_snapshot.pickle")
print(torch.cuda.memory_summary())

# Write log info
with open(f"{checkpoint_name}.log", mode="w+t") as log_file:
    log_file.write(f"Elapsed time: {(datetime.now()-start_time)} h:m:s\n")
    log_file.write(f"Final train loss: {losses[-1].item()}\n")
    log_file.write(f"Average train loss: {avg_train_loss}\n")
    log_file.write(f"Final val loss: {avg_val_loss}\n")
    pprint.pprint(hyperparameters, log_file)
    log_file.write(f"n_train = {n_train}\n")
    log_file.write(
        f"CUDA max_memory_allocated: {torch.cuda.max_memory_allocated()/1e9}GB\n"
    )
    log_file.write(f"losses:\n")
    pprint.pprint(losses.tolist(), log_file)
    log_file.write(f"lossi:\n")
    pprint.pprint(lossi, log_file)
    log_file.write(torch.cuda.memory_summary())

print(f"Checkpoint files are in {checkpoint_dir}")

from matplotlib import pyplot as plt

# Take the list of log10 losses and divide into n_lossi_bins bins then compute mean of each bin
plt.plot(torch.tensor(lossi).view(-1, min(len(lossi), n_lossi_bins)).mean(1))
plt.plot(losses.tolist())
plt.savefig(f"{checkpoint_name}.png")
plt.show()
