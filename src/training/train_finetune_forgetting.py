import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.logger import init_wandb, log_metrics, finish
from src.models.descriptor_wrapper import DescriptorWrapper
from src.datasets.hpatches_loader import get_dataloader

def train_finetune(config):
    wandb = init_wandb(run_name="finetune_forgetting", config=config)
    model = DescriptorWrapper(model_name="superpoint", pretrained_path=config["pretrained"]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.TripletMarginLoss(margin=1.0)
    loader = get_dataloader(config["train_domain"], batch_size=config["batch_size"])

    for epoch in range(config["epochs"]):
        losses = []
        for batch in tqdm(loader):
            anchor, pos, neg = [b.cuda() for b in batch]
            loss = criterion(model(anchor), model(pos), model(neg))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        log_metrics("train", {"loss": sum(losses)/len(losses)}, step=epoch)
    torch.save(model.state_dict(), config["save_path"])
    finish()
