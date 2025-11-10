import wandb
from datetime import datetime

def init_wandb(project="meta-feature-matching", run_name=None, config=None):
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(project=project, name=run_name, config=config)
    return wandb

def log_metrics(split, metrics, step):
    wandb.log({f"{split}/{k}": v for k, v in metrics.items()} | {f"{split}/step": step})

def log_images(split, images, captions, step):
    wandb.log({f"{split}/examples": [wandb.Image(im, caption=c) for im, c in zip(images, captions)]}, step=step)
