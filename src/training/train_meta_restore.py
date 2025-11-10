from src.meta.maml import MAMLTrainer
from src.models.descriptor_wrapper import DescriptorWrapper
from src.utils.logger import init_wandb, log_metrics
from src.datasets.hpatches_loader import get_meta_tasks
import torch.nn as nn

def train_meta(config):
    wandb = init_wandb(run_name="meta_restore", config=config)
    model = DescriptorWrapper(model_name="superpoint", pretrained_path=config["pretrained_path"]).cuda()
    maml = MAMLTrainer(model, lr_inner=config["meta_lr_inner"], lr_outer=config["meta_lr_outer"])
    loss_fn = nn.TripletMarginLoss(margin=1.0)

    for epoch in range(config["meta_epochs"]):
        tasks = get_meta_tasks(config["tasks"], meta_batch_size=config["meta_batch_size"])
        loss_val = maml.meta_train_step(tasks, loss_fn)
        log_metrics("meta", {"outer_loss": loss_val}, step=epoch)
