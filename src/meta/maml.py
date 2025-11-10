import torch
import higher

class MAMLTrainer:
    def __init__(self, model, lr_inner=1e-3, lr_outer=1e-4):
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.outer_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_outer)

    def meta_train_step(self, tasks, loss_fn):
        meta_loss = 0.0
        for task in tasks:
            support_x, support_y, query_x, query_y = task
            self.model.train()
            with higher.innerloop_ctx(self.model, torch.optim.SGD(self.model.parameters(), lr=self.lr_inner)) as (fmodel, diffopt):
                for _ in range(3):
                    support_preds = fmodel(support_x)
                    loss = loss_fn(support_preds, support_y)
                    diffopt.step(loss)
                query_preds = fmodel(query_x)
                task_loss = loss_fn(query_preds, query_y)
                meta_loss += task_loss

        meta_loss /= len(tasks)
        self.outer_optimizer.zero_grad()
        meta_loss.backward()
        self.outer_optimizer.step()
        return meta_loss.item()
