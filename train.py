import torch
from tqdm import tqdm
import numpy as np
import apex
device = torch.device('cuda')

class Trainer:
    def __init__(self, CFG, scheduler = None):
        self.CFG = CFG
        self.scheduler = scheduler

    def train_one_epoch(self, model, optim, train_loader, loss_fn, epoch, soft=False):
        model = model.train();
        running_loss = None
        preds_all = []
        targets_all = []
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
        for step, (x, y_true) in pbar:
            x = x.to(device).float()
            y_true = y_true.to(device)
            if not self.CFG['soft_labels_file']:
                y_true = y_true.long()
            y_pred = model(x)
            l = loss_fn(y_pred, y_true)
            optim.zero_grad()
            if self.CFG['fp16']:
                with apex.amp.scale_loss(l, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                l.backward()
            optim.step()
            preds_all += [torch.argmax(y_pred, 1).detach().cpu().numpy()]
            if soft:
                y_true = torch.argmax(y_true, 1)
            targets_all += [y_true.detach().cpu().numpy()]
            if running_loss is None:
                running_loss = l.item()
            else:
                running_loss = running_loss * .99 + l.item() * .01

            if ((step + 1) % self.CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'tain epoch {epoch} loss: {running_loss:.4f}'
                pbar.set_description(description)

        if self.scheduler is not None:
            self.scheduler.step()
        preds_all = np.concatenate(preds_all)
        targets_all = np.concatenate(targets_all)
        print("Target acc = ", (preds_all == targets_all).mean())
        with open(self.CFG["log_file"], 'a+') as logger:
            logger.write(f"Epoch # {epoch}, train acc = {(preds_all == targets_all).mean()}, ")


    def valid_one_epoch(self, model, val_loader, loss_fn, epoch):
        loss_sum = 0
        sample_num = 0
        preds_all = []
        targets_all = []
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        with torch.no_grad():
            model = model.eval();
            for step, (x, y_true) in pbar:
                x = x.to(device).float()
                y_true = y_true.to(device).long()
                y_pred = model(x)
                preds_all += [torch.argmax(y_pred, 1).detach().cpu().numpy()]
                targets_all += [y_true.detach().cpu().numpy()]
                l = loss_fn(y_pred, y_true)
                loss_sum += l.item() * y_true.shape[0]
                sample_num += y_true.shape[0]

            if ((step + 1) % self.CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
                description = f'val epoch {epoch} loss: {loss_sum / sample_num:.4f}'
                pbar.set_description(description)

        preds_all = np.concatenate(preds_all)
        targets_all = np.concatenate(targets_all)
        print('validation multi-class accuracy = {:.4f}'.format((preds_all == targets_all).mean()))
        with open(self.CFG['log_file'], 'a+') as logger:
            logger.write(f"val acc = {(preds_all == targets_all).mean()}\n")
        return (preds_all == targets_all).mean(), loss_sum / sample_num