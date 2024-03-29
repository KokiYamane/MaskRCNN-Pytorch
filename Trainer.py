from cmath import nan
import torch
import wandb
import time
import os
import shutil

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class Tranier():
    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        calc_loss: callable,
        optimizer: torch.optim.Optimizer,
        out_dir: str,
        wandb_flag: bool = False,
        gpu: list = [0],
        early_stopping_count: int = 1e10,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.calc_loss = calc_loss
        self.optimizer = optimizer
        self.out_dir = out_dir
        self.wandb_flag = wandb_flag
        self.early_stopping_count = early_stopping_count
        self.lr_scheduler = lr_scheduler

        self.train_losses = []
        self.valid_losses = []
        self.total_elapsed_time = 0
        self.early_stopping_counter = 0
        self.best_test = 1e10

        self.fig_loss = plt.figure(figsize=(10, 10))

        # device setting
        self.device = torch.device(f'cuda:{gpu[0]}'
                                   if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        print('device:', self.device)
        if torch.cuda.device_count() > 1 and len(gpu) > 1:
            print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
            model = torch.nn.DataParallel(model, device_ids=gpu)
        model.to(self.device)

        # acceleration
        self.scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True

        # make directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print(f'save to {out_dir}')

    def _print_progress_bar(
        self,
        i: int,
        length: int,
        start_time: time.time,
        width: int = None,
        end: str = '\n',
        header: str = '',
        footer: str = '',
    ):
        digits = len(str(length))
        i_str = format(i + 1, '0' + str(digits))
        now = time.time()
        elapsed_time = now - start_time
        total_time = elapsed_time / (i + 1) * length
        footer_count = f'{i_str}/{length}'
        footer_time = f'({elapsed_time:.2f} [s] / {total_time:.2f} [s])'
        footer = f'{footer_count} {footer_time}  {footer}'

        if width is None:
            terminal_size = shutil.get_terminal_size()
            width = terminal_size.columns - len(header) - len(footer) - 5

        if i >= length - 1:
            progress_bar = '=' * width
            end = end
        else:
            num = round(i / (length - 1) * width)
            progress_bar = '=' * (num - 1) + '>' + ' ' * (width - num)
            end = ''

        print('\r\033[K' + f'{header} [{progress_bar}] {footer}', end=end)

    def train(
        self,
        n_epochs: int,
        callback: callable(int) = lambda x: None,
    ):
        for epoch in range(n_epochs + 1):
            start = time.time()

            # train
            running_loss = 0.0
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                self._print_progress_bar(
                    i,
                    len(self.train_loader),
                    start_time=start,
                    end='',
                    header=f'epoch: {epoch} (train)',
                )

                with torch.cuda.amp.autocast():
                    loss = self.calc_loss(batch)

                self.optimizer.zero_grad()

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # loss.backward()
                # self.optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(self.train_loader)
            self.train_losses.append(train_loss)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # valid
            valid_start = time.time()
            running_loss = 0.0
            self.model.eval()
            for i, batch in enumerate(self.valid_loader):
                self._print_progress_bar(
                    i,
                    len(self.valid_loader),
                    start_time=valid_start,
                    end='',
                    header=f'epoch: {epoch} (valid)',
                )

                with torch.no_grad():
                    loss = self.calc_loss(batch, valid=True)
                running_loss += loss.item()
            valid_loss = running_loss / len(self.valid_loader)
            self.valid_losses.append(valid_loss)

            # print
            end = time.time()
            elapsed_time = end - start
            self.total_elapsed_time += elapsed_time
            log = '\r\033[K' + f'epoch: {epoch}'
            log += f'  train loss: {train_loss:.6f}'
            log += f'  valid loss: {valid_loss:.6f}'
            log += f'  elapsed time: {elapsed_time:.3f}'
            log += f'  early stopping: {self.early_stopping_counter}'
            if self.lr_scheduler is not None:
                log += f'  lr: {self.lr_scheduler.get_last_lr()[-1]}'
            print(log)

            # save model
            if epoch % 100 == 0 or epoch < 100:
                encoder_param_dir = os.path.join(self.out_dir, 'model_param')
                if not os.path.exists(encoder_param_dir):
                    os.mkdir(encoder_param_dir)
                path_encoder_param = os.path.join(
                    encoder_param_dir,
                    f'model_param_{epoch:06d}.pt')
                torch.save(self.model.state_dict(), path_encoder_param)

                # if self.wandb_flag:
                #     wandb.save(path_encoder_param)

            # save checkpoint
            path_checkpoint = os.path.join(self.out_dir, 'checkpoint.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }, path_checkpoint)

            path_model_param_latest = os.path.join(
                self.out_dir, 'model_param_latest.pt')
            torch.save(self.model.state_dict(), path_model_param_latest)

            # wandb
            if self.wandb_flag:
                wandb.log({
                    'epoch': epoch,
                    'iteration': len(self.train_loader) * epoch,
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                })
                if self.lr_scheduler is not None:
                    wandb.log({
                        'epoch': epoch,
                        'learning_rate': self.lr_scheduler.get_last_lr()[-1],
                    })
                wandb.save(path_checkpoint)
                wandb.save(path_model_param_latest)

            if train_loss <= self.best_test:
                self.best_test = train_loss
            # if valid_loss <= self.best_test:
            #     self.best_test = valid_loss
                self.early_stopping_counter = 0

                # save model
                path_model_param_best = os.path.join(
                    self.out_dir, 'model_param_best.pt')
                torch.save(self.model.state_dict(), path_model_param_best)
                if self.wandb_flag:
                    wandb.save(path_model_param_best)

            else:
                # Early Stopping
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_count or \
                        valid_loss is nan:
                    print('Early Stopping!')
                    break

            # plot loss
            self.fig_loss.clf()
            ax = self.fig_loss.add_subplot(111)
            ax.plot(self.train_losses, label='train')
            ax.plot(self.valid_losses, label='valid')
            ax.legend()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            self.fig_loss.savefig(os.path.join(self.out_dir, 'loss.png'))

            callback(epoch)

        print(f'total elapsed time: {self.total_elapsed_time} [s]')
