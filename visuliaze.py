from torch.utils.tensorboard import SummaryWriter
import os


class Visualizer():
    def __init__(self, log_dir):
        self.log_dir = os.path.join(log_dir, 'logs')
        self.writer = SummaryWriter(self.log_dir)

    def plot_current_errors(self, errors, step):
        self.writer.add_scalar("Total Loss", errors, step)

    def plot_acc(self, acc, epoch):
        print(acc, epoch)
        self.writer.add_scalar("Total Accuracy", acc, epoch)