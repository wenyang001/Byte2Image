import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from FIFTY_DATASET import FIFTY_DATASET
from visuliaze import Visualizer
from sklearn.metrics import confusion_matrix
from torchvision import transforms

from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from wide_deep_new import SimpleRes_4k
import os

BATCH_SIZE = 512
TEST_BATCH_SIZE = 1024
LOAD_MODEL_LOC = None
SAVE_MODEL_LOC = "model_"
PRINT = True
PRINT_GRAPH = False
PRINT_CM = False
SAVE = True

SCEN = "1"
NUMBER_OF_ClASS = {
    "0": 16,
    "1": 75,
    "2": 11,
    "3": 25,
    "4": 5,
    "5": 2,
    "6": 2
}
NUMBER_OF_LABELS = {
    "0": ['jpg', 'gif', 'doc', 'xls', 'ppt', 'html', 'text', 'pdf', 'rtf', 'png', 'log', 'csv', 'gz', 'swf', 'eps',
          'ps'],
    "1": ["jpg", "arw", "cr2", "dng", "gpr", "nef", "nrw", "orf", "pef", "raf", "rw2", "3fr", "tiff", "heic", "bmp",
          "gif", "png", "ai", "eps", "psd", "mov", "mp4", "3gp", "avi", "mkv", "ogv", "webm", "apk", "jar", "msi",
          "dmg", "7z", "bz2", "deb", "gz", "pkg", "rar", "rpm", "xz", "zip", "exe", "mach-o", "elf", "dll", "doc",
          "docx", "key", "ppt", "pptx", "xls", "xlsx", "djvu", "epub", "mobi", "pdf", "md", "rtf", "txt", "tex", "json",
          "html", "xml", "log", "csv", "aiff", "flac", "m4a", "mp3", "ogg", "wav", "wma", "pcap", "ttf", "dwg",
          "sqlite"],
    "2": ["bmp", "raw", "vec", "vid", "arc", "exe", "off", "pub", "hr", "aud", "oth"],
    "3": ["jpg", "arw", "cr2", "dng", "gpr", "nef", "nrw", "orf", "pef", "raf", "rw2", "3fr", "tiff", "heic", "bmp",
          "gif", "png", "mov", "mp4", "3gp", "avi", "mkv", "ogv", "webm", "oth"],
    "4": ["jpg", "raw", "vid", "5_bmps", "oth"],
    "5": ["jpg", "oth"],
    "6": ["jpg", "oth"]
}


def create_data_loader(opt, scen: str):
    train_loader, test_loader, val_loader, train_dataset = None, None, None, None

    if opt.phase == 'train':
        train_transform = transforms.Compose([
            # Cutout(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomErasing(),
        ])

        if opt.data == '512':
            train_dataset = FIFTY_DATASET('/media/liu/Data/DATASETS/', 'train', scenario=scen, multi=True,
                                          transform=train_transform)
            val_dataset = FIFTY_DATASET('/media/liu/Data/DATASETS/', 'val', scenario=scen, multi=True,
                                        transform=None)

        else:
            train_dataset = FIFTY_DATASET('/media/liu/Data/DATASETS/', 'train', block_size='4k', scenario=scen,
                                          multi=True,
                                          transform=train_transform)
            val_dataset = FIFTY_DATASET('/media/liu/Data/DATASETS/', 'val', block_size='4k', scenario=scen,
                                        multi=True,
                                        transform=None)
        train_loader = DataLoader(
            train_dataset,
            pin_memory=True,
            batch_size=BATCH_SIZE,
            num_workers=8,
            drop_last=True,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            pin_memory=True,
            batch_size=TEST_BATCH_SIZE,
            num_workers=8
        )

    else:
        if opt.data == '512':
            test_dataset = FIFTY_DATASET('/media/liu/Data/DATASETS/', 'test', scenario=scen, multi=True,
                                         transform=None)
        else:
            test_dataset = FIFTY_DATASET('/media/liu/Data/DATASETS/', 'test', block_size='4k', scenario=scen,
                                         multi=True,
                                         transform=None)

        test_loader = DataLoader(
            test_dataset,
            pin_memory=True,
            batch_size=TEST_BATCH_SIZE,
            num_workers=8
        )

    return train_loader, test_loader, val_loader, train_dataset



# measures accuracy of predictions at the end of an epoch (bad for semantic segmentatio
def accuracy(model, loader, num_classes=NUMBER_OF_ClASS[SCEN]):
    num = 30
    pred, truth = [], []
    with torch.no_grad():
        for i, (x, y, z) in enumerate(loader):
            if i > num:
                break

            x, y, z = x.to(DEVICE), y.to(DEVICE), z.to(DEVICE)

            y_ = model(z, x)
            y_ = torch.argmax(y_, dim=1)

            pred.append(y_.cpu().numpy())
            truth.append(y.cpu().numpy())

    if len(pred[0]) > 1:
        pred = np.concatenate(pred, 0).squeeze()
        truth = np.concatenate(truth, 0).squeeze()

    correct = (pred == truth).sum()
    class_labels = list(range(num_classes))
    C1 = confusion_matrix(truth, pred, labels=class_labels)

    if PRINT_CM:
        sn.set(rc={'figure.figsize': (30, 30)})
        sn.heatmap(C1, fmt='g', annot=True, cbar=False, xticklabels=class_labels, yticklabels=class_labels)
        plt.show()

    return (correct / (num * BATCH_SIZE)).item() * 100

def test(model, loader):
    model.eval()
    num_classes = NUMBER_OF_ClASS[SCEN]
    pred, truth = [], []
    class_labels = list(range(num_classes))
    labels = NUMBER_OF_LABELS[SCEN]
    cur_correct = 0
    total = 0

    with torch.no_grad():
        progress = tqdm(loader)
        for i, (x, y, z) in enumerate(progress):

            x, y, z = x.to(DEVICE), y.to(DEVICE), z.to(DEVICE)
            y_ = model(z, x)
            y_ = torch.argmax(y_, dim=1)

            pred.append(y_.cpu().numpy())
            truth.append(y.cpu().numpy())

            cur_correct += (y_.cpu().numpy() == y.cpu().numpy()).sum()
            total += y.size(0)

            # make the progress bar display loss
            progress.set_postfix({'ACC': cur_correct.item() / total, 'correct': cur_correct.item(), 'num': i})

    if len(pred[0]) > 1:
        pred = np.concatenate(pred, 0).squeeze()
        truth = np.concatenate(truth, 0).squeeze()

    # 当前 SCEN 的标签类别
    labels = NUMBER_OF_LABELS[SCEN]

    # JPEG 类型的索引（根据当前 SCEN）
    jpeg_idx = labels.index("jpg") if "jpg" in labels else None

    # 计算整体 accuracy
    overall_accuracy = np.mean(pred == truth)

    # 计算 JPEG 类型的 accuracy
    if jpeg_idx is not None:
        jpeg_mask = (truth == jpeg_idx)
        jpeg_accuracy = np.mean(pred[jpeg_mask] == truth[jpeg_mask]) if np.any(jpeg_mask) else 0.0
    else:
        jpeg_accuracy = None

    # 打印结果
    print(f"Overall Accuracy: {overall_accuracy:.2%}", end=" ")
    if jpeg_accuracy is not None:
        print(f"JPEG Accuracy: {jpeg_accuracy:.2%}")
        return overall_accuracy, jpeg_accuracy
    else:
        print("JPEG Accuracy: No JPEG labels in the current SCEN.")
        return overall_accuracy, None


def opt_parse():
    import argparse
    parser = argparse.ArgumentParser(description='Byte classification.')
    parser.add_argument('--name', type=str, default='512', help='512, 4096')
    parser.add_argument('--scen', type=str, default='1', help='1,2,3,4,5,6')
    parser.add_argument('--data', type=str, default='512', help='512, 4096')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--which_epoch', type=str, default='2',
                        help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=60000,
                        help='frequency of saving the latest results')

    # parser.add_argument('--weight_decay', default=1e-4, type=float)
    # parser.add_argument('--lr', default=1e-4, type=float)


    # mixup/cutmix
    parser.add_argument('--mixup_active', action='store_true', default=True,
                       help='disable mixup training')  # 触发为真
    parser.add_argument('--mixup', type=float, default=0.8,
                       help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                       help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                       help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                       help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                       help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                       help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                       help='Turn off mixup after this epoch, disabled if 0 (default: 0)')

    parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')


    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    opt = opt_parse()
    SCEN = opt.scen

    from config import get_config
    from scheduler import build_scheduler
    from optimizer import build_optimizer

    ngram = 16
    config = get_config()
    model = SimpleRes_4k(
        image_size=(ngram * 8, 512 - ngram),
        num_classes=NUMBER_OF_ClASS[SCEN],
        in_channel=8
    )

    train_loader, test_loader, val_loader, train_dataset = create_data_loader(opt, SCEN)

    DEVICE = "cuda:0"
    model = torch.nn.DataParallel(model.to(DEVICE), device_ids=list(map(int, str.split(opt.gpu_ids, ','))))
    vis = Visualizer(os.path.join('./checkpoints/', opt.name))

    if opt.mixup_active:
        mixup_args = dict(
            mixup_alpha=opt.mixup, cutmix_alpha=opt.cutmix, cutmix_minmax=opt.cutmix_minmax,
            prob=opt.mixup_prob, switch_prob=opt.mixup_switch_prob, mode=opt.mixup_mode,
            label_smoothing=opt.smoothing, num_classes=NUMBER_OF_ClASS[SCEN])
        mixup_fn = Mixup(**mixup_args)

    # setup loss function
    if opt.mixup_active:
        loss_function = SoftTargetCrossEntropy()
    else:
        loss_function = nn.CrossEntropyLoss()

    if not os.path.exists(os.path.join('./checkpoints/', opt.name, 'model')):
        os.mkdir(os.path.join('./checkpoints/', opt.name, 'model'))
        os.mkdir(os.path.join('./checkpoints/', opt.name, 'optimizer'))
        os.mkdir(os.path.join('./checkpoints/', opt.name, 'lr'))

    # optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizer = build_optimizer(config, model)

    if opt.phase == 'train':
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
        start_epoch, epoch_iter = 1, 0
        iter_path = os.path.join('./checkpoints/', opt.name, 'iter.txt')

        if opt.continue_train:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
            print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
            checkpoint = {
                'model': os.path.join('./checkpoints/', opt.name, 'model/model_latest'),
                'optimizer': os.path.join('./checkpoints/', opt.name, 'optimizer/optimizer_latest'),
                'lr_scheduler': os.path.join('./checkpoints/', opt.name, 'lr/lr_latest')
            }
            model.load_state_dict(torch.load(checkpoint['model']))
            optimizer.load_state_dict(torch.load(checkpoint['optimizer']))
            lr_scheduler.load_state_dict(torch.load(checkpoint['lr_scheduler']))
            print(optimizer.param_groups[0]['lr'], 'current_lr')

        dataset_size = len(train_dataset)
        total_steps = (start_epoch - 1) * dataset_size + epoch_iter  # 每个epoch下的停在哪里
        print_delta = total_steps % opt.print_freq
        save_delta = total_steps % opt.save_latest_freq

        for epoch in range(start_epoch, config.TRAIN.EPOCHS):
            print(optimizer.param_groups[0]['lr'], 'current_lr')

            # train
            model.train()
            if epoch != start_epoch:
                epoch_iter = epoch_iter % (BATCH_SIZE * len(train_loader))  # epoch_iter 一个epoch中当前训练到哪里了

            progress = tqdm(train_loader)
            for index, (x, y, z) in enumerate(progress, start=epoch_iter // BATCH_SIZE):

                total_steps += BATCH_SIZE  # 总体当前训练到哪里了
                epoch_iter += BATCH_SIZE

                x, y, z = x.to(DEVICE), y.to(DEVICE), z.to(DEVICE)

                if opt.mixup_active:
                    x, y = mixup_fn(x, y)

                y_ = model(z, x)
                loss = loss_function(y_, y)

                # make the progress bar display loss
                progress.set_postfix({'loss': loss.item(), 'index': index})

                # back propagation
                optimizer.zero_grad()  # zeros out the gradients from previous batch
                loss.backward()
                optimizer.step()
                lr_scheduler.step_update((epoch - 1) * len(train_loader) + index)

                # log for each batch
                vis.plot_current_errors(loss.item(), total_steps)  # if total_steps % opt.print_freq == print_delta:

                ### save latest model
                if total_steps % opt.save_latest_freq == save_delta:
                    torch.save(model.state_dict(), os.path.join('./checkpoints/', opt.name, 'model/model_latest'))
                    torch.save(optimizer.state_dict(),
                               os.path.join('./checkpoints/', opt.name, 'optimizer/optimizer_latest'))
                    torch.save(lr_scheduler.state_dict(), os.path.join('./checkpoints/', opt.name, 'lr/lr_latest'))
                    print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                    np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

                # # break loop
                if epoch_iter >= (BATCH_SIZE * len(train_loader)):
                    epoch_iter = 0
                    break


            if SAVE:
                if epoch < 40 and epoch % 5 == 0:
                    torch.save(model.state_dict(),
                               os.path.join('./checkpoints/', opt.name, 'model/model_' + str(epoch)))
                    torch.save(optimizer.state_dict(),
                               os.path.join('./checkpoints/', opt.name, 'optimizer/optimizer_' + str(epoch)))
                    torch.save(lr_scheduler.state_dict(),
                               os.path.join('./checkpoints/', opt.name, 'lr/lr_' + str(epoch)))
                    print('saving the latest model epoch %d' % epoch)
                    np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
                elif epoch >= 40:
                    torch.save(model.state_dict(),
                               os.path.join('./checkpoints/', opt.name, 'model/model_' + str(epoch)))
                    torch.save(optimizer.state_dict(),
                               os.path.join('./checkpoints/', opt.name, 'optimizer/optimizer_' + str(epoch)))
                    torch.save(lr_scheduler.state_dict(),
                               os.path.join('./checkpoints/', opt.name, 'lr/lr_' + str(epoch)))
                    print('saving the latest model epoch %d' % epoch)
                    np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
                else:
                    pass

            ## val
            model.eval()
            val_acc = accuracy(model, train_loader)
            vis.plot_acc(val_acc, epoch)
            print("Test Accuracy for epoch (" + str(epoch) + ") is: " + str(val_acc))


    else:
        accuracy_per_epoch_test = []
        start, end = 50, 51
        for epoch in range(start, end):
            pretrained_path = './checkpoints/' + opt.name + '/model/model_' + str(epoch)
            model.load_state_dict(torch.load(pretrained_path, map_location='cpu')) # load model
            print("Epoch: ", epoch)
            acc, jpeg_acc = test(model, test_loader)
            accuracy_per_epoch_test.append((acc, jpeg_acc))
                
        print(accuracy_per_epoch_test)
