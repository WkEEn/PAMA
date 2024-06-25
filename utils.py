import math
import shutil
import logging
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import csv


class Record(object):
    def __init__(self, save_path):
        self.save_path = save_path
        with open(self.save_path, 'w') as f:
            f_w = csv.writer(f)
            f_w.writerow(['Epoch', '[Train]Loss', '[Train]acc1', '[Train]acc2', '[Train]micro_auc', '[Train]macro_auc',
                          '[Train]weighted_auc', '[Val]Loss', '[Val]acc1', '[Val]acc2', '[Val]micro_auc',
                          '[Val]macro_auc', '[Val]weighted_auc', '[Val]f1_micro', '[Val]f1_macro'])

    def update(self, record):
        with open(self.save_path, 'a') as f:
            f_w = csv.writer(f)
            f_w.writerow(record)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ConfusionMatrix(object):
    def __init__(self, classes):
        self.confusion_matrix = torch.zeros(len(classes), len(classes))
        self.classes = classes

    def update_matrix(self, preds, targets):
        # print(preds)
        preds = torch.max(preds, 1)[1].cpu().numpy()
        # preds = torch.softmax(preds.cpu(), dim=-1).detach().numpy()
        # print("====", preds)
        targets = targets.cpu().numpy()
        for p, t in zip(preds, targets):
            self.confusion_matrix[t, p] += 1

    def plot_confusion_matrix(self, normalize=True, save_path='./Confusion Matrix.jpg'):
        cm = self.confusion_matrix.numpy()
        classes = self.classes
        num_classes = len(classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        im = plt.matshow(cm, cmap=plt.cm.Blues)  # cm.icefire
        plt.xticks(range(num_classes), classes)
        plt.yticks(range(num_classes), classes)
        ax = plt.gca()
        ax.xaxis.set_ticks_position('bottom')
        tempMax = 0
        for i in range(len(classes)):
            tempSum = 0
            for j in range(num_classes - 1):
                tempS = cm[i, j] * 100
                tempSum += tempS
                color = 'white' if tempS > 50 else 'black'
                if cm[i, j] != 0:
                    plt.text(j, i, format(tempS, '0.2f'), color=color, ha='center')
            tempS = 100 - tempSum
            tempMax = tempS if tempS > tempMax else tempMax
            color = 'white' if tempS > 50 else 'black'
            if float(format(abs(tempS), '0.2f')) != 0:
                plt.text(num_classes - 1, i, format(abs(tempS), '0.2f'), color=color, ha='center')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)
        cb = plt.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=5)
        cb.set_ticks(np.linspace(0, tempMax / 100., 6))
        cb.set_ticklabels(str("%.2f" % (100 * l)) for l in np.linspace(0, tempMax / 100., 6))

        plt.savefig(save_path)
        plt.close()


class AUCMetric(object):
    def __init__(self, classes):
        self.targets = []
        self.preds = []
        self.classes = np.arange(len(classes))
        self.classes_list = classes

    def update(self, preds, targets):
        preds = torch.softmax(preds.cpu(), dim=-1).detach().numpy()
        targets = targets.cpu().numpy()
        for p, t in zip(preds, targets):
            self.preds.append(p)
            self.targets.append(t)

    def calc_auc_score(self):
        preds = np.array(self.preds)
        targets = label_binarize(np.array(self.targets), classes=self.classes)
        micro_auc = metrics.roc_auc_score(targets, preds, average='micro')
        macro_auc = metrics.roc_auc_score(targets, preds, average='macro')
        weighted_auc = metrics.roc_auc_score(targets, preds, average='weighted')
        return micro_auc, macro_auc, weighted_auc

    def calc_binary_auc_score(self):
        preds = np.array(self.preds)
        targets = label_binarize(np.array(list(self.targets)), classes=self.classes)
        auc = metrics.roc_auc_score(targets, preds[:, 1])

        return auc


    def calc_f1_score(self):
        preds = np.array(self.preds)
        targets = np.array(self.targets)
        f1_micro = metrics.f1_score(targets, np.argmax(preds, axis=1), average='micro')
        f1_macro = metrics.f1_score(targets, np.argmax(preds, axis=1), average='macro')
        return f1_micro, f1_macro
    

    def plot_micro_roc_curve(self, save_path):
        preds = np.array(self.preds)
        targets = label_binarize(np.array(self.targets), classes=self.classes)
        print(preds, "------")
        print(targets)
        fpr, tpr, thresholds, = metrics.roc_curve(targets.ravel(), preds.ravel())
        auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label='AUC={:.3f}'.format(auc))
        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.savefig(save_path)
        plt.close()

    def plot_every_class_roc_curve(self, save_path):
        preds = np.array(self.preds)
        targets = label_binarize(np.array(self.targets), classes=self.classes)
        fpr = dict()
        tpr = dict()
        auc = dict()
        if len(self.classes) == 5:
            colors = ["aqua", "darkorange", "cornflowerblue", "navy", "deeppink"]
        if len(self.classes) == 9 or 8:
            colors = ["aqua", "darkorange", "cornflowerblue", "navy", "deeppink", "blue", "purple", "green", "gray"]
        for i, color in zip(range(len(self.classes)), colors):
            fpr[i], tpr[i], thresholds, = metrics.roc_curve(targets[:, i], preds[:, i])
            auc[i] = metrics.auc(fpr[i], tpr[i])
            plt.plot(
                fpr[i],
                tpr[i],
                ls="--",
                color=color,
                lw=2,
                alpha=0.7,
                label="ROC of {0} (area={1:0.2f})".format(self.classes_list[i], auc[i]),
            )

        # plot micro_roc_curve
        fpr_micro, tpr_micro, thresholds_micro, = metrics.roc_curve(targets.ravel(), preds.ravel())
        auc_micro = metrics.auc(fpr_micro, tpr_micro)

        plt.plot(fpr_micro, tpr_micro, c='r', lw=2, alpha=0.7,
                 label="AUC (area = {:.3f})".format(auc_micro))

        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.savefig(save_path)
        plt.close()

    def plot_binary_roc_curve(self, save_path):
        preds = np.array(self.preds)
        targets = np.array(self.targets)
        fpr, tpr, thresholds, = metrics.roc_curve(targets.ravel(), preds[:, 1].ravel())
        auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label='AUC={:.3f}'.format(auc))
        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.savefig(save_path)
        plt.close()