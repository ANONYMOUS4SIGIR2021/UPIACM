from data_loader import DataLoader
from model import UPIACM
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, f1_score


class RunConfig(object):
    def __init__(self, args):
        self.args = args

        self.dataset = args.dataset

        self.lr = args.lr
        self.epoch = args.epoch
        self.wd = args.wd

        self.data = self._prepare_data()
        self.model, self.criterion = self._prepare_model()
        self.optimizer = self._prepare_optimizer()

    def _prepare_data(self):
        print('prepare data ...')
        return DataLoader(args=self.args)

    def _prepare_model(self):
        print('prepare model ...')
        model = UPIACM(
            args=self.args,
            n_user=self.data.n_user,
            n_item=self.data.n_item,
            n_entity=self.data.n_entity,
            n_relation=self.data.n_relation
        )
        criterion = nn.BCELoss()
        model.cuda()
        criterion.cuda()

        return model, criterion

    def _prepare_optimizer(self):
        print('prepare optimizer ...')
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), self.lr,
                                     weight_decay=self.wd)
        return optimizer

    def _unpack_batch(self, batch):
        return [Variable(temp).cuda() for temp in batch]

    def train(self):
        for epoch in range(self.epoch):
            n_batch = len(self.data.train_batches)
            for i in range(n_batch):
                self.model.train()
                self.optimizer.zero_grad()

                inputs = self._unpack_batch(self.data.train_batches[i])
                logits = self.model(inputs)
                loss = self.criterion(logits, inputs[-1].float())

                loss.backward()
                self.optimizer.step()

                if i == int(n_batch / 2) or i == (n_batch - 1):
                    print('============Epoch/phase: {}/{}============'.format(epoch, i))
                    self.model.eval()
                    train_auc, train_acc, train_f1 = self._evaluation(self.data.train_batches)
                    eval_auc, eval_acc, eval_f1 = self._evaluation(self.data.eval_batches)
                    test_auc, test_acc, test_f1 = self._evaluation(self.data.test_batches)
                    print(
                        'train auc: %.4f acc: %.4f f1: %.4f   eval auc: %.4f acc: %.4f f1: %.4f  test auc: %.4f acc: %.4f f1: %.4f'
                        % (train_auc, train_acc, train_f1, eval_auc, eval_acc, eval_f1, test_auc, test_acc, test_f1))

    def _evaluation(self, data):
        auc_list = []
        acc_list = []
        f1_list = []
        total = len(data)
        for i in range(total):
            inputs = self._unpack_batch(data[i])
            logits = self.model(inputs)
            auc, acc, f1 = self._get_metrics(logits.detach().cpu().numpy(), inputs[-1].cpu().numpy())
            auc_list.append(auc)
            acc_list.append(acc)
            f1_list.append(f1)
        return float(np.mean(auc_list)), float(np.mean(acc_list)), float(np.mean(f1_list))

    def _get_metrics(self, logits, labels):
        auc = roc_auc_score(y_true=labels, y_score=logits)
        predictions = [1 if i >= 0.5 else 0 for i in logits]
        acc = np.mean(np.equal(predictions, labels))
        f1 = f1_score(y_true=labels, y_pred=predictions)
        return auc, acc, f1
