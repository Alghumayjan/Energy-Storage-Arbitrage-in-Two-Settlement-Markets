from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DLinear, PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
            'PatchTST': PatchTST,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        ##################### for rescale #####################
        self.Data = data_set
        ##################### for rescale #####################
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_test_loss = 0
        m_test = 0
        mse_test_ls = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder
                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim].to(self.device)
                
                # mse test
                test_loss = nn.MSELoss()(outputs, batch_y)
                total_test_loss += test_loss.item() * batch_x.size(0)
                m_test += batch_x.size(0)
                mse_test_ls.append(total_test_loss / m_test)

        self.model.train()
        return np.average(mse_test_ls)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
            
        # path for saving the training progrgess
        training_path = './training/' + setting + '/'
        if not os.path.exists(training_path):
            os.makedirs(training_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        

        for epoch in range(self.args.train_epochs):
            iter_count = 0            
            total_train_loss = 0
            m_train = 0
        
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder
                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim].to(self.device)

                # backward the MSE loss
                train_loss_mse = nn.MSELoss()(outputs, batch_y)
                total_train_loss += train_loss_mse.item() * batch_x.size(0)
                m_train += batch_x.size(0)
                train_loss_mse.backward()
                # update the model
                model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            
            mse_train = total_train_loss / m_train
            mse_vali = self.vali(vali_data, vali_loader, criterion)
            mse_test = self.vali(test_data, test_loader, criterion)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print("Epoch: {0}, Steps: {1} | Train MSE: {2:.4f} Vali MSE: {3:.4f} Test MSE: {4:.4f}".format(
                epoch + 1, train_steps, mse_train, mse_vali, mse_test))
            
            f = open("traininig_progress.txt", 'a')
            f.write(setting + "  \n")
            f.write("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            f.write("Epoch: {0}, Steps: {1} | Train MSE: {2:.4f} Vali MSE: {3:.4f} Test MSE: {4:.4f}".format(
                     epoch + 1, train_steps, mse_train, mse_vali, mse_test))
            f.write('\n')
            f.write('\n')
            f.close()
            
        
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
        
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            
        preds = []
        trues = []
        total_test_loss = 0
        m_test = 0
        
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder
                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
               
                preds_np = outputs.cpu().detach().numpy()
                trues_np = batch_y.cpu().detach().numpy()
                                
                # mse test
                test_loss = nn.MSELoss()(outputs, batch_y)
                total_test_loss += test_loss.item() * batch_x.size(0)
                m_test += batch_x.size(0)
                
                preds.append(preds_np)
                trues.append(trues_np)                    

        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((preds - trues) / trues))
        
        print('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae, rmse, mape))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae, rmse, mape))
        f.write('\n')
        f.write('\n')
        f.close()
        
        ############### Rescale to get the prediction in original shape ###################
        mean_X, std_X = self.Data.scaler.mean_, self.Data.scaler.scale_
        preds = preds * std_X[-1] + mean_X[-1]
        trues = trues * std_X[-1] + mean_X[-1]
        ############### Rescale to get the prediction in original shape ###################

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder
                outputs = self.model(batch_x)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
