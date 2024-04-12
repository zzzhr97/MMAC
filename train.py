import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Data
from process import trainAUG, validAUG, getPreProcess
from utils import setSeed, diceCoef, getModel, getLoss, getOptimizer, getLrScheduler, getTime
from param import getArgs

class Train(object):
    def __init__(self, args):
        self.args = args

        print(f"Set seed {args.seed}")
        setSeed(args.seed)

        print(f"Use device {args.device}")
        self.device = torch.device(args.device)

        os.makedirs(f"./model/{args.dataset}", exist_ok=True)
        os.makedirs(f"./result/{args.dataset}", exist_ok=True)

        self.prepareData()
        self.prepareModel()
        self.prepareResultFile()

    def prepareData(self):
        print(f"Load {self.args.dataset} data")
        self.preProcess = smp.encoders.get_preprocessing_fn(
            self.args.encoder, self.args.encoder_weights)
        self.trainData = Data(
            baseDataPathList=self.args.base_data_path,
            datasetName=self.args.dataset,
            aug=trainAUG(),
            preProcess=getPreProcess(self.preProcess),
        )
        self.validData = Data(
            baseDataPathList=self.args.base_data_path,
            datasetName=self.args.dataset,
            aug=validAUG(),
            preProcess=getPreProcess(self.preProcess),
        )
        print(f"Train data: {str(self.trainData)} --- size {len(self.trainData)}")
        print(f"Valid data: {str(self.validData)} --- size {len(self.validData)}")

        self.trainLoader = DataLoader(
            self.trainData,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=8,
        )
        self.validLoader = DataLoader(
            self.validData,
            batch_size=1,
            shuffle=False,
            num_workers=8,
        )

    def prepareModel(self):
        print(f"Use model {self.args.model}")
        self.model = getModel(self.args)
        self.model.to(self.device)

        self.preProcess = smp.encoders.get_preprocessing_fn(
            self.args.encoder, 
            self.args.encoder_weights
        )

        print("Loss function: ", self.args.loss_func)
        self.loss = getLoss(self.args.loss_func) 

        print("Optimizer: ", self.args.optimizer)
        self.optimizer = getOptimizer(
            self.args.optimizer,
            self.model.parameters(),
            lr=self.args.lr,
            wd=self.args.wd
        )

        print("LRScheduler: ", self.args.lr_scheduler)
        self.lrScheduler = getLrScheduler(
            self.args.lr_scheduler,
            self.optimizer,
        )

    def prepareResultFile(self):
        self.timeID = getTime()
        self.resultFile = open(
            f"./result/{self.args.dataset}/" + \
            f"{self.timeID}_{self.args.training_id}.txt",
            'w'
        )
        # parameter setting
        for key, value in sorted(vars(self.args).items(), key=lambda x: x[0]):
            self.write(f"{'[' + key + ']':<20} --- {value}\n")
        self.flush()

    def write(self, string, printstdout=False, flush=False):
        if printstdout:
            print(string, end='')
        self.resultFile.write(string)
        if flush:
            self.flush()
    
    def flush(self):
        self.resultFile.flush()

    def close(self):
        self.resultFile.close()

    def train(self):
        bestScore = -1
        bestEpoch = -1

        with torch.no_grad():
            x, y = [], []
            for i in range(self.args.batch_size):
                xx, yy, __ = self.trainData[i]
                x.append(xx)
                y.append(yy)
            x = torch.stack(x).to(self.device)
            y = torch.stack(y).to(self.device)
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            assert y_pred.shape == y.shape
            print("Input shape: ", x.shape)
            print("Output shape: ", y_pred.shape)
            print("Target shape: ", y.shape)

        for epochIdx in range(self.args.epoch):
            curEpoch = epochIdx + 1
            trainResult = self.trainOnce(curEpoch)
            validResult = self.validOnce(curEpoch)
            self.lrScheduler.step()

            self.write(f"[Epoch {curEpoch:03d}] ")
            self.write(f"TrainLoss - {trainResult['trainLoss']:.8f} ")
            self.write(f"ValidLoss - {validResult['validLoss']:.8f} ")
            self.write(f"ValidDice - {validResult['dice']:.8f}\n", flush=True)

            if self.args.save_each > 0:
                if curEpoch % self.args.save_each == 0:
                    torch.save(
                        self.model,
                        f"./model/{self.args.dataset}/" + \
                        f"{self.timeID}_{self.args.training_id}_{self.args.model}_epoch{curEpoch:03d}.pth"
                    )

            if validResult['dice'] > bestScore:

                self.write(
                    f"\t[A new best model at epoch {curEpoch:03d}]\n", 
                    flush=True
                )
                bestScore = validResult['dice']
                bestEpoch = curEpoch
                torch.save(
                    self.model,
                    f"./model/{self.args.dataset}/" + \
                    f"{self.timeID}_{self.args.training_id}_{self.args.model}_best.pth"
                )

        self.write(
            f"\nBest model: {bestEpoch:03d}\nDice: {bestScore:.08f}\n", 
            printstdout=True
        )
        self.close()

    def trainOnce(self, epoch):
        self.model.train()
        losses = []

        with tqdm(
            self.trainLoader, desc=f"Training epoch [{epoch:03d}]", 
            unit='batch'
        ) as pbar:
            for x, y, fileID in pbar:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                y_pred = self.model(x)
                assert y_pred.shape == y.shape

                loss = self.loss(y_pred, y)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                pbar.set_postfix_str(f"AvgLoss: {np.mean(losses):.08f}")

        return {
            'trainLoss': np.mean(losses),
        }

    def validOnce(self, epoch):
        self.model.eval()
        losses = []
        diceScores = []
        diceFunc = diceCoef

        with tqdm(
            self.trainLoader, desc=f"Validation epoch [{epoch:03d}]", 
            unit='batch'
        ) as pbar:
            with torch.no_grad():
                for x, y, fileID in pbar:
                    x, y = x.to(self.device), y.to(self.device)

                    y_pred = self.model(x)
                    assert y_pred.shape == y.shape

                    loss = self.loss(y_pred, y)
                    dice = diceFunc(y_pred, y)

                    losses.append(loss.item())
                    diceScores.append(dice.item())

                    pbar.set_postfix_str(
                        f"AvgLoss: {np.mean(losses):.08f}, " + \
                        f"AvgDice: {np.mean(diceScores):.08f}")

        return {
            'validLoss': np.mean(losses),
            'dice': np.mean(diceScores),
        }

def main():
    args = getArgs()
    trainer = Train(args)
    trainer.train()

if __name__ == '__main__':
    main()