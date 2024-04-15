import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, precision_score
from sklearn.metrics import r2_score, mean_absolute_error

import test_data
import model

# subdirs = ['CN', 'FS', 'LC']
subdirs = ['LC']
lesion = {
    'LC': 'Lacquer_Cracks',
    'CN': 'Choroidal_Neovascularization',
    'FS': 'Fuchs_Spot'
}

class Test(object):
    def __init__(self):
        self.device = torch.device("cpu")
        self.seed = 42

        self.subdirs = subdirs
        self.testData = {}
        self.testLoader = {}
        for subdir in self.subdirs:
            self.testData[subdir] = test_data.Data(
                baseDataPathList=['../Data1', '../Data2'],
                datasetName=subdir,
            )
            self.testLoader[subdir] = DataLoader(
                self.testData[subdir],
                batch_size=1,
                shuffle=False,
                num_workers=8,
            )

            print(f"Test data: {str(self.testData[subdir])} --- size {len(self.testData[subdir])}")

        self.model = model.model()
        self.model.load('./')

    def test(self, subdir):
        dice = []
        recall = []
        precision = []
        with tqdm(
            self.testLoader[subdir], desc=f"Testing {subdir}", 
            unit='image'
        ) as pbar:
            with torch.no_grad():
                for x, y, fileID in pbar:
                    x = x.squeeze(0).numpy()
                    y = np.array(y.squeeze(0))
                    assert x.shape == (800, 800, 3), y.shape == (800, 800, 1)
                    y_pred = self.model.predict(x, lesion[subdir], None)
                    y_pred = np.array(y_pred > 128, dtype=np.uint8)
                    metrics = segmentation_metrics(y, y_pred)
                    dice.append(metrics['dice'])
                    recall.append(metrics['recall'])
                    precision.append(metrics['precision'])

                    pbar.set_postfix_str(
                        f"Dice     : {np.mean(dice):.4f}, " + \
                        f"Recall   : {np.mean(recall):.4f}, " + \
                        f"Precision: {np.mean(precision):.4f}"
                    )

        return {
            'dice': np.mean(dice),
            'recall': np.mean(recall),
            'precision': np.mean(precision),
        }

def segmentation_metrics(gt, pred, classId=1):
    gt, pred = gt.flatten(), pred.flatten()
    intersection = np.logical_and(gt == classId, pred == classId)
    dice = (2. * intersection.sum()) / (gt.sum() + pred.sum())
    recall = recall_score(gt, pred, labels=[1], zero_division=0)
    precision = precision_score(gt, pred, labels=[1], zero_division=0)
    return dict(dice=dice, recall=recall, precision=precision)

def main():
    tester = Test()
    results = {}
    for subdir in tester.subdirs:
        results[subdir] = tester.test(subdir)

    dice, recall, precision = 0.0, 0.0, 0.0
    for subdir in tester.subdirs:
        result = results[subdir]
        dice += result['dice']
        recall += result['recall']
        precision += result['precision']
        print(f"[{subdir}]:")
        print(f"\tDice: {result['dice']}")
        print(f"\tRecall: {result['recall']}")
        print(f"\tPrecision: {result['precision']}")
    dice /= len(tester.subdirs)
    recall /= len(tester.subdirs)
    precision /= len(tester.subdirs)

    print(f"Average:")
    print(f"\tDice     : {dice}")
    print(f"\tRecall   : {recall}")
    print(f"\tPrecision: {precision}")

if __name__ == '__main__':
    main()