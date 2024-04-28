from torch.utils.data import Dataset
import os
import cv2

taskPath = '2. Segmentation of Myopic Maculopathy Plus Lesions'
dataPathDict = {
    'LC': '1. Lacquer Cracks',
    'CN': '2. Choroidal Neovascularization',
    'FS': '3. Fuchs Spot'
}

class Data(Dataset):
    """
    Read data, transform images and augmentation.

    Input:
    - baseDataPath: base path to data (images, masks)
    - datasetName: dataset name (LC, CN, FS)
    - aug: augmentation function
    - preProcess: pre-processing function
    """
    def __init__(
        self,
        baseDataPathList,   # ['./Data1', './Data2']
        datasetName,        # 'LC'
        aug=None,
        preProcess=None,
    ):
        self.aug = aug
        self.preProcess = preProcess
        self.baseDataPathList = baseDataPathList
        self.datasetName = datasetName

        # Get data path, such as 
        # ['../Data1/2. Segmentation of Myopic Maculopathy Plus Lesions/1. Lacquer Cracks']
        dataPath = []
        if datasetName in dataPathDict:
            datasetDir = dataPathDict[datasetName]
            for baseDataPath in baseDataPathList:
                dataPath.append(os.path.join(baseDataPath, taskPath, datasetDir))
        else:
            raise ValueError("Invalid dataset name")

        # Get image and mask file path
        imagePathList = []
        maskPathList = []
        for path in dataPath:
            # innerPath: ['1. Training Set'] or ['2. Validation Set']
            innerPath = os.listdir(os.path.join(path, '1. Images'))
            imagePathList.append(os.path.join(path, '1. Images', innerPath[0]))
            maskPathList.append(os.path.join(path, '2. Groundtruths', innerPath[0]))

        # Get image and mask file names
        baseNameList = [os.listdir(imagePath) for imagePath in imagePathList]
        imageFileLists = [
            [os.path.join(imagePathList[i], baseName) for baseName in baseNameList[i]]
            for i in range(len(imagePathList))
        ]
        maskFileLists = [
            [os.path.join(maskPathList[i], baseName) for baseName in baseNameList[i]]
            for i in range(len(maskPathList))
        ]
        self.imageFileList = [item for sublist in imageFileLists for item in sublist]
        self.maskFileList = [item for sublist in maskFileLists for item in sublist]

    def __str__(self):
        str = f"Dataset {self.datasetName} in "
        for path in self.baseDataPathList:
            str += f"{path}, " if path != self.baseDataPathList[-1] else f"{path}"
        return str

    def __len__(self):
        return len(self.imageFileList)

    def __getitem__(self, idx):
        # (800, 800, 3)
        image = cv2.imread(self.imageFileList[idx], cv2.IMREAD_COLOR)
        # (800, 800)
        mask = cv2.imread(self.maskFileList[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask == 255).astype('float').reshape(mask.shape[0], mask.shape[1], 1)
        fileName = os.path.split(self.imageFileList[idx])[-1]
        
        # augmentation
        if self.aug:
            sample = self.aug(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # pre-processing
        if self.preProcess:
            sample = self.preProcess(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask, fileName
    
def test1():
    data = Data(
        ['./Data1', './Data2'],
        'LC',
    )
    os.makedirs("./show/image", exist_ok=True)
    os.makedirs("./show/mask", exist_ok=True)
    print(len(data))
    for sample in data:
        print(sample['fileName'], sample['image'].shape, sample['mask'].shape)
        cv2.imwrite("./show/image/"+sample['fileName'], sample['image'])
        cv2.imwrite("./show/mask/"+sample['fileName'], sample['mask'])

if __name__ == "__main__":

    test1()