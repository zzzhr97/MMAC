import os
import cv2
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu

class model:
    def __init__(self):
        self.model_CN = []
        self.model_FS = []
        self.model_LC = []
        self.preProcess = getPreProcess()

        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cpu")

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        # self.model_LC = deeplabv3_resnet50(num_classes=1)
        # checkpoint_path_LC = os.path.join(dir_path, self.checkpoint_LC)
        # self.model_LC.load_state_dict(torch.load(checkpoint_path_LC, map_location=self.device))
        # self.model_LC.to(self.device)
        # self.model_LC.eval()

        # self.model_CNV = deeplabv3_resnet50(num_classes=1)
        # checkpoint_path_CNV = os.path.join(dir_path, self.checkpoint_CNV)
        # self.model_CNV.load_state_dict(torch.load(checkpoint_path_CNV, map_location=self.device))
        # self.model_CNV.to(self.device)
        # self.model_CNV.eval()

        # self.model_FS = deeplabv3_resnet50(num_classes=1)
        # checkpoint_path_FS = os.path.join(dir_path, self.checkpoint_FS)
        # self.model_FS.load_state_dict(torch.load(checkpoint_path_FS, map_location=self.device))
        # self.model_FS.to(self.device)
        # self.model_FS.eval()
        subdirs = ['upload_model/CN', 'upload_model/FS', 'upload_model/LC']
        for subdir in subdirs:
            for pth_name in os.listdir(os.path.join(dir_path, subdir)):
                pth_path = os.path.join(dir_path, subdir, pth_name)
                model = torch.load(pth_path, map_location=self.device)
                model.eval()
                eval(f"self.model_{subdir}.append(model)")

    def predict(self, input_image, lesion_type, patient_info_dict):
        """
        perform the prediction given an image and the metadata.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :param lesion_type: a string indicates the lesion type of the input image: 'Lacquer_Cracks' or 'Choroidal_Neovascularization' or 'Fuchs_Spot'.
        :param patient_info_dict: a dictionary with the metadata for the given image,
        such as {'age': 52.0, 'sex': 'male', 'height': nan, 'weight': 71.3},
        where age, height and weight are of type float, while sex is of type str.
        :return: a ndarray indicates the predicted segmentation mask with the shape 800 x 800.
        The pixel value for the lesion area is 255, and the background pixel value is 0.
        """
        image = cv2.resize(input_image, (512, 512))
        image = self.preProcess(image=image)['image']
        image = torch.tensor(image).unsqueeze(0).to(self.device, torch.float)
        pred_mask = None
        with torch.no_grad():
            if lesion_type == 'Lacquer_Cracks':
                models = self.model_LC
            elif lesion_type == 'Choroidal_Neovascularization':
                models = self.model_CN
            elif lesion_type == 'Fuchs_Spot':
                models = self.model_FS
            else:
                assert False   

            pred_mask = torch.zeros((1, 1, 512, 512))
            cnt = len(models)
            for model in models:
                cur_pred_mask = model(image)
                if type(cur_pred_mask) == type(()):
                    cur_pred_mask = cur_pred_mask[0]
                pred_mask = torch.add(pred_mask, cur_pred_mask)
            pred_mask = pred_mask / cnt

        pred_mask = pred_mask.detach().squeeze().numpy()
        pred_mask = np.array(pred_mask > 0.5, dtype=np.uint8) * 255
        pred_mask = cv2.resize(pred_mask, (800, 800), interpolation=cv2.INTER_NEAREST)
        return pred_mask

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def getPreProcess():
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnext50_32x4d','imagenet')
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)