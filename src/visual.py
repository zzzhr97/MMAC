
import albumentations as albu
import cv2
import os

def visual_image(image_path, mask_path, aug_func, save_image_path, save_mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask == 255).astype('float').reshape(mask.shape[0], mask.shape[1], 1)
    if aug_func is not None:
        sample = aug_func(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']
    mask = (mask * 255).astype('uint8')

    cv2.imwrite(save_image_path, image)
    cv2.imwrite(save_mask_path, mask)

def show_aug(aug, aug_name):
    os.makedirs('show/image', exist_ok=True)
    os.makedirs('show/mask', exist_ok=True)
    image_path = "./Data1/2. Segmentation of Myopic Maculopathy Plus Lesions/1. Lacquer Cracks/1. Images/1. Training Set/mmac_task_2_train_LC_0001.png"
    mask_path = "./Data1/2. Segmentation of Myopic Maculopathy Plus Lesions/1. Lacquer Cracks/2. Groundtruths/1. Training Set/mmac_task_2_train_LC_0001.png"
    save_image_path = f"./show/image/mmac_task_1_train_0001_{aug_name}.png"
    save_mask_path = f"./show/mask/mmac_task_1_train_0001_{aug_name}.png"
    visual_image(image_path, mask_path, aug, save_image_path, save_mask_path)

if __name__ == '__main__':
    show_aug(
        albu.Compose([
            albu.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
        ]),
        aug_name="RandomBrightnessContrast",
    )
