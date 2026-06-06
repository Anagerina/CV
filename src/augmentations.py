import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_augmentation(img_size=(256, 256)):
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=5,
            p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1, 
            p=0.3),
        A.HorizontalFlip(p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        A.ElasticTransform(
            alpha=5,
            sigma=10,
            p=0.3,
            approximate=True),
        A.CoarseDropout(
            num_holes_range=(1,2),
            hole_height_range=(0.05, 0.1),
            hole_width_range=(0.05, 0.1),
            p=0.2)
    ])

def get_val_augmentation(img_size=(256, 256)):
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225])
    ])

