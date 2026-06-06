import segmentation_models_pytorch as smp
from config import NUM_CLASSES

encoder_name = "efficientnet-b3"
encoder_weights = "imagenet"

model = smp.Unet(
    encoder_name=encoder_name,
    encoder_weights=encoder_weights,
    in_channels=3,
    classes=NUM_CLASSES
)

def model_info():
    print(f"Model: UNet with {encoder_name} encoder")
    print(f"Encoder weights: {encoder_weights}")
    print(f"Number of classes: 12")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
