import timm
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, config = "vit_base_patch16_224", num_classes = 6):
        super().__init__()
        self.vit = timm.create_model(config, pretrained = True)
        in_features = self.vit.head.in_features
        self.vit.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

if __name__ == "__main__":
    model = ViT()
    print(model)
