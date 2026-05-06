import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalFusionNet(nn.Module):
    def __init__(self, num_tabular_features=5, num_classes=14):
        super(MultimodalFusionNet, self).__init__()
        
        # ==========================================
        # 1. Image Branch (CheXNet / DenseNet121)
        # ==========================================
        # We load the pretrained weights to give our model a massive head start
        self.image_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        
        # DenseNet121's default classifier is a single Linear layer: in_features=1024, out_features=1000
        # We replace it with an Identity layer to just pass the 1024-dim vector right through
        num_image_features = self.image_model.classifier.in_features
        self.image_model.classifier = nn.Identity() 
        
        # ==========================================
        # 2. Tabular Branch (Clinical MLP)
        # ==========================================
        self.tabular_model = nn.Sequential(
            nn.Linear(num_tabular_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout prevents the MLP from memorizing the data
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # ==========================================
        # 3. Fusion Head
        # ==========================================
        # Concatenate the 1024 image features with the 16 tabular features
        fusion_dim = num_image_features + 16
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
            # NOTE: We do NOT use a Sigmoid activation here. We will use BCEWithLogitsLoss 
            # in our training loop, which applies Sigmoid internally in a more mathematically stable way.
        )

    def forward(self, image, tabular):
        # Extract image features
        img_features = self.image_model(image) # Shape: [Batch, 1024]
        
        # Extract tabular features
        tab_features = self.tabular_model(tabular) # Shape: [Batch, 16]
        
        # Fuse (Concatenate along the feature dimension)
        fused_features = torch.cat((img_features, tab_features), dim=1) # Shape: [Batch, 1040]
        
        # Final classification
        output = self.classifier(fused_features) # Shape: [Batch, 14]
        return output
