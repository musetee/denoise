import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# ==============================
# 1. Vision Transformer (ViT) for Projection Domain Denoising
# ==============================
from timm.models.vision_transformer import VisionTransformer

class ProjectionDenoiser(nn.Module):
    def __init__(self, img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=8):
        super(ProjectionDenoiser, self).__init__()
        self.vit = VisionTransformer(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, num_classes=embed_dim
        )
        self.fc = nn.Linear(embed_dim, img_size * img_size)
    
    def forward(self, x):
        x = self.vit(x)
        x = self.fc(x).reshape(x.shape[0], 1, 512, 512)  # Assuming single-channel projection data
        return x

# ==============================
# 2. Transformer-Based CT Reconstruction (Projection -> CT Image)
# ==============================
class ReconstructionTransformer(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=8):
        super(ReconstructionTransformer, self).__init__()
        self.encoder = VisionTransformer(
            img_size=512, patch_size=16, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, num_classes=embed_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512*512),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.reshape(x.shape[0], 1, 512, 512)  # Output CT Image

# ==============================
# 3. Image Domain Enhancement (ViT for CT Denoising)
# ==============================
class CTImageEnhancer(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=8):
        super(CTImageEnhancer, self).__init__()
        self.vit = VisionTransformer(
            img_size=512, patch_size=16, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, num_classes=embed_dim
        )
        self.fc = nn.Linear(embed_dim, 512 * 512)
    
    def forward(self, x):
        x = self.vit(x)
        x = self.fc(x).reshape(x.shape[0], 1, 512, 512)
        return x

# ==============================
# 4. Full Model Pipeline: Projection Domain + Reconstruction + Image Enhancement
# ==============================
class FullCTReconstruction(nn.Module):
    def __init__(self):
        super(FullCTReconstruction, self).__init__()
        self.projection_denoiser = ProjectionDenoiser()
        self.reconstruction_transformer = ReconstructionTransformer()
        self.ct_image_enhancer = CTImageEnhancer()
    
    def forward(self, projection):
        denoised_projection = self.projection_denoiser(projection)
        reconstructed_ct = self.reconstruction_transformer(denoised_projection)
        final_ct = self.ct_image_enhancer(reconstructed_ct)
        return final_ct

# ==============================
# 5. Training Setup
# ==============================
# Define dataset and dataloader
class CTDataset(Dataset):
    def __init__(self, projections, ct_images):
        self.projections = projections
        self.ct_images = ct_images
    
    def __len__(self):
        return len(self.projections)
    
    def __getitem__(self, idx):
        return torch.tensor(self.projections[idx]).float().unsqueeze(0), torch.tensor(self.ct_images[idx]).float().unsqueeze(0)

# Define training loop
def train_model(model, dataloader, epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        for projection, ct_image in dataloader:
            projection, ct_image = projection.to(device), ct_image.to(device)
            optimizer.zero_grad()
            output = model(projection)
            loss = loss_fn(output, ct_image)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# Instantiate and train
model = FullCTReconstruction()
dataloader = DataLoader(CTDataset(projections=[], ct_images=[]), batch_size=2, shuffle=True)  # Replace with real data
train_model(model, dataloader)
