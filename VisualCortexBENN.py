#   Title:  VisualCortexBENN2
#   Desc:   A program that allows for better image processing
#           utilizing a more incentive reward system.
#   Author: Angela Trainor
#   Date:   10/24/2025

import torch
import torch.nn as nn
import torch.nn.functional as F

#start of class
class VisualCortexBENN2(nn.Module):
    def __init__(self):
        super().__init__()

        # Low-Level Feature Extraction (V1–V3 simulation)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))  # Preserve spatial layout

        # Color & Texture Embedding
        self.color_fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

        self.texture_fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

        # Shape Emotion Mapping
        self.symmetry_fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid()
        )

        self.curvature_fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid()
        )

        self.complexity_fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid()
        )

        # Centroid Estimation (for spatial matching)
        self.centroid_fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 32),
            nn.Tanh()  # Normalized spatial coordinates
        )

        # Entity Classifier (optional expansion)
        self.entity_fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU()
        )

        # Attention Gating
        self.attention_fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        # Feature Extraction
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        pooled = self.pool(x)  # (B, 128, 8, 8)
        flat = pooled.view(pooled.size(0), -1)  # (B, 8192)

        # Emotional Embeddings
        color_embed = self.color_fc(flat)
        texture_embed = self.texture_fc(flat)

        symmetry = self.symmetry_fc(flat)
        curvature = self.curvature_fc(flat)
        complexity = self.complexity_fc(flat)
        shape_emotion = symmetry + curvature + complexity  # (B, 8)

        # Centroid for spatial matching
        centroid = self.centroid_fc(flat)  # (B, 32)

        # Entity Embedding
        entity_embed = self.entity_fc(flat)  # (B, 128)

        # Attention Score
        attention = self.attention_fc(flat)  # (B, 1)

        return {
            "color": color_embed,
            "texture": texture_embed,
            "shape_emotion": shape_emotion,
            "centroid": centroid,
            "entities": entity_embed,
            "attention": attention
        }


