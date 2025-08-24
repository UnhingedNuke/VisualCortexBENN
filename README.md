# VisualCortexBENN
The Visual Cortex BENN(Biologically-inspired Emotional Neural Network) is a machine learning model designed to identify, discern, and recreate different images at a rate faster than other learning models by utilizing an emotionally based learning algorithm that closely mimicks the actual neuron firing pattern to, within, and from the visual cortex within the human brain. 

Please respect the license and cite this source if you use it. By citing my name, you're helping me to have a future pursing my passion, preventing my work from being stolen in the future, and giving me the recognition to keep helping our society. 

Thank you.  - Angela Louise Trainor

![Python CI](https://github.com/<UnhingedNuke>/
VisualCortexBENN/actions/workflows/ci.yml/badge.svg)
**VisualCortexBENN** is a neural network inspired by the
human visual cortex.
It processes images by extracting low-level features
(edges, textures, colors),
mapping shape emotions, estimating centroids, and producing
embeddings that can
be used for recognition, attention scoring, or higher-level
AI reasoning.
---
## Features
- Convolutional Feature Extraction (V1–V3 simulation)
- Color & Texture Embeddings
- Shape Emotion Mapping (symmetry, curvature, complexity)
- Centroid Estimation for spatial reasoning
- Entity Embeddings for classification or retrieval
- Attention Gating for relevance filtering
- Fully trainable with a generic Trainer class
- Supports validation and accuracy tracking
---
## Requirements - Python 3.8+
- PyTorch >= 2.0
- TorchVision
Install dependencies:
```bash
pip install torch torchvision
  🎨 🧠 📦✨
     Getting Started
Follow these steps to quickly run and train
VisualCortexBENN from GitHub:
1. Clone the Repository
git clone https://github.com/<UnhingedNuke>/
VisualCortexBENN.git
cd VisualCortexBENN
2. Install Dependencies
pip install -r requirements.txt
3. Run Built-in Tests
The model includes unit tests or fallback testing:
python visual_cortex_ben.py
Verifies forward pass, output shapes, and gradients.
4. Prepare Training Data
Use a built-in dataset or your own:
from torchvision import datasets, transforms
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])
train_dataset = datasets.CIFAR10(root="./data", train=True,
download=True, transform=transform)

 val_dataset = datasets.CIFAR10(root="./data", train=False,
download=True, transform=transform)
For custom datasets:
from torchvision.datasets import ImageFolder
custom_dataset = ImageFolder(root="./my_images",
                             transform=transforms.Compose([
transforms.Resize((64,64)),
                             ]))
5. Train the Model
transforms.ToTensor()
from visual_cortex_ben import VisualCortexBENN, Trainer
model = VisualCortexBENN()
trainer = Trainer(model, classifier_out=10)
trainer.train(train_dataset, val_dataset=val_dataset,
epochs=3, batch_size=32)
   Concept Reference
VisualCortexBENN is inspired by the human visual cortex,
with layered processing similar to V1–V3 areas.
For detailed explanation, see the PDF:
[Visual Cortex-inspired Neural Network Design](./docs/
VisualCortexBENN_Concept.pdf)
   Project Structure
  
 VisualCortexBENN/
├── README.md
├── visual_cortex_ben.py ├── requirements.txt ├── docs/
│ ├── VisualCortexBENN_Concept.pdf
│ └── workflow.png ├── data/
├── notebooks/
├── tests/
└── examples/
Next Steps
• Swapdatasetsorincreaseimagesize
• Fine-tuneattentionmodule
• Extendshapeemotionembeddings
• Monitortrainingwithvalidationmetrics
• PublishworkviaGitHub+preprintserver(arXiv)
   Workflow Diagram
Below is a visual schematic of how VisualCortexBENN
processes data:
  
• Input→VisualCortexBENN→Embeddings→Classifier→ Predictions / Metrics
• Thishelpsusersunderstandthefullpipelineata glance.
Using GitHub CI
This repository includes a GitHub Actions workflow that
automatically runs tests when you: • Pushcodetothemainbranch
• Openapullrequest
Badge above shows the current build status.

## License

This project is licensed under the terms of the GPL 3.0 LICENSE.

Author: Angela Trainor
Date: 08/22/2025

