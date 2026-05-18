# This project applice transfer learning with a pretrained ResNet18 on the  The Oxford-IIIT Pet Dataset

    Dataset used: Oxford-IIIT Pet Dataset
    Model: ResNet18 (pretrained)
    Framework: PyTorch
    Technique: Transfer Learning / Linear Probing


It explore both:
    
    1. Binary classification (Cat vs Dog)
    2. Multi-classification (37 breeds using linear probing)



1. BinaryClass.py
This script solves the binary classification task:

    Classifies images into Cat (0) or Dog (1)
    Uses a pretrained ResNet18 model
    Freezes convolutional layers and trains only the final classification layer

Run: 

    python BinaryClass.py


2. linearprobing.py 
This script solves the multi-class classification task:

    Classifies pet images into 37 different breeds
    Uses ResNet18 pretrained on ImageNet
    Implements linear probing by freezing the backbone and training only the final fully connected layer

Run:

    python linearprobing.py

