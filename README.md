# This project applice transfer learning with a pretrained ResNet18 on the  The Oxford-IIIT Pet Dataset

    Dataset used: Oxford-IIIT Pet Dataset
    Model: ResNet18 (pretrained)
    Framework: PyTorch
    Technique: Transfer Learning / Linear Probing


It explore both:
    
    1. Binary classification (Cat vs Dog)
    2. Multi-classification (37 breeds using linear probing)
    3. Effect of transfer learning strategies on performance


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


3. Experiments (Project Goals)

    Linear Probing:

        Freeze all convolutional layers
        Train only final layer
        Evaluate pretrained feature quality

    Fine-tuning:

        Unfreeze last l layers
            Compare:

                l = 1
                l = 2
                l = 3
                l = 4
        
    Fine-tuning with limited data:

        Train with:

            100% data
            10% data
            1% data

    Fine-tuning with imbalanced classes:

        Reduce samples per class  20% of the training images for each cat breed
        Evaluate:

            Per-class accuracy
            F1-score

        Try:

            Imbalanced
            weighted cross-entropy
            Oversampling