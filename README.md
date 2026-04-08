# RD_Infro_Technology
Machine learning Internship projects

Multi-Label Emotion Classification using Deep Learning
📌 Overview

This project focuses on building a multi-label emotion classification model that predicts multiple emotions from short English text. The model identifies five emotions:

Anger
Fear
Joy
Sadness
Surprise

Each text can express more than one emotion, making this a multi-label classification problem.

🎯 Objective

The primary goal of this project is to develop a robust deep learning model capable of identifying multiple emotions from text. The project leverages a pre-trained transformer model, roberta-base, and applies advanced techniques such as class imbalance handling, threshold optimization, and efficient training strategies.

🧠 Model Architecture
Pre-trained RoBERTa model
Dropout layer (0.1)
Fully connected classification layer (5 outputs)
Sigmoid activation for multi-label prediction
⚙️ Key Techniques Used
✅ Multi-label classification using BCEWithLogitsLoss
✅ Class imbalance handling using pos_weight
✅ Learning rate scheduling with warm-up
✅ Gradient clipping for stability
✅ Early stopping to prevent overfitting
✅ Threshold optimization for improved F1-score
🔤 Data Preprocessing
Text cleaning (whitespace normalization)
Case preservation (RoBERTa is case-sensitive)
Train-validation split (90% / 10%)
Tokenization using RoBERTa tokenizer (BPE-based)
🔧 Tech Stack
Python
PyTorch
Hugging Face Transformers
Scikit-learn
NumPy, Pandas
📈 Evaluation Metrics
Macro F1-score
Precision & Recall
Threshold-based evaluation
🚀 Project Pipeline
1. Training Pipeline
Load and preprocess dataset
Tokenize text
Train model with validation monitoring
2. Validation
Evaluate using macro F1-score
Apply early stopping
3. Inference Pipeline
Load trained model
Generate predictions
Apply optimized thresholds
Create submission file

Project Structure
├── train.csv
├── test.csv
├── sample_submission.csv
├── app.py / main.py
├── model.pth
├── optimal_thresholds.npy
├── README.md
