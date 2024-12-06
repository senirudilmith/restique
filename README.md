# EEG-Based Sleep Stage Classification Neural Network

This repository presents the final implementation of a Convolutional Neural Network (CNN) designed for accurate sleep stage classification using EEG signals from the F3 and F4 channels. The model employs advanced deep learning techniques to optimize wake-up times and reduce sleep inertia by identifying sleep stages in real-time.

## Features and Architecture

- **Dual-Channel Input**: Processes EEG signals from the F3 and F4 channels through parallel convolutional branches to enhance spatial feature extraction.
- **Architecture Highlights**:
  - **Convolutional Layers**: Sequential Conv1D layers (64 to 1024 filters) with Batch Normalization and LeakyReLU activations for hierarchical feature learning.
  - **Global Max Pooling**: Retains the most prominent activations for each feature map, reducing dimensionality.
  - **Dense Layers**: Fully connected layers with 768 and 384 units, Batch Normalization, LeakyReLU activations, and Dropout (0.4) for improved generalization.
  - **Softmax Output Layer**: Generates a probability distribution over five sleep stages (Wake, NREM1, NREM2, NREM3, REM).
  
![Neural Network Architecture](https://github.com/user-attachments/assets/c3cee55b-d128-473d-aba2-aac7950856c3)

*Figure 1: Neural network architecture for EEG-based sleep stage classification. This diagram illustrates the dual-channel input, convolutional layers, and output classifications.*

## Results and Performance

- **Accuracy**:
  - Achieved **92.4% for Wake**, **95.2% for NREM3**, and **89% for REM** classification, demonstrating strong performance across key sleep stages.
  - An average F1-score of over **0.90**, reflecting balanced precision and recall across all classes.
- **Model Robustness**: The confusion matrix reveals minimal misclassifications, particularly in adjacent sleep stages, underscoring the model’s reliability.

![cm](https://github.com/user-attachments/assets/2e731bd4-4d98-4ba5-ae0e-08fd3fce184b)
<br>
*Figure 2: Normalized confusion matrix for sleep stage classification. This matrix highlights classification accuracy across five sleep stages: Wake (W), NREM1 (N1), NREM2 (N2), NREM3 (N3), and REM (R). High accuracy is observed in N3 (95.2%), W (92.4%), and REM (89.1%), with minimal misclassification between adjacent stages.*
<br>
<br>

![Balanced and Normalized confusion matrix](https://github.com/user-attachments/assets/fce49a99-9d8f-49e8-8087-83a0a0740f3c)
<br>
*Figure 3: Normalized confusion matrix for sleep stage classification. This matrix demonstrates the classification accuracy for NREM1 (N1), NREM2 (N2), NREM3 (N3), REM (R), and Wake (W) stages. High precision is noted in N3 (95%), N2 (94%), and Wake (93%), with minimal misclassification across most stages.*



## Applications

- **Sleep Optimization**: Enables smart alarms to wake users during light sleep stages, minimizing sleep inertia.
- **Clinical Utility**: Supports clinicians in diagnosing sleep disorders through precise stage classification.
- **Consumer Integration**: Adaptable for wearable devices to provide personalized sleep insights and data visualization.

## Dataset and Training

- **Data Source**: Based on the Human Sleep Project dataset, annotated by certified polysomnography technicians.
- **Training Pipeline**:
  - Dataset split: 80% for training, 20% for testing.
  - Data preprocessing: Feature normalization and one-hot encoding for robust model performance.

## License

This project is released under the **Creative Commons License**, allowing collaborative use and modification while maintaining attribution.

---

For more information, explore the repository documentation or contribute enhancements to further the impact of this work. Feedback and contributions are welcomed to expand the model's capabilities.
