<div align="center">
<img src="https://github.com/mat-mcc/Lenet-5-Digit-Classifier-Implementation/blob/main/LenetDemo.gif" width="700" height="500"/>
</div>

# Lenet-5 Digit Classifier Implementation with Interactive Visualization

A Python implementation of the classic **LeNet-5 convolutional neural network** for handwritten digit recognition, extended with modern improvements, interactive drawing and classification, and live CNN visualization. Users can draw digits directly in a GUI and see the model’s predictions, probability distributions, and intermediate feature maps. 

---

## Features

- **Original and Modified LeNet-5:**  
  - Classic LeNet-5 with Tanh activations and average pooling.  
  - Modified version with ReLU, dropout, and MaxPooling for improved performance.  

- **Interactive Drawing UI:**  
  - Draw digits with the mouse in a dedicated canvas.  

- **Live Prediction Visualization:**  
  - Shows **predicted digit** and **probabilities for all classes (0–9)** in real-time.  
  - Embedded **Conv1 feature maps** display what the CNN “sees” in early layers.  

- **Training & Testing:**  
  - Trains on **MNIST dataset**, with model checkpoints saved automatically.  
  - Displays **test accuracy** and **confusion matrix** for training.  

---

## Dependencies
- torch
- torchvision
- numpy
- matplotlib
- opencv-python
- Pillow
- scikit-learn

## Project Structure

```text
LeNet-Project/
│
├─ draw_digit_ui.py          # Interactive drawing GUI with live predictions
├─ lenet_models.py           # LeNet-5 and Modified LeNet-5 architectures
├─ main.py                   # Train, test, and launch GUI
├─ saved_models/             # Stores trained model weights
└─ README.md                 # Project documentation
