# IMAGE-CLASSIFICATION-MODEL

"COMPANY": CODTECH IT SOLUTIONS

"NAME": BHUMIREDDY CHARITHA REDDY

"INTERN ID": CT06DZ465

"DOMAIN": MACHINE LEARNING

"DURATION": 6 WEEKS

"MENTOR": Neela Santhosh

#DESCRIPTION

(A Convolutional Neural Network (CNN) is a deep learning model designed specifically for image data. CNNs automatically learn features like edges, textures, and shapes by applying convolutional filters, making them highly effective for image classification tasks.

In this project, we use TensorFlow (Keras) to build a CNN and classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 classes (such as airplane, car, dog, and ship).

📌 Implementation Steps
Data Loading & Preprocessing
The CIFAR-10 dataset is loaded directly from TensorFlow's datasets. Images are normalized by dividing pixel values by 255 to scale them between 0 and 1.

Model Architecture
A sequential CNN is built using:

Convolutional layers to extract features

MaxPooling layers to reduce spatial dimensions

Flatten layer to convert feature maps into vectors

Dense layers to perform classification

Example architecture:

Conv2D → ReLU → MaxPool

Conv2D → ReLU → MaxPool

Conv2D → Flatten → Dense → Output

Compilation & Training
The model is compiled using the Adam optimizer and Sparse Categorical Crossentropy as the loss function. It is trained for a number of epochs while tracking training and validation accuracy.

Evaluation & Visualization
The trained model is evaluated on the test set to measure accuracy. Additionally, training history (accuracy and loss over epochs) is plotted to analyze model performance.)




