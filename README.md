This project leverages deep learning and computer vision to classify lung cancer stages and detect diabetic retinopathy from clinical images using advanced neural networks and visualization tools. It utilizes pre-trained models like InceptionV3 and VGG16, integrated with TensorFlow/Keras, to build efficient CNN-based classifiers. The workflow includes image preprocessing with OpenCV, data augmentation using ImageDataGenerator, and interactive visualizations through Plotly and Matplotlib. The models are trained with techniques like early stopping, learning rate reduction, and class balancing to improve performance and reliability in real-world medical diagnosis.


Lung Cancer Classification
Uses CT scan images to classify lung conditions as Benign, Malignant, or Normal. Built with CNN architectures using InceptionV3 and VGG16.
-  Test Accuracy: 87.88%
-  Techniques: Transfer learning, Image augmentation, Early stopping, Class balancing.

Diabetic Retinopathy Detection
Analyzes retinal fundus images to classify whether a patient has Diabetic Retinopathy (DR) or No DR.
- Accuracy: 92.55%
- Built using TensorFlow/Keras, with ImageDataGenerator for augmentation, categorical accuracy for performance, and deployed-ready with TensorFlow Lite for mobile or embedded inference.
