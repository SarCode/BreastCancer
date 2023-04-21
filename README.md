<p align="left"> <img src="https://komarev.com/ghpvc/?username=sarcode&label=Profile%20views&color=0e75b6&style=flat" alt="sarcode" /> </p>

[![LinkedIn][linkedin-shield]][linkedin-url]


# BreastCancer
The objective of these predictions is to assign patients to either a benign group that is noncancerous or a malignant group that is cancerous.
BreastCancer.py & Keras.ipynb [ Deep Learning ]

# Dataset
This breast cancer database was obtained from Dr. Wolberg’s office at the University of Wisconsin
Hospitals, Madison. Each record here contains values for different morphological and pathological
features of a tumor dissected from any given patient. The class column indicates whether the patient
has been characterized as the benign tumor or a malignant tumor.

Contains 700*11 rows and columns respectively  

Cancer.csv - should be in same directory as BreastCancer.py

# Resources
Spyder  
Pandas (library)  
Keras  (library)


# Classification Methods
Random Forrest  (RF)  
Support Vector Machine (SVM) 

# Confusion Matrix  
[ True Positive   False Positive
  False Negative  True Negative]  
  
  True Positive  = Answer (Benign)    - Predicted (Benign)  
  False Positive = Answer (Malignant) - Predicted (Benign)
  False Negative = Answer (Benign)    - Predicted (Malignant)
  True Negative  = Answer (Malignant) - Predicted (Malignant)
  
  # Support Vector Machine (SVM)
  True Positive  = 82   
  False Positive = 3  
  False Negative = 1  
  True Negative  = 54  
  
  SVM False Positive count i.e. predicted as having benign tumor but actually have malignant tumor = 3  
  
  # Random Forrest (RF)
   True Positive  = 83   
  False Positive = 2    
  False Negative = 2    
  True Negative  = 53  
  RF False Positive count i.e. predicted as having benign tumor but actually have malignant tumor = 2    
  
  # Accuracy
  SVM = 97.14 %
  Random Forrest = 97.14 %
 
 For having less number of false positive we should use random forrest i.e. 2 
 
 # Deep Learning Model
 Keras.ipynb is application of deep learning model on Breast Cancer dataset
  
  
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/sarthak-agarwal-dell/

