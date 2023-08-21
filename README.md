# AI-brain-tumor-classification
# BraTum Scans -- Brain Tumor Classification using MobileNet-v2
## 1 SCOPE
Brain Tumor is one of the deadly diseases prevailing in today's world. It's timely detection is very important to start the treatment process immediately.Brain tumor classification using machine learning is a significant and rapidly evolving field with broad implications for medical diagnosis and treatment planning. As medical imaging technologies continue to advance, the amount of data generated from brain scans has grown exponentially, creating both opportunities and challenges. Machine learning techniques offer a powerful means to process and interpret this data, assisting medical professionals in accurately identifying and classifying different types of brain tumors. The scope of this brain tumor classification project is to classify the brain mri images into four classes yhat are pituitary tumor, meningioma tumor, no tumor or pituitary tumor. MobileNet V2 model is used in this project along with application of explainabled AI at the end to evaluate the model predictions more accurately. The ultimate goal of this project  is to provide clinicians with efficient and reliable tools to aid in early detection, precise diagnosis, and personalized treatment recommendations. With the potential to expedite diagnoses, minimize subjectivity, and improve patient outcomes, brain tumor classification using machine learning holds immense promise in the realm of medical image analysis and healthcare innovation.
## 2 INTRODUCTION:
Brain tumor classification project involves the classification of brain mri images into further classes on basis of feature learning using AI model.Road map of this project includes cpollection of suitable data, selection of applropriate AI for the dataset, training of model and testing it afterwards. All the details of model are provided in sections below.
## 3 PROCESS OF Brain Tumor Classification
Steps and challenges faced during the project include collection of suitable dataset, model selection of basis of it's architecture suitablility with the dataset, preprocessing of data that includes division and augmentation of data , model training, predictions and finally application of exlainable AI to get better understanding of model's working. 
<br />

![flowchart1](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/f126c22b-835f-43e7-be8c-4486ee5d3f96)
### 3.1 DATASET
In my Brain Tumor Classification project, I embarked on a comprehensive journey to construct an advanced neural network model that can effectively distinguish between different types of brain tumors. To begin, I accessed a high-quality dataset from Kaggle, containing a diverse range of brain images representing various tumor categories. Original dataset consist of two divisions testing and training sets. Each testing and training set further contain four classes  glioma_tumor, meningioma_tumor, no_tumor and pituitary_tumor. For my project I divided the data into three sets training, testing and validation in 70, 10 and 20 percent proportion respectively. 
<br />
![dataset](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/c8732fdf-a599-438c-b0b7-008727a313e6)
### 3.2 DATA PREPROCESSING:
Data preprocessing involves the steps the dataset particularly the training set undergoes before starting the training of model. Preprocessing steps done in brain tumor classification project are mentioned below;
#### 3.2.1 COMBINED AND SEGREGATED DATASETS
As in dataset brain mri scans are taken from different angles like top angle images, left angle images and back angle images. For better understanding of images classification I segregated those images for each class and created sub classes like glioma_tumor_top_Angle, mengiona_tumor_Left_Angle and so on. I applied model once on combined dataset and once on each angle dataset. 
#### 3.2.2  DATA AUGMENTATION
For segregated dataset I used data augmentations. As now I divided the already existing classes into different sub classes I faced issue of less data. To enhance data I used data augmentations. To enhance the robustness of the model, I applied data preprocessing techniques such as data augmentation like CLahe, Random Brightness, Random BrightnessandContrast.
##### 3.2.2.1 CLAHE
CLAHE stands for "Contrast Limited Adaptive Histogram Equalization." It is a computer image processing technique used to enhance the contrast and visibility of details in an image. Unlike standard histogram equalization, which spreads out the pixel intensity values across the entire range, CLAHE divides the image into smaller regions and applies histogram equalization separately to each region. This adaptive approach prevents the over-amplification of noise in areas with low contrast, leading to improved image quality. CLAHE is particularly useful for enhancing details in images with uneven lighting conditions or those containing both dark and bright regions. It finds applications in medical imaging, remote sensing, and other fields where enhancing image contrast is crucial for analysis and visualization.
![aug1](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/8eb1081f-6fb6-4da3-a04e-b3fa38437819)
<br />
##### 3.2.2.2 RANDOM BRIGHTNESS
Random brightness augmentation is a data augmentation technique commonly used in image processing and computer vision tasks. It involves adjusting the brightness of an image randomly to create variations of the original image. This technique is often applied to augment training datasets for machine learning models, helping the models become more robust and generalizable by exposing them to a wider range of lighting conditions.
![aug2](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/fd2c7360-db0e-42ed-9f14-332d53230f18)
<br />
##### 3.2.2.3 RANDOM BRIGHTNESS AND CONTRAST
Random brightness and contrast augmentation are two distinct data augmentation techniques commonly employed in image processing and computer vision tasks to enhance the diversity and quality of training data for machine learning models.By applying both random brightness and contrast augmentation, machine learning models become more resilient to changes in lighting conditions and are better equipped to recognize important features regardless of how the image is captured. These techniques contribute to reducing overfitting, enhancing generalization, and improving the model's performance on real-world data. When combined with other augmentation techniques such as rotation, scaling, and flipping, random brightness and contrast augmentation form a comprehensive strategy for creating a diverse and representative training dataset that helps build more robust and accurate models.
![aug3](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/32d8e471-31d2-405c-9806-de78f69d750a)
<br />
### 3.3 MODEL ARCHITECTURE:
MobileNet-V2 has been selected for brain tumor classifciation project because of it's efficiency and less computer sources utility.Mobilenet-V2 model has been used in this project of brain tumor classification. MobileNet v2 is a lightweight convolutional neural network (CNN) architecture that is designed for efficient computation on mobile and embedded devices. It uses depth-wise separable convolutions and residual connections to reduce the number of parameters and improve the performance of the network. MobileNet is used for feature extraction, which means it is responsible for identifying key features in the input image that can be used for object detection.
![model](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/950254ab-4b65-460e-9b22-9afa56f47d5c)
<br />
### 3.4 TRAINING LOOP:
After data augmentation training of model started. This project include four datasets which are combined dataset, top angle data set, left angle data set and back angle dataset which further have their own training, validation and testing sets.. So training has to be done for each dataset's training set  individually. Models are trained on 30 epochs. After training the models comes the step of validation. models are being validated on validation set of respective dataset. After validation models are being tested.
### 3.5 TESTING
After training and validation models are being saved and then reloaded for proper testing. Results got after testing are mentioned in next section. test accuracy should be compared with validation accuracy to interpret the working of models.
#### 3.5.1 RESULTS
##### 3.5.1.1 MODEL ON COMBINED DATASET
I trained Mobilenet-v2 model on combined dataset that contain main classes glioma_tumor, meningioma_tumor, no_tumor and pituitary_tumor. I trained the model at 30 Epoch. After I validated the model on validation set. After this I tested the model on testing set to check accuracy of model. I got the following results;
Train Loss: 0.0781 | Train Acc: 0.9755
Valid Loss: 1.0393 | Valid Acc: 0.8451
Test Accuracy: 92.97%
Test Loss: 0.3987
![cm1](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/a786497e-8835-471a-9542-10e16ceb6d0d)
##### 3.5.1.2 SEGREGATED DATASETS
###### 3.5.1.2.1 TOP_ANGLE_DATASET
In this dataset images taken from top angle for each tumor are present into different sub classes like glioma_Tumor_Top_Angle , meningioma_tumor_Top_Angle and so on. It further consist of training , validation and testing set. I trained the model at 30 Epoch. After I validated the model on validation set. After this I tested the model on testing set to check accuracy of model. I got the following results;
Train Loss: 0.0103 | Train Acc: 0.9965
Valid Loss: 0.7859 | Valid Acc: 0.8511
Test Accuracy: 87.02%
Test Loss: 0.7121
![cm2](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/910b8003-9357-4998-83f4-210925b6ae9b)
<br />
###### 3.5.1.2.2 BACK ANGLE DATASET
In this dataset images taken from Back angle for each tumor are present into different sub classes like glioma_Tumor_Back_Angle , meningioma_tumor_Back_Angle and so on. It further consist of training , validation and testing set. I trained the model at 30 Epoch. After I validated the model on validation set. After this I tested the model on testing set to check accuracy of model. I got the following results;
Train Loss: 0.0031 | Train Acc: 1.0000
Valid Loss: 0.2240 | Valid Acc: 0.9427
Test Accuracy: 94.74%
Test Loss: 0.2796
![cm3](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/c70fba72-76c2-429d-9279-5928bfd9e926)
<br />
###### 3.5.1.2.3 LEFT ANGLE DATASET
In this dataset images taken from Left angle for each tumor are present into different sub classes like glioma_Tumor_Left_Angle , meningioma_tumor_Left_Angle and so on. It further consist of training , validation and testing set. I trained the model at 30 Epoch. After I validated the model on validation set. After this I tested the model on testing set to check accuracy of model. I got the following results;
Train Loss: 0.0005 | Train Acc: 1.0000
Valid Loss: 0.7610 | Valid Acc: 0.8827
Test Accuracy: 92.78%
Test Loss: 0.2037
![cm4](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/8ab0f234-c484-47a7-b69f-2577da30031e)
<br />
### 3.6 EXPLAINABLE AI
Explainable AI, often abbreviated as XAI, refers to the capability of artificial intelligence systems to provide understandable and transparent explanations for their decisions and actions. In complex machine learning models, like deep neural networks, the decision-making process can be challenging to interpret, leading to concerns about trust, accountability, and bias. Explainable AI aims to bridge this gap by enabling users, including both developers and end-users, to comprehend why a particular AI model arrived at a specific outcome. By offering insights into the factors and features that influenced a decision, explainable AI enhances transparency, facilitates model debugging and validation, and ensures that AI applications are used responsibly and ethically across various domains, including healthcare, finance, and autonomous systems. I applied explainable AI o my model to get the better understanding of how model is learning the features.
some results are as fellow;
#### 3.6.1 EXPLAINABLE AI ON COMBINED DATASET:
According to confusion matrix  of combined dataset mentioned above  there are five false negative in this model testing process all these belongs to meningioma_tumor class which model has predicted as no_tumor. After applying explainable AI I got the following heat maps;
![e1](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/497a3089-90dd-428e-aa6b-cc7d6b79a8ea)
Let’s compare them with the correctly predicted image;
![e2](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/f28fb1a8-525a-4236-9367-9dd648a35e52)
As model predicted some of meningioma_tumor images into other classes;
![e3](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/2a76091d-d9b9-4832-8ec8-be625cead6ca)
Some corrected predicted images by model are as follow;
![e4](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/94d2c5c1-13b6-4337-91c7-e01c18b482bf)
<br />
#### 3.6.2 EXPLAIANBLE AI ON SEGREGATED DATASET
##### 3.6.2.1 TOP-ANGLE DATASET
Model on testing classify few images as false negative  For example image  below belongs to meningioma_tumor which model classified as no_tumor.
![e5](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/54639fc3-d100-4be9-b7d6-507dbb647b61)
Some correctly predicted images by model are as follow;
![e6](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/fc64a0b8-a0a3-42be-a7ac-9ef8c1d6e858)
##### 3.6.2.2 BACK_ANGLE DATASET
This model does not classify any image as false negative. 
Some of the correctly predicted images by model are as follow;
![e7](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/d433ef22-ceb3-40f2-957f-5ffee60d1be5)
##### 3.6.2.3 LEFT_ANGLE DATASET
This model does not classify any image as false negative. 
Some of the correctly predicted images by model are as follow;
![e8](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/4b41f639-7f8e-4753-8419-9a5655769b88)
model also predicted some images into classes different than their actual class like;
![e9](https://github.com/ioptime-official/ai-brain-tumor-classification/assets/138657622/f1893b54-ce8c-46ef-b1c3-a8d9139d0864)
#### 3.7 POINTS ON EXPLAINABLE AI RESULTS 
 In above models mostly meningioma_tumor is classified as no_tumor (false negative). Models are getting confused in detecting meningioma_tumor correctly in some of images particularly top angle images. Correct classification depends on the area model is capturing for feature learning. In ideal situation, model should look to meninges, glioma and pituitary region of brain to classify these images. To some extend models learned accurate features according to my understanding. In case of left angle I think model is not learning the features of pituitary_left_angle accurately. Although it is classifying images into correct class but according to my perception model learned wrong features during training and classify test images accordingly.
 #### CONCLUSION
 In conclusion, our brain tumor classification project has successfully utilized a combination of cutting-edge techniques to achieve commendable results. By employing data augmentation, we expanded the diversity of our training dataset, enhancing the model's ability to generalize and accurately classify brain tumors across various scenarios. The incorporation of the MobileNetV2 model, known for its efficiency and effectiveness, proved pivotal in maintaining a balance between computational resources and accuracy. Furthermore, the integration of explainable AI methodologies empowered us to gain insights into the decision-making process of the model, ensuring transparency and accountability in its predictions. Appreciable accuracy has been achieved in the project. In explainable AI implementation, it came to know that for pituitary_Tumor_Left_Angle model has learned wrong feature although it is classifying images into right class but features learned by model are not features for tumor that may cause issue if we give it a unseen images. This part of project require some more investigation and working.

