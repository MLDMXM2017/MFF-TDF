# Introduction
This repository contains the dataset and train code utilized in our research on developing **MFF-TDF**, a Tongue Diagnosis Framework with Multi-source Feature Fusion for the Fatty Liver Diagnosis (FLD). This framework employs the supervised learning for binary classification, specifically predicting whether a participant has an FLD, based on his tongue image and physiological indicators.
# Paper Title
Automated Tongue Diagnosis for Fatty Liver Disease Using Multi-Source Feature Fusion Framework
# Released Dataset
We constructed and publicly released a medium-sized standardized tongue image dataset annotated for FLD. The dataset included the participants' tongue images, basic physiological indicators, and FLD labels. The participants were recruited from a Prospective Cohort of Cancer and Chronic Disease Residents in Fuqing City, which was the first large dynamic prospective cohort study in Fujian Province, China. This cohort study was approved by the Ethics Review Committee of Fujian Medical University (approval numbers [2017-07] and [2020-58]), and written informed consent was obtained from all participants.
The inclusion criteria were as follows: (1) permanent residency in Fuqing City, Fujian Province; (2) age between 35 and 75 years; and (3) completion of both epidemiological data collection and tongue image acquisition between January and December 2021. The exclusion criteria were as follows: (1) incomplete tongue image collection, (2) incomplete epidemiological data, and (3) tongue images classified as noise samples owing to incorrect tongue protrusion posture or motion blur.
We utilized a "TCM Diagnostic Device" equipped with an 8-megapixel camera to capture participants' facial images with protruded tongues. To respect privacy and enhance presentation, we segmented tongues from facial images and released tongue images with a black background. After excluding noise samples, the dataset comprised 5,717 samples, encompassing tongue images of 3,690 Non-FLD participants and 2,027 FLD patients. Additionally, each participant’s data included eight physiological indicators: sex, age, height, waist circumference, hip circumference, weight, systolic blood pressure (SBP), and diastolic blood pressure (DBP).
![image](https://github.com/MLDMXM2017/MFF-TDF/blob/main/imgs/dataset_collection.png)
# Method and Results
Our proposed tongue diagnosis framework consisting of four steps: image preprocessing, multi-scale feature extraction, multi-source feature fusion diagnosis, and training or validation.
![image](https://github.com/MLDMXM2017/MFF-TDF/blob/main/imgs/Framework.png)
# Usage
Python: 3.8.19

    pip install -r requirements.txt
    unzip ./data/Tongue_Images.zip -d ./data
    python main.py

# Cite this repository
If you find this code or dataset useful in your research, please consider citing us:
@inproceedings{gao2024MFF-TDF,
  title={Automated Tongue Diagnosis for Fatty Liver Disease Using Multi-Source Feature Fusion Framework},
  link={https://github.com/MLDMXM2017/MFF-TDF}
}

# Reference
[https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py%20https://github.com/ndsclark/MCANet)
[https://github.com/ndsclark/MCANet](https://github.com/ndsclark/MCANet)
