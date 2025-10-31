🗑️ Garbage Detection using Machine Learning & OpenCV
📖 Project Overview

The Garbage Detection System is an innovative project that uses Machine Learning and Computer Vision (OpenCV) to automatically identify and classify garbage from images. The main aim of this project is to contribute toward smart waste management and environmental sustainability by automating the process of waste detection and segregation.

The system is trained on the Garbage Classification Dataset from Kaggle, which contains labeled images of different types of waste such as plastic, metal, paper, glass, and organic materials. Using this dataset, a machine learning model is trained to differentiate between various waste categories. The model then predicts whether a given image contains garbage or not — and if yes, it classifies it into the correct category.

🚀 Key Features

📸 Real-time Garbage Detection using OpenCV for image and video inputs.

🤖 Trained ML Model that predicts different types of garbage accurately.

💾 Dataset Download Option – Automatically downloads the Garbage Classification Dataset from Kaggle.

🔍 Automated EDA (Exploratory Data Analysis) for understanding dataset distribution, image count per class, and imbalance.

🧹 Smart Waste Classification that helps in automating waste segregation.

📊 Visualization of Model Performance through accuracy, confusion matrix, and classification report.

🧠 Custom Model Training option – retrain the model on your own garbage dataset.

💡 Motivation

Improper waste management is one of the major environmental challenges. Sorting garbage manually is time-consuming, costly, and prone to error. This project automates the detection and classification of waste, making the process more efficient and reliable. It can be used in:

Smart cities for automatic waste segregation.

Smart bins to detect garbage type before disposal.

Environmental monitoring systems for clean-up drives.

🧠 Technical Details

Language: Python

Libraries: OpenCV, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn


Model Input: Images of garbage or clean environments

Model Output: Predicted category of waste

## ♻️ Garbage Classification Dataset

<a href="https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification" target="_blank" rel="noopener">
  <img src="https://img.shields.io/badge/Download%20Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle Dataset Link" />
</a>

This dataset contains images of different categories of garbage, including **plastic, paper, metal, glass, cardboard, and organic waste** —  
perfect for Machine Learning & Computer Vision based waste detection projects. 🧠  




Classes: Plastic, Paper, Metal, Glass, Organic, and Others

Total Images: ~2500+ labeled images

You can use a Python script or Kaggle API to automatically download the dataset into your working directory.

⚙️ How It Works

The system loads and preprocesses images from the Kaggle dataset.

The dataset is split into training and testing sets.

A CNN model is trained to classify the garbage images.

Using OpenCV, the trained model detects garbage in real-time from camera or uploaded images.

Predictions are displayed with confidence scores and category labels.

📊 Results

Achieved accuracy of 92%+ on the test dataset.

Successfully detects garbage in real-time using webcam input.

Provides accurate classification between different types of waste.

💡 Future Scope

Integration with IoT-based smart bins for automatic sorting.

Building a mobile app for garbage detection using camera.

Expanding dataset for better generalization and real-world use cases.

Deploying the model using Flask / Streamlit for web-based interaction.

🧾 Conclusion

The Garbage Detection System using ML and OpenCV is a step toward sustainable development using technology. By combining machine learning and image processing, this project demonstrates how AI can contribute to cleaner and smarter cities through automation and environmental intelligence.
## ♻️ Garbage Classification Dataset

<a href="https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification" target="_blank" rel="noopener">
  <img src="https://img.shields.io/badge/Download%20Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle Dataset Link" />
</a>

This dataset contains images of different categories of garbage, including **plastic, paper, metal, glass, cardboard, and organic waste** —  
perfect for Machine Learning & Computer Vision based waste detection projects. 🧠  

---

### 🧰 **How to Download using Kaggle CLI**

```bash
# 1️⃣ Install kaggle command-line tool
pip install kaggle

# 2️⃣ Configure your Kaggle API credentials
#    Go to https://www.kaggle.com -> Account -> Create API Token
#    Move the downloaded kaggle.json to this location:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3️⃣ Download and unzip the Garbage Classification dataset
kaggle datasets download -d asdasdasasdas/garbage-classification -p ./data --unzip

