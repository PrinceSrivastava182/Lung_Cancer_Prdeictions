# 🫁 Lung Cancer Risk Predictor

A machine learning-powered web application for assessing the **risk of lung cancer** based on a patient's symptoms, lifestyle, and demographic features. Built with **Streamlit**, **Random Forest**, and an intuitive UI with powerful visualizations.

> ⚠️ This is a **decision-support tool**, not a diagnostic device. Always consult a healthcare provider for medical evaluation.

---

## 📌 Project Overview
This project focuses on predicting the likelihood of lung cancer in patients using machine learning models trained on relevant clinical data. The goal is to provide a scalable and interpretable system that aids early detection.

---

## 🌟 Features

- 🧠 **ML-based predictions** using a trained Random Forest Classifier
- 📊 **Gauge chart** to display real-time risk percentage
- 🔍 **Feature importance bar chart** highlighting major contributing factors
- 💬 **Personalized health tips** based on input
- 📚 **Model insights**: algorithm, accuracy, preprocessing, tuning
- 📥 **Downloadable reports** (CSV)
- 📱 **Mobile-responsive design** via Streamlit

---

## 📂 Demo

Try it live: [🔗 Lung Cancer Risk Assessment App](https://lung-cancer-risk-assessment.streamlit.app/)

---

## 🧠 ML Workflow
- **Data Preprocessing:** Missing value handling, encoding, scaling  
- **EDA:** Correlation matrix, class balance, distributions  
- **Model Training:** Logistic Regression, Random Forest, XGBoost  
- **Evaluation:** Accuracy, Precision, Recall, ROC AUC, Confusion Matrix  
- **Tuning:** Hyperparameter optimization using GridSearchCV  
- **Feature Selection:** SHAP/feature importance

---
## 📊 Visualizations
- Confusion Matrix  
- ROC Curve  
- Feature Importance (Tree-based and SHAP)  
- Class Distribution

---


## 📁 Project Structure
```text
📦 lung-cancer-risk-assessment/
├── data/                     # Raw and processed data files  
├── ml models/                # Trained ML models and tuning results  
├── scripts/                  # Data preprocessing, EDA, and training scripts  
├── app.py                    # Streamlit web application (main file)  
├── random_forest_best.pkl    # Final trained Random Forest model  
├── data_preprocessed.pkl     # Scaler and preprocessing artifacts  
├── requirements.txt          # Python dependencies  
└── README.md                 # Project overview and instructions
```
---

## 🔐 Disclaimer

> This tool is for **educational and early risk assessment** purposes only.  
> It does not replace clinical testing or medical advice.  
> Please consult licensed healthcare professionals for diagnosis or treatment.
---

## 🔧 Future Improvements
- Handle data imbalance with SMOTE  
- More robust cross-validation  
- Integration with real-time health data  
- Dockerize for containerized deployment  
- CI/CD via GitHub Actions

---

## 📜 License
This project is open-source. Feel free to modify and enhance it based on your requirements.

---
## ⚡ Challenges Faced
- **Data Imbalance:** Handling skewed classes to avoid biased models.
- **Feature Engineering:** Choosing the most impactful features for prediction.
- **Model Interpretability:** Balancing complex models with understandable outputs.
- **Deployment:** Integrating machine learning pipelines into a seamless user-facing application.

---
## 🏁 Conclusion
This project successfully demonstrates a full machine learning pipeline for predicting lung cancer risk. It highlights the importance of thorough preprocessing, model evaluation, and clear visualizations. Future upgrades aim to make the system more accurate, scalable, and ready for real-world deployment.

---



## 👨‍💻 Author
**Prince Srivastava**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/prince-srivastava3012/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PrinceSrivastava182/Lung_Cancer_Risk_Assessment)

## ⭐ Credits
- **Dataset**: Public lung cancer dataset from [Kaggle](https://www.kaggle.com/datasets/iamtanmayshukla/lung-cancer-data?resource=download)
- **Built with ❤️** using [Streamlit](https://lung-cancer-risk-assessment.streamlit.app/)
