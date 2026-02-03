# Guardian News Classifier: Multinomial NB vs. Logistic Regression

This repository explores supervised machine learning techniques to classify article extracts from **The Guardian**. The project evaluates two different feature extraction methods—**Bag of Words (BoW)** and **Term Frequency-Inverse Document Frequency (TF-IDF)**—across two robust classification models: **Multinomial Naive Bayes** and **Multinomial Logistic Regression**.

---

## 🚀 Getting Started

### Prerequisites
Before running the project, please install the dependencies listed in the requirements file:
```bash
pip install -r requirements.txt
```

Note: Ensure you are using a compatible Python version (3.8+ recommended).

## GPU Support
This project supports GPU acceleration. If you are using a **Linux system**, you may need to configure your CUDA environment or modify specific library paths to ensure the GPU is utilized correctly.

## 📂 Resources & Downloads

To keep the repository lightweight, large files are hosted externally. Please download and extract them into the appropriate directories.

**Pre-trained Models**: Download larger models here and extract them directly into the /Model folder.

    https://drive.google.com/file/d/1C9G3flGouk5MCFUMqiG81JLzpieBWqgC/view?usp=sharing

**Training Dataset**: Download the original database here and place it in the data directory.

    https://drive.google.com/file/d/1yhOijAx_4uKIZuyvHMzSu_899-Tf4akV/view?usp=sharing

## 🛠 Methodology

The project follows a standard NLP pipeline to transform raw Guardian text into classified categories.

## Feature Extraction

**Bag of Words** (BoW): Creates a vocabulary of all unique words and tracks their frequency.

**TF-IDF**: Normalizes word frequency by how often terms appear across all documents to highlight unique "signature" words.

## Classification Models
**Multinomial Naive Bayes**: A fast, baseline probabilistic classifier.

**Multinomial Logistic Regression**: A discriminative model that uses the Softmax function for multi-class prediction.
