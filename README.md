# AutoDomain AI 🚀

Smart Research Paper Domain Classifier using Machine Learning and NLP.

## 📌 About Project

**AutoDomain AI** is an AI-powered web application that classifies research papers into Computer Science domains using the paper’s **title** and **abstract**.
The project uses **TF-IDF Vectorization** and **Logistic Regression** to predict the most relevant research domain.

The application is built using **Python**, **Streamlit**, **Scikit-learn**, and **Plotly** for interactive visualizations.

Main application logic is implemented in `app.py` 
Model training pipeline is implemented in `main.py` 

---

# ✨ Features

* 📄 Research paper domain prediction
* 📊 Confidence score visualization
* 🧠 Explainable AI keywords
* 📁 CSV file batch prediction
* 🔍 Research paper comparison
* 📈 Domain trend visualization (2010–2024)
* 📜 Prediction history
* 📥 PDF report generation
* 🎨 Interactive dashboard with Plotly charts
* 📄 PDF text extraction support

---

# 🛠️ Technologies Used

## Frontend

* Streamlit
* HTML/CSS
* Plotly

## Backend

* Python

## Machine Learning

* Scikit-learn
* Logistic Regression
* TF-IDF Vectorizer

## Data Processing

* Pandas
* NumPy

## Visualization

* Matplotlib
* Plotly

## PDF Handling

* PyPDF2
* ReportLab

---

# 📂 Project Structure

```bash
AutoDomain-AI/
│
├── app.py
├── main.py
├── model.pkl
├── vectorizer.pkl
├── metrics.json
├── confusion.npy
│
├── data/
│   └── arxiv-metadata-oai-snapshot.json
│
├── utils/
│   ├── preprocess.py
│   └── helper.py
│
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/AutoDomain-AI.git
cd AutoDomain-AI
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Project

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 🤖 Machine Learning Model

The model is trained using:

* **Algorithm:** Logistic Regression
* **Vectorizer:** TF-IDF
* **Dataset:** arXiv Metadata Dataset
* **Training Samples:** 50,000 papers
* **Balanced Classes:** 3000 samples per category

Training process available in `main.py` 

---

# 📊 Model Accuracy

Current trained model accuracy:

```json
Accuracy: 62.21%
```

Stored in `metrics.json` 

---

# 🧠 Supported Features

## 📄 Predict Domain

Predicts top 3 domains with confidence score.

## 🔍 Compare Papers

Compares two research papers and calculates:

* Similarity score
* Shared domains
* Unique domains

## 📁 File Match

Upload two CSV files and:

* Predict domains
* Find top 5 similar papers

## ⏳ Time Machine

Visualizes growth of CS domains from arXiv dataset.

## 📊 Dashboard

Interactive analytics for research categories.

## 📜 History

Stores previous prediction history during session.

---

# 📷 Screenshots

Add screenshots here:

```bash
screenshots/
```

Example:

* Home Page
* Prediction Page
* Dashboard
* Comparison Result

---

# 📦 Required Python Libraries

```txt
streamlit
pandas
numpy
scikit-learn
plotly
matplotlib
PyPDF2
reportlab
joblib
```

---

# 🚀 Future Improvements

* Deep Learning based classification
* BERT / Transformer integration
* Multi-label classification
* User authentication
* Database integration
* API deployment
* Research recommendation system

---

# 👨‍💻 Author

Developed by **Karmadipsinh Jadeja** **&** **Jenish Khakhkhar**

---

# 📜 License

This project is for educational and research purposes.

---

# 🙌 Acknowledgment

* arXiv Dataset
* Streamlit
* Scikit-learn
* Plotly
* Open Source Community
