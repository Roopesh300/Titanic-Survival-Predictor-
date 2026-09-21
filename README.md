# 🚢 Titanic Survival Predictor

**Author:** RoopeshRajuSV  
**Project:** End-to-end Machine Learning application for Titanic survival prediction  
**Dataset:** [Titanic Survival Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/arjunsinghgangwar/titanic-survival-prediction-dataset/data)

---

## 📋 Project Description

This project predicts whether a Titanic passenger would have survived the disaster based on personal and ticket attributes. It is built entirely in Python — covering data ingestion, preprocessing, feature engineering, model training, evaluation, and an interactive web-based user interface — with no hand-written HTML or JavaScript.

The application uses a **Random Forest Classifier** (scikit-learn) trained on the classic Titanic dataset and provides an interactive **Gradio** web interface for real-time predictions.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 🔮 **Prediction** | Enter passenger details and get instant survival odds |
| 📊 **Dataset Analytics** | Interactive charts — survival by sex, class, age & fare |
| 🤖 **Model Performance** | Confusion matrix, feature importances, classification report |
| ℹ️ **About** | Dataset facts, feature descriptions, author info, tech-stack summary |

---

## 🏗️ Technologies Used

| Layer | Library / Version |
|-------|---------|
| Data Manipulation | `pandas >= 2.0.0`, `NumPy >= 1.24.0` |
| Machine Learning | `scikit-learn >= 1.3.0` (Random Forest) |
| Visualisation | `matplotlib >= 3.7.0`, `seaborn >= 0.12.0` |
| **Frontend / UI** | **`Gradio >= 4.0.0`** — 100% Python, no HTML/JS |
| Image Support | `Pillow >= 10.0.0` |

---

## 📁 Project Structure

```
.
├── RoopeshRajuSV_TitanicSurvivalPredictor.py      ← Main Python script (full pipeline + UI)
├── requirements.txt                               ← Python dependencies
├── RoopeshRajuSV_TitanicSurvivalReport.docx       ← Full project report (Word)
├── README.md                                      ← This file
├── Titanic-Dataset.csv                            ← Dataset (must stay in same directory)
└── charts/                                        ← Generated chart images
```

---

## 🚀 Setup & Run Instructions

### Prerequisites
- Python **3.9** or higher
- `pip` package manager

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Run the Python Script

```bash
python RoopeshRajuSV_TitanicSurvivalPredictor.py
```

The app launches a local web server and opens your browser at
**http://localhost:7860** automatically.

---

## 🔮 How to Use the Predictor

1. Open the **Predict Survival** tab
2. Set the passenger details:
   - **Passenger Class** — 1st, 2nd, or 3rd
   - **Sex** — Male / Female
   - **Age** — drag the slider
   - **Fare** — ticket price in pounds
   - **Port of Embarkation** — S (Southampton) · C (Cherbourg) · Q (Queenstown)
   - **SibSp** — number of siblings / spouses aboard
   - **Parch** — number of parents / children aboard
3. Click **⚡ Predict**
4. See the verdict, survival probability percentage, and confidence level

> 💡 Use the **Quick Examples** section to auto-fill real passenger data from the original Titanic manifest.

---

## 🧠 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Classifier |
| Number of Trees | 200 |
| Max Depth | 8 |
| Min Samples Split | 4 |
| Min Samples Leaf | 2 |
| Train / Test Split | 80% / 20% (stratified) |
| Validation | 5-fold cross-validation |
| **Test Accuracy** | **~81.56%** |
| **5-Fold CV Accuracy** | **~82.38%** |

### Features Used

```
Pclass · Sex · Age · SibSp · Parch · Fare · Embarked
Title · FamilySize · IsAlone · AgeBand · FareBand
```

> **Title** is extracted from the passenger's name (Mr, Mrs, Miss, Master, Rare) and is one of the strongest predictors.

---

## 📊 Sample Results

| Passenger | Class | Sex | Age | Prediction | Survival % |
|-----------|-------|-----|-----|------------|-----------|
| Mrs. Cumings | 1st | Female | 38 | ✅ Survived | ~97% |
| Mr. Braund | 3rd | Male | 22 | ❌ Not Survived | ~15% |
| Miss. Heikkinen | 3rd | Female | 26 | ✅ Survived | ~72% |
| Miss. Bonnell | 1st | Female | 58 | ✅ Survived | ~94% |

---

## 📦 Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
gradio>=4.0.0
Pillow>=10.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🗂️ Dataset

The Titanic dataset used in this project is sourced from Kaggle:

**[Titanic Survival Prediction Dataset](https://www.kaggle.com/datasets/arjunsinghgangwar/titanic-survival-prediction-dataset/data)**  
by Arjun Singh Gangwar

| Column | Description |
|--------|-------------|
| `PassengerId` | Unique identifier for each passenger |
| `Survived` | Survival flag — 0 = No, 1 = Yes (target variable) |
| `Pclass` | Ticket class — 1 = 1st, 2 = 2nd, 3 = 3rd |
| `Name` | Full name of the passenger |
| `Sex` | Gender of the passenger |
| `Age` | Age in years |
| `SibSp` | Number of siblings / spouses aboard |
| `Parch` | Number of parents / children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare in British pounds |
| `Cabin` | Cabin number (many missing values) |
| `Embarked` | Port of embarkation — C = Cherbourg, Q = Queenstown, S = Southampton |

> The CSV file (`Titanic-Dataset.csv`) must be placed in the same directory as `RoopeshRajuSV_TitanicSurvivalPredictor.py`.

---

## 📄 License

This project is released for educational purposes.  
Dataset sourced from the public Titanic passenger manifest (Kaggle).

---

*Titanic Survival Predictor — RoopeshRajuSV · Built with Python · scikit-learn · Gradio*
