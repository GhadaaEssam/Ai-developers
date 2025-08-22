# Optimizing AI Developer Productivity: Insights from Habit Analysis

## 📌 Project Overview

This project explores the relationship between **developer habits** (sleep, coffee intake, AI tool usage, and distractions) and **productivity outcomes** such as task success rate, bug occurrence, commit frequency, and cognitive load.

We developed a **predictive machine learning model** capable of forecasting daily developer success based on self-reported habits. The model powers a **Streamlit web app**, *Task success predictor*, which acts as a personal productivity companion for developers.

---

## 📊 Dataset

The dataset used in this project was published on Kaggle.
You can access it here: [AI Developer Productivity Dataset (Kaggle)](https://www.kaggle.com/datasets/atharvasoundankar/ai-developer-productivity-dataset)

---

## 🚀 Key Features

* **Data Analysis & Insights**:

  * 6–8 hours of sleep → 20% higher task success rate.
  * <6 hours sleep → 87% more critical bugs.
  * 1–3 hours/day AI tool usage → 28% more commits.
  * 300–500mg coffee → 48% task success rate with 6% lower cognitive load.
  * > 5 hours/day AI usage → 23% higher cognitive load.
  * 3+ major distractions/hour → 40% higher cognitive load.

* **Machine Learning Pipeline**:

  * Outlier detection using **IQR**.
  * **Feature engineering** (habit ratios, squared usage, log transforms).
  * Handling class imbalance with **SMOTE**.
  * **Feature scaling** for model stability.
  * **Hyperparameter tuning** for optimal performance.

* **Results**:

  * Predictive model achieved **92% F1-score**.
  * Identified optimal developer profile:

    * Sleep: 6–8 hours
    * AI usage: 3–5 hours/day
    * Coffee: 300–500mg
    * Distractions: <2/hour

* **Web App (Streamlit)**:

  * Input daily habits (sleep, AI usage, caffeine, distractions).
  * Get a **personalized productivity prediction**.
  * Simple, accessible design.

---

## 🛠️ Tech Stack

* **Python** (Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib/Seaborn)
* **Streamlit** (Web App)
* **SMOTE** (Imbalanced data handling)
* **Jupyter** (Development & reporting)

---

## 📂 Project Structure

```
├── data/                # Raw and processed datasets
├── notebooks/           # Exploratory analysis & model development
├── src/                 # Preprocessing, training, evaluation scripts
├── app/                 # Streamlit app files
├── report/              # Project report (PDF)
├── README.md            # Project documentation
```

---

## 📊 Results Summary

| Factor        | Optimal Value | Effect                  |
| ------------- | ------------- | ----------------------- |
| Sleep         | 6–8 hours     | +20% task success       |
| AI Usage      | 3–5 hours/day | +66% task success rate  |
| Coffee Intake | 300–500mg     | 6% lower cognitive load |
| Distractions  | <2 major/hour | +7% code output         |

---

## 📖 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/habit-whisperer.git
   cd habit-whisperer
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:

   ```bash
   streamlit run app/main.py
   ```

---

## 🔮 Future Work

* Expand dataset with more developer samples.
* Add visualization dashboards for habit-performance trends.
* Deploy app with Docker for cloud-based usage.

---

## 📎 Links

* 📂 [GitHub Repository](https://github.com/GhadaaEssam/Ai-developers)
* 📑 [Full Project Report](https://github.com/GhadaaEssam/Ai-developers/blob/main/report/Project%20report.pdf)
* 🌐 [Habit Whisperer Demo (Streamlit)](https://task-success-predictor.streamlit.app/)

---
