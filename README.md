# 📊 Customer Churn Prediction System

A machine learning-powered web application that predicts whether a customer is likely to churn (leave a service) based on behavioral and demographic data. The project provides an interactive dashboard with prediction, analytics, and user authentication.

---

## 🚀 Features

* 🔐 **User Authentication**

  * Login & Signup functionality
  * Session-based access control

* 🏠 **Home Page**

  * Clean UI with analytics-themed background (bar charts, pie charts)
  * Entry point for users

* 🔍 **Churn Prediction**

  * Input customer details
  * Predict churn using trained ML model
  * Displays prediction result with probability

* 📊 **Analytics Dashboard**

  * Visual insights of customer data
  * Churn distribution
  * Feature relationships (e.g., tenure, monthly charges)

* 👤 **User Profile**

  * Basic user information display

* 🚪 **Logout**

  * Secure session handling

---

## 🧠 Machine Learning

* Model(s) used:

  * Logistic Regression
  * Random Forest 
  * XgBoost

* Tasks performed:

  * Data preprocessing
  * Feature encoding
  * Model training & evaluation

* Evaluation Metrics:

  * Accuracy
  * Confusion Matrix
  * Precision, Recall, F1-score

---

## 🖥️ Tech Stack

* **Frontend & Backend**: Streamlit
* **Programming Language**: Python
* **Libraries**:

  * pandas
  * numpy
  * scikit-learn
  * matplotlib / seaborn

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── app.py                  # Login / Signup (Home Page)
├── pages/
│   └── Main.py             # Dashboard (Profile, Predict, Analytics)
│
├── model/                  # Saved ML model (optional)
├── data/                   # Dataset files
├── requirements.txt        # Dependencies
└── README.md
```

---

## 🔄 Application Flow

```
Login / Signup Page
        ↓
Main Dashboard (Sidebar Navigation)
        ↓
Profile | Prediction | Analytics
        ↓
Logout → Back to Home
```

---

## 🎨 UI Design

* Modern dashboard layout with sidebar navigation
* Background includes sketched data visualizations (bar graphs, pie charts)
* Clean and responsive design
* User-friendly interaction

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### 3️⃣ Activate Environment

* Windows:

```bash
venv\Scripts\activate
```

* Mac/Linux:

```bash
source venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 📊 Example Inputs for Prediction

* Age
* Gender
* Tenure
* Monthly Charges
* Contract Type
* Internet Service
* Payment Method

---

## 🔐 Authentication Note

* Current authentication uses **session-based storage** (for demo purposes)
* Not suitable for production
* Can be upgraded using:

  * Firebase Authentication
  * Database (MySQL / MongoDB)
  * Password hashing

---

## 📈 Future Enhancements

* 📁 Upload CSV for bulk prediction
* 📊 Advanced analytics dashboard
* ☁️ Deploy on cloud (Streamlit Cloud / AWS / Render)
* 🔐 Secure authentication system
* 📉 Real-time model updates

---

## 🎯 Purpose

This project demonstrates:

* End-to-end ML workflow
* Model deployment using Streamlit
* UI/UX design for data applications
* Real-world business problem solving

---

## 👨‍💻 Author

* Architha Ojha
* GitHub: https://github.com/ARCHITHAOJHA/CustomerChurnPredictor.git

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute!
