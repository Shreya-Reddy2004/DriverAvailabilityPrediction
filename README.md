# 🚗 Driver Availability Prediction

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A machine learning-powered web app built using Flask that predicts whether a driver is available or not based on location and time data. Ideal for use cases like ride-sharing systems, logistics, and delivery scheduling.

---

## 🚀 Features

- 📍 Input pickup point, day, and time
- 🧠 Predict driver availability using a trained ML model
- 📊 Visual feedback for availability status
- 🌐 Lightweight Flask web interface

---

## 🛠 Tech Stack

| Layer        | Technology |
|--------------|------------|
| Backend      | Python, Flask |
| ML Model     | Scikit-learn |
| Frontend     | HTML5, CSS3, Bootstrap |
| Dataset      | CSV-based | 

---

## 📂 Project Structure

```
Driver-Availability/
├── app.py                 # Main Flask app
├── model.pkl              # Trained ML model
├── scaler.pkl             # Preprocessing scaler
├── templates/             # HTML templates (UI)
├── static/                # CSS and assets
├── driver.csv             # Sample dataset
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 💻 How to Run Locally

```bash
# Step 1: Clone the repository
git clone https://github.com/DikshithML/Driver-Availability.git
cd Driver-Availability

# Step 2: Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the app
python app.py
```

🌐 Open your browser at: `http://localhost:5000`

---

## 🧠 Machine Learning Logic

- Trained using classification techniques (likely Decision Tree or Logistic Regression)
- Uses features like:
  - Pickup Point
  - Day of the Week
  - Time Slot
- `model.pkl` contains the trained model
- `scaler.pkl` standardizes inputs for prediction

---

## ⚠️ Notes

- Make sure both `model.pkl` and `scaler.pkl` are in the root directory
- You can retrain the model using `driver.csv` and save new `.pkl` files
- Can be extended with geolocation APIs or real-time driver data

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to use, improve, and share it.

