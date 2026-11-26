# 🌾 FarmLink – AI-Powered Agricultural Decision Support Platform

FarmLink is a Flask web application that helps farmers make data-driven decisions. It combines three machine-learning modules in one platform:

1. **Crop recommendation + price estimate** – suggests the best crop for the given soil and weather conditions and predicts its expected market price.
2. **Market demand forecasting** – an LSTM model that forecasts crop demand for the next 6 months for a chosen region and crop.
3. **Farmer–buyer matching** – a hybrid recommender that connects a buyer's requirement with the most suitable farmers.

The app also includes user registration and login, so each farmer or buyer has their own account.

---

## ✨ Features

| Module | What it does | Technique |
|---|---|---|
| **Crop Recommendation** | Takes N, P, K, temperature, humidity, pH, rainfall and soil moisture, then returns the recommended crop and the predicted marketing price | Random Forest Classifier + Random Forest Regressor |
| **Market Forecast** | Takes a region and a crop, plots actual vs. predicted demand and shows a 6-month demand forecast | 2-layer LSTM (Keras/TensorFlow), 12-month input window |
| **Buyer Matching** | Takes crop, quantity needed, maximum price and location, then returns the top 5 matching farmers with a match score | Content-based filtering (cosine similarity) with a hybrid score (0.6 × similarity + 0.4 × past-interaction score) |
| **User Accounts** | Register / login / logout | SQLite + hashed passwords (Werkzeug) |

---

## 🧠 How It Works

### 1. Crop recommendation and price prediction
- **Dataset:** `crop_new/Crop_recommendation_new.csv` – 2,200 samples, 22 crops.
- **Features (8):** `N, P, K, temperature, humidity, ph, rainfall, moisture`
- **Targets:** crop label (classification) and `marketing price` (regression).
- **Pipeline:** `MinMaxScaler` → `StandardScaler` → `RandomForestClassifier` / `RandomForestRegressor` (200 trees each), 80/20 train–test split.
- **Supported crops:** rice, maize, jute, cotton, coconut, papaya, orange, apple, muskmelon, watermelon, grapes, mango, banana, pomegranate, lentil, blackgram, mungbean, mothbeans, pigeonpeas, kidneybeans, chickpea, coffee.

### 2. Market demand forecasting
- **Dataset:** `market/dataset.csv` – monthly `Market_Demand` by `Region` and `Crop`.
- **Model:** `LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(1)`, trained with Adam / MSE and early stopping.
- **Inference:** the last 12 months of the selected region–crop series are used to forecast the next 6 months recursively. The app returns two charts (actual vs. predicted, and the 6-month forecast).

### 3. Farmer–buyer matching
- **Data:** `customer_system/` – 2,000 farmers, 2,000 buyers and 2,000 transactions (CSV).
- **Method:** crop, quantity, price and location are encoded and min-max scaled, then buyer–farmer similarity is computed with cosine similarity. A rating-based collaborative score from past transactions is blended in during offline training; for a new buyer in the web app the interaction score is zero, so ranking relies on content similarity.

---

## 🗂️ Project Structure

```
farm link/
├── app.py                      # Flask app – all routes
├── users.db                    # SQLite user database (auto-created if missing)
├── templates/                  # HTML templates (index, login, register, crop_form,
│                               #   crop_result, market, customer, about, contact)
├── static/uploads/             # Uploaded/static images
├── crop_new/                   # Crop recommendation + price module
│   ├── train.py                #   train Random Forest models
│   ├── test.py                 #   command-line prediction test
│   ├── Crop_recommendation_new.csv
│   ├── rf_crop_model.pkl, rf_price_model.pkl
│   ├── minmaxscaler.pkl, standscaler.pkl
│   └── static/images/          #   crop images
├── market/                     # Demand forecasting module
│   ├── train.py, test.py
│   ├── dataset.csv
│   └── crop_demand_model.h5, scaler.pkl
├── customer_system/            # Farmer–buyer matching module
│   ├── train.py, test.py
│   └── farmers.csv, buyers.csv, transactions.csv, matched_farmer_buyer_pairs.csv
└── video.mp4                   # Project demo video
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or later
- `pip`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install flask numpy pandas scikit-learn tensorflow matplotlib werkzeug
```

### Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

> Run the command from the folder that contains `app.py`, because the models and datasets are loaded using relative paths.

### (Optional) Retrain the models

```bash
cd crop_new && python train.py          # crop + price models
cd ../market && python train.py         # LSTM demand model
cd ../customer_system && python train.py  # buyer–farmer matches
```

---

## 🌐 Routes

| Route | Description |
|---|---|
| `/` | Home page |
| `/register`, `/login`, `/logout` | User authentication |
| `/crop_recommendation` | Crop recommendation + predicted price |
| `/market` | Market demand forecast |
| `/customer_support` | Buyer → farmer matching |
| `/about`, `/contact` | Information pages |

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, SQLite
- **ML / Data:** scikit-learn, TensorFlow / Keras, pandas, NumPy
- **Visualization:** Matplotlib (charts rendered as base64 images)
- **Frontend:** HTML, CSS (Jinja2 templates)

---

## ⚠️ Known Limitations

- The LSTM in `market/train.py` is trained on a single region–crop series and then reused for every region and crop, so forecast quality varies between selections.
- In the web app, new buyers have no transaction history, so matching uses content similarity only.
- The Flask `secret_key` is hard-coded and the app runs with `debug=True`. Move the secret to an environment variable and disable debug mode before any real deployment.

## 🔮 Future Improvements

- Per-crop / per-region demand models
- Live weather and soil-data integration
- Real farmer and buyer data with ratings feedback loop
- Deployment with Docker and a production WSGI server

---

## 👤 Author

**Sanjay A**
B.E. Computer Science and Engineering (AI & ML), S.J.C. Institute of Technology

---

## 📄 License

Add a license of your choice (for example, MIT) before publishing.
