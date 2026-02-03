
# ============================================================
# Harmful Algal Bloom Prediction in Niger Delta Coastal Waters
# Using Remote Sensing and Machine Learning (Synthetic Data)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# 1. Synthetic Data Generator
# ------------------------------------------------------------

def generate_hab_data(samples=3000):
    np.random.seed(42)

    sea_surface_temp_c = np.random.normal(29.5, 1.5, samples).clip(25, 35)
    chlorophyll_mg_m3 = np.random.lognormal(mean=1.2, sigma=0.5, size=samples).clip(0.5, 60)
    turbidity_ntu = np.random.normal(18, 7, samples).clip(2, 60)
    cdom_index = np.random.normal(1.8, 0.6, samples).clip(0.3, 4.5)
    nitrate_mg_l = np.random.normal(2.5, 1.2, samples).clip(0.2, 8)
    phosphate_mg_l = np.random.normal(0.9, 0.4, samples).clip(0.05, 3)

    risk_score = (
        0.30 * (chlorophyll_mg_m3 / 60) +
        0.20 * (sea_surface_temp_c / 35) +
        0.15 * (nitrate_mg_l / 8) +
        0.15 * (phosphate_mg_l / 3) +
        0.10 * (turbidity_ntu / 60) +
        0.10 * (cdom_index / 4.5)
    )

    hab_event = (risk_score > 0.45).astype(int)

    return pd.DataFrame({
        "sea_surface_temperature_c": sea_surface_temp_c,
        "chlorophyll_mg_m3": chlorophyll_mg_m3,
        "turbidity_ntu": turbidity_ntu,
        "cdom_index": cdom_index,
        "nitrate_mg_l": nitrate_mg_l,
        "phosphate_mg_l": phosphate_mg_l,
        "hab_event": hab_event
    })

# Generate dataset
df = generate_hab_data()

# ------------------------------------------------------------
# 2. Train-Test Split & Scaling
# ------------------------------------------------------------

X = df.drop("hab_event", axis=1)
y = df["hab_event"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 3. Machine Learning Model
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# 4. Evaluation
# ------------------------------------------------------------

y_pred = model.predict(X_test)

print("Harmful Algal Bloom Prediction Performance")
print("-----------------------------------------")
print(classification_report(
    y_test, y_pred,
    target_names=["No Bloom", "HAB Detected"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------
# 5. Feature Importance
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.barh(X.columns, model.feature_importances_)
plt.xlabel("Importance Score")
plt.title("Drivers of Harmful Algal Blooms")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. Early Warning Simulation
# ------------------------------------------------------------

def hab_warning_simulation():
    sample = pd.DataFrame([{
        "sea_surface_temperature_c": 32.0,
        "chlorophyll_mg_m3": 42,
        "turbidity_ntu": 35,
        "cdom_index": 3.2,
        "nitrate_mg_l": 5.5,
        "phosphate_mg_l": 1.8
    }])

    scaled = scaler.transform(sample)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    print("\nHAB Early Warning Output")
    print("-------------------------")
    print(f"Bloom Probability: {probability:.2f}")

    if prediction == 1:
        print("HAB ALERT – High bloom risk")
    else:
        print("NO BLOOM – Low risk")

hab_warning_simulation()
