Harmful Algal Bloom Prediction in Niger Delta Coastal Waters
Using Remote Sensing and Machine Learning
📌 Project Overview

This project develops a machine learning–based framework for predicting Harmful Algal Bloom (HAB) events in Niger Delta coastal waters using synthetic satellite-derived and water quality data. It demonstrates how remote sensing indicators and environmental variables can support early warning systems for coastal ecosystem management.

🎯 Objectives

Simulate realistic coastal water quality and oceanographic data

Model HAB occurrence using machine learning techniques

Identify key environmental drivers of algal blooms

Demonstrate an early warning decision-support workflow

🧠 Methodology

Synthetic data generation inspired by satellite ocean-color products

Feature scaling and stratified train–test split

Random Forest classification for HAB prediction

Model evaluation using precision, recall, F1-score, and confusion matrix

Feature importance analysis for environmental interpretation

📂 Project Structure
├── harmful_algal_bloom_prediction_ml.py
├── harmful_algal_bloom_niger_delta_dataset.xlsx
├── README.md

📊 Dataset Description

The dataset includes the following variables:

Sea Surface Temperature (°C)

Chlorophyll-a concentration (mg/m³)

Turbidity (NTU)

Colored Dissolved Organic Matter (CDOM index)

Nitrate concentration (mg/L)

Phosphate concentration (mg/L)

HAB event label (0 = No Bloom, 1 = Bloom)

Note: The dataset is fully synthetic and intended for research demonstration and modeling practice.

⚙️ Requirements

Python 3.8+

NumPy

Pandas

Scikit-learn

Matplotlib

Install dependencies using:

pip install numpy pandas scikit-learn matplotlib

▶️ How to Run
python harmful_algal_bloom_prediction_ml.py


The script will:

Generate synthetic data

Train a Random Forest model

Evaluate performance

Display feature importance

Simulate an HAB early warning scenario

🌍 Applications

Coastal water quality monitoring

Fisheries and aquaculture management

Environmental risk assessment

Climate and ecosystem impact studies

📌 Disclaimer

This project uses synthetic data and is intended for educational, research, and methodological demonstration purposes only.

👤 Author

AGBOZU EBINGIYE NELVIN

LinkedIn: *https://www.linkedin.com/in/agbozu-ebi/


📄 License

This project is released for academic and research use. You may adapt and extend it with appropriate attribution.
