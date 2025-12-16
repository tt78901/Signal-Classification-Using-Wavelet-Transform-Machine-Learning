# Signal Classification Using Wavelet Transform & Machine Learning

## Overview

This repository contains an end-to-end **signal processing and machine learning pipeline** that leverages **Wavelet Transform–based feature extraction** for effective classification of time-series data. The project demonstrates how transforming signals into the **time–frequency domain** helps capture non-stationary patterns that are often missed by traditional Fourier-based methods.

The implementation is experiment-driven and documented in a Jupyter Notebook, making it suitable for learning, experimentation, and extension to real-world signal analytics problems.

---

## Key Features

* **Discrete Wavelet Transform (DWT)** for time–frequency analysis
* Robust **feature extraction** from wavelet coefficients
* **Supervised ML models** for classification
* Clean separation of **training, testing, and real-world inference results**
* Reproducible experiments documented in a Jupyter Notebook

---

## Tech Stack

* **Python**
* **NumPy & Pandas** – data handling
* **PyWavelets** – wavelet transforms
* **Scikit-learn** – machine learning models
* **Matplotlib / Seaborn** – visualization
* **Jupyter Notebook** – experimentation

---

## Repository Structure

```
├── Wavelet Transformation project.ipynb   # Core analysis, feature extraction, and modeling
├── train.csv                              # Training dataset
├── test.csv                               # Validation/Test dataset
├── real_test_result.csv                   # Final predictions on unseen data
```

---

## Methodology

1. **Data Loading & Preprocessing**
   Raw signal data is loaded from CSV files and normalized to ensure consistency across samples.

2. **Wavelet Transformation**
   Discrete Wavelet Transform (DWT) is applied to decompose signals into multiple frequency bands, preserving both time and frequency information.

3. **Feature Engineering**
   Statistical features (energy, mean, variance, entropy, etc.) are extracted from wavelet coefficients to form a compact and informative feature vector.

4. **Model Training**
   Supervised learning models are trained on wavelet-domain features using the training dataset.

5. **Evaluation & Inference**
   Models are evaluated on the test dataset and finally used to generate predictions on real-world unseen data.

---

## Why Wavelets?

Traditional FFT-based approaches lose temporal localization. Wavelets overcome this limitation by enabling **multi-resolution analysis**, making them ideal for:

* Biomedical signals (ECG, EEG)
* Sensor and IoT data
* Fault detection systems
* Non-stationary time-series classification

---

## How to Run

```bash
pip install numpy pandas pywavelets scikit-learn matplotlib seaborn
jupyter notebook
```

Open **Wavelet Transformation project.ipynb** and run the cells sequentially.

---

## Results

The wavelet-based feature representation provides improved signal interpretability and stronger classification performance compared to raw time-domain features, especially for signals with localized frequency variations.

---

## Future Improvements

* Hyperparameter optimization and model comparison
* Deep learning on wavelet coefficients
* Automated feature selection
* Deployment as an inference API

---

## License

This project is open for educational and research purposes.

---

## Author

Built as a hands-on exploration of **signal processing + machine learning**, with a focus on clarity, reproducibility, and real-world applicability.
