# 🌱 PlantProtein AI — Intelligent Protein Blend Optimizer

An AI-powered system that generates scientifically valid plant-based protein blends optimized to match the amino acid profile of egg protein using advanced optimization and machine learning.

---

## 🚀 Overview

PlantProtein AI is a hybrid system that combines:

* Mathematical optimization (SLSQP)
* Machine learning predictions
* Nutritional science (FAO/WHO standards)

To generate **realistic, high-quality plant protein blends** that approximate the amino acid profile of egg protein.

---

## 🎯 Key Features

### 🧠 Smart Optimization Engine

* Uses **Scipy SLSQP** for constrained optimization
* Enforces strict nutritional constraints:

  * Max ingredient weight ≤ 60%
  * Low-protein ingredients ≤ 5%
  * Target protein ≥ 20g/100g

---

### 🤖 AI-Powered Scoring

* Hybrid scoring system:

```
Score = 0.4 × Cosine Similarity + 0.4 × PDCAAS + 0.2 × ML Prediction
```

* ML model trained on:

  * 603 real foods
  * 5000 synthetic blends
  * R² ≈ 0.97

---

### 🧬 Amino Acid Matching

* Compares blend vs **Egg Protein Reference**
* Uses:

  * Cosine similarity (shape)
  * PDCAAS / DIAAS (quality)

---

### ⚠️ Intelligent Fallback System

When no exact solution exists:

* System generates the **closest valid blend**
* Removes low-protein ingredients automatically
* Maintains scientific validity

---

### 🍽 AI Recipe Generation

* Converts blends into real-world recipes
* Fully safe (no crashes, fallback supported)

---

## 📊 Scientific Foundations

* FAO / WHO protein standards
* Essential Amino Acid (EAA) modeling
* PDCAAS & DIAAS calculations
* g/100g normalization for fair comparison

---

## 🧱 Project Structure

```
├── app.py              # Main backend (Flask + optimizer)
├── index.html          # Frontend UI
├── ml_model.py         # ML training pipeline
├── merged.xlsx         # Food dataset (603 foods)
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python app.py
```

ثم افتح:

```
http://localhost:5000
```

---

## 🧪 Example Output

* Protein: 21.2g / 100g
* Similarity to Egg: 94.8%
* Blend:

  * Chickpeas (60%)
  * Pumpkin Seeds (40%)

---

## ⚠️ Handling Edge Cases

### Optimization Failure

If constraints conflict:

* System does NOT crash ❌
* System returns:
  ✔ Closest valid blend
  ✔ Scientifically acceptable output

---

### Data Safety

* No undefined values
* No invalid arrays
* Full frontend/backend validation

---

## 🧠 Design Philosophy

> The system prioritizes **scientific correctness over forced results**

* No fake values
* No artificial inflation
* No breaking constraints

---

## 🎓 Academic Value

This project demonstrates:

* Constraint-based optimization
* Hybrid AI systems
* Nutritional modeling
* Real-world ML application

---

## 🏁 Conclusion

PlantProtein AI is not just a prototype —
it is a **robust, explainable, and scientifically grounded AI system** capable of generating real, defensible nutritional solutions.

---

## 👨‍💻 Author

Mahmoud Abdo
Graduation Project — AI & Nutrition Optimization

---

## 📌 Notes

* All values normalized to **g/100g**
* PDCAAS may be shown unclipped for analysis
* System designed for transparency and academic defense

---

## 🔥 Future Improvements

* Top-N alternative blends
* Real-time constraint explanation
* Advanced recipe generation
* UI analytics dashboard

---

**"When no exact solution exists, the system finds the closest valid truth — not a fake answer."**
