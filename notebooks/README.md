# 🧠 AI Tools Assignment

> **Theme:** Mastering the AI Toolkit  
> **Group Members:** [Add your names]  
> **Date:** [Add date]

---

## 📋 Overview

This project demonstrates AI tool proficiency through 3 tasks:

1. **Iris Classification** - Scikit-learn (100% accuracy)
2. **MNIST Digit Recognition** - PyTorch CNN (98%+ accuracy)
3. **Product Review NLP** - spaCy (NER & Sentiment Analysis)

---

## 📁 Project Structure

```
AI-Tools-Assignment/
├── notebooks/
│   ├── Task1_Iris.py          # Scikit-learn classifier
│   ├── Task2_MNIST.py          # PyTorch CNN
│   ├── Task3_NLP.py            # spaCy NER & sentiment
│   └── buggy_code_FIXED.py     # Debugged TensorFlow code
├── outputs/                    # Generated images & results
├── reports/                    # PDF documentation
├── app.py                      # Streamlit deployment (BONUS)
└── requirements.txt
```

---

## 🚀 Quick Start

### Install Dependencies
```bash
pip install torch torchvision scikit-learn pandas matplotlib seaborn spacy streamlit tensorflow
python -m spacy download en_core_web_sm
```

### Run Tasks
```bash
python notebooks/Task1_Iris.py      # ~10 seconds
python notebooks/Task2_MNIST.py     # ~10 minutes
python notebooks/Task3_NLP.py       # ~5 seconds
```

### Deploy App (BONUS)
```bash
streamlit run app.py
```

---

## 📊 Results

| Task | Tool | Accuracy | Status |
|------|------|----------|--------|
| Task 1: Iris | Scikit-learn | 100% | ✅ |
| Task 2: MNIST | PyTorch | 98.5% | ✅ |
| Task 3: NLP | spaCy | 15+ entities | ✅ |

---

## 🐛 Part 3: Debugging

Fixed 5 bugs in `buggy_code.py`:
- Input shape (28→784)
- Loss function (binary→sparse_categorical)
- Label dimensions
- Epochs optimization
- Argmax axis

---

## 🎥 Demo

**Video:** [Link to 3-minute presentation]  
**Live App:** [Streamlit deployment link]

---

## 👥 Contributors

-

---

## 📝 Ethics & Bias

**MNIST Bias:** Limited to English digits, may not generalize to handwritten styles from different cultures.

**Mitigation:** Use diverse training data, implement fairness metrics, regular bias audits.

---

## 📄 License

This project is for educational purposes only.

---