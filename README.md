
Machine Learning Classification Framework & Cloud Deployment


1. Project Objective

The goal of this project is to construct, benchmark, and deploy five distinct machine learning classification algorithms using a standardized tabular dataset. The finalized models are serialized into reusable production artifacts and integrated into an interactive Streamlit web dashboard hosted on the Streamlit Community Cloud. This deployment enables real-time user-driven testing, allowing stakeholders to instantly evaluate test inputs, generate dynamic confusion matrices, and review classification metrics without needing to execute any underlying training code.

2. Dataset Specifics

• Dataset Identifier: Breast Cancer Wisconsin (Diagnostic) Dataset (sourced from the UCI Machine Learning Repository)
• Task Type: Binary Classification (Target categories: Malignant [0] / Benign)
• Sample Count: 569 total records
• Feature Vector: 30 continuous, real-valued attributes extracted from digitized images of fine needle aspirate (FNA) samples.
• Class Balance: 212 Malignant observations, 357 Benign observations

3. Version Control Repository

(https://github.com/ayushsinghee-ml/ml-classifier-deployment)

4. Algorithmic Comparison & Evaluation Matrix


Algorithm	Accuracy	Area Under ROC (AUC)	Precision	Recall (Sensitivity)	F1-Score	Matthews Correlation Coefficient (MCC)
Logistic Regression	0.9561	0.9977	0.9459	0.9859	0.9655	0.9068
Decision Tree	0.9474	0.9440	0.9577	0.9577	0.9577	0.8880
K-Nearest Neighbors (KNN)	0.9561	0.9959	0.9342	1.0000	0.9660	0.9086
Naive Bayes	0.9737	0.9984	0.9595	1.0000	0.9793	0.9447
Support Vector Machine (SVM)	0.9474	0.9934	0.9221	1.0000	0.9595	0.8907


5. Analytical Diagnostic of Model Performance


Algorithm	Performance Analysis & Behavioral Diagnostic
Logistic Regression	Leverages high-dimensional linearity efficiently, producing an exceptional AUC of 0.9977 along with balanced error mitigation across both target categories.
Decision Tree	Constrained by its orthogonal, axis-aligned partitioning on heavily correlated radiological inputs, leading to the lowest generalization capability (AUC: 0.9440, MCC: 0.8880).
K-Nearest Neighbors (KNN)	Reaches absolute sensitivity (1.0000 Recall) by catching every true case, though it suffers from slight boundary confusion in tight feature spaces, lowering its precision to 0.9342.
Naive Bayes	Sets the benchmark for the group with an accuracy of 0.9737 and an MCC of 0.9447. The underlying assumption of feature independence works remarkably well with these continuous metrics.
Support Vector Machine (SVM)	Optimizes structural risk boundaries perfectly for the benign class (1.0000 Recall) while maintaining high overall discriminative stability (AUC: 0.9934).
Optimal Model Selection	The Gaussian Naive Bayes Classifier stands out as the optimal choice. It leads the evaluation across all major benchmarks, achieving top marks in Accuracy (0.9737), F1-Score (0.9793), AUC (0.9984), and MCC (0.9447).
