# Signal Detection Analysis — UCI Coursework Project

This project implements a signal detection theory (SDT) analysis to assess perceptual sensitivity and decision-making in a psychological task. It includes calculation of hit/miss/false alarm rates, d-prime (d′), criterion (c), and ROC curve visualization.

## Features
- Computation of SDT metrics (d′, c)
- ROC curve generation and AUC analysis
- Data input/output from structured datasets
- Unit-tested analysis pipeline

## Tools & Libraries
Python, NumPy, Matplotlib

## Analysis
The SDT model reveals that the trial difficulty (easy vs hard) had a significant negative effect on sensitivity with d' = -3.157, while stimulus type (simple vs complex) had little to no impact with d' = 0.018. The criterion additionally shows that difficulty manipulation has a stronger effect than stimulus type manipulation. The delta plots for each participant revealed that response times increased across trial difficulties while manipulation of stimulus types resulted in little change. Both models show that trial difficulty substantially affects reaction time and overall decision sensitivity whereas stimulus type has little to no impact.
