#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAT2 - Tech Salaries Dataset 2017
Ethan Zhuo
"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/edankashin/Documents/techSalaries2017.csv')


df_full = df[df['Education'].notna() & df['Race'].notna()].copy()

base_num      = ['yearsofexperience', 'yearsatcompany', 'Age', 'Height', 'Zodiac', 'SAT', 'GPA']
edu_dummies   = ['Masters_Degree', 'Bachelors_Degree', 'Doctorate_Degree', 'Some_College']
race_dummies  = ['Race_Asian', 'Race_White', 'Race_Two_Or_More', 'Race_Black']
all_pred      = base_num + edu_dummies + race_dummies

X_all = df_full[all_pred]
y_all = df_full['totalyearlycompensation']

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42)


# Question 1: OLS Multiple Regression

ols = LinearRegression().fit(X_train, y_train)

r2_tr = r2_score(y_train, ols.predict(X_train))
r2_te = r2_score(y_test,  ols.predict(X_test))
rmse_tr = np.sqrt(mean_squared_error(y_train, ols.predict(X_train)))
rmse_te = np.sqrt(mean_squared_error(y_test,  ols.predict(X_test)))

print("=== Q1: OLS ===")
print(f"R2 train={r2_tr:.3f}  R2 test={r2_te:.3f}")
print(f"RMSE train=${rmse_tr:,.0f}  RMSE test=${rmse_te:,.0f}")

# Best single predictor
best_pred, best_r2 = None, -999
for col in all_pred:
    Xi = df_full[[col]]
    Xtr, Xte, ytr, yte = train_test_split(Xi, y_all, test_size=0.3, random_state=42)
    r2 = r2_score(yte, LinearRegression().fit(Xtr, ytr).predict(Xte))
    if r2 > best_r2:
        best_r2, best_pred = r2, col

Xi_b = df_full[[best_pred]]
Xtr_b, Xte_b, ytr_b, yte_b = train_test_split(Xi_b, y_all, test_size=0.3, random_state=42)
m_b = LinearRegression().fit(Xtr_b, ytr_b)
r2_single = r2_score(yte_b, m_b.predict(Xte_b))
rmse_single = np.sqrt(mean_squared_error(yte_b, m_b.predict(Xte_b)))

print(f"Best single predictor: {best_pred}  R2={r2_single:.3f}  RMSE=${rmse_single:,.0f}")

# Q1 figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
y_pred_q1 = ols.predict(X_test)
axes[0].scatter(y_test/1000, y_pred_q1/1000, alpha=0.25, s=5, color='steelblue')
axes[0].plot([0,1500],[0,1500], 'r--', lw=1.5, label='Perfect prediction')
axes[0].set(xlabel='Actual ($K)', ylabel='Predicted ($K)',
            title='Full OLS Model\nPredicted vs. Actual', xlim=(0,1500), ylim=(0,1500))
axes[0].legend()

xr = np.linspace(df_full[best_pred].min(), df_full[best_pred].max(), 200).reshape(-1,1)
axes[1].scatter(df_full[best_pred], y_all/1000, alpha=0.15, s=5, color='steelblue')
axes[1].plot(xr, m_b.predict(xr)/1000, 'r-', lw=2)
axes[1].set(xlabel='Years of Experience', ylabel='Total Compensation ($K)',
            title=f'Best Predictor: {best_pred}\n(R²={r2_single:.3f})', ylim=(0,1500))
plt.tight_layout()
plt.savefig('q1_figure.png', dpi=150, bbox_inches='tight')
plt.close()


# Question 2: Ridge Regression

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

alphas = np.logspace(-2, 6, 100)
ridge = RidgeCV(alphas=alphas, cv=5).fit(X_tr_s, y_train)

r2_ridge_tr = r2_score(y_train, ridge.predict(X_tr_s))
r2_ridge_te = r2_score(y_test,  ridge.predict(X_te_s))
rmse_ridge_tr = np.sqrt(mean_squared_error(y_train, ridge.predict(X_tr_s)))
rmse_ridge_te = np.sqrt(mean_squared_error(y_test,  ridge.predict(X_te_s)))

print("\n=== Q2: Ridge ===")
print(f"Optimal alpha (lambda): {ridge.alpha_:.2f}")
print(f"R2 train={r2_ridge_tr:.3f}  R2 test={r2_ridge_te:.3f}")
print(f"RMSE train=${rmse_ridge_tr:,.0f}  RMSE test=${rmse_ridge_te:,.0f}")

ols_s = LinearRegression().fit(X_tr_s, y_train)
coef_compare = pd.DataFrame({'OLS': ols_s.coef_, 'Ridge': ridge.coef_}, index=all_pred)
print(coef_compare.round(1))

# Q2 figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x_pos = np.arange(len(all_pred))
w = 0.35
axes[0].bar(x_pos-w/2, ols_s.coef_, w, label='OLS', color='steelblue', alpha=0.8)
axes[0].bar(x_pos+w/2, ridge.coef_,  w, label='Ridge', color='darkorange', alpha=0.8)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(all_pred, rotation=45, ha='right', fontsize=8)
axes[0].set(ylabel='Coefficient (standardized)', title='OLS vs. Ridge Coefficients')
axes[0].axhline(0, color='black', lw=0.8); axes[0].legend()

axes[1].scatter(y_test/1000, ridge.predict(X_te_s)/1000, alpha=0.25, s=5, color='darkorange')
axes[1].plot([0,1500],[0,1500], 'r--', lw=1.5, label='Perfect prediction')
axes[1].set(xlabel='Actual ($K)', ylabel='Predicted ($K)',
            title=f'Ridge Regression\nPredicted vs. Actual (R²={r2_ridge_te:.3f})',
            xlim=(0,1500), ylim=(0,1500))
axes[1].legend()
plt.tight_layout()
plt.savefig('q2_figure.png', dpi=150, bbox_inches='tight')
plt.close()


# Question 3: Lasso Regression

lasso = LassoCV(cv=5, random_state=42, max_iter=10000).fit(X_tr_s, y_train)

r2_lasso_tr = r2_score(y_train, lasso.predict(X_tr_s))
r2_lasso_te = r2_score(y_test,  lasso.predict(X_te_s))
rmse_lasso_tr = np.sqrt(mean_squared_error(y_train, lasso.predict(X_tr_s)))
rmse_lasso_te = np.sqrt(mean_squared_error(y_test,  lasso.predict(X_te_s)))
n_zero = (lasso.coef_ == 0).sum()
zero_vars = [all_pred[i] for i in range(len(all_pred)) if lasso.coef_[i] == 0]

print("\n=== Q3: Lasso ===")
print(f"Optimal alpha (lambda): {lasso.alpha_:.2f}")
print(f"R2 train={r2_lasso_tr:.3f}  R2 test={r2_lasso_te:.3f}")
print(f"RMSE train=${rmse_lasso_tr:,.0f}  RMSE test=${rmse_lasso_te:,.0f}")
print(f"Coefficients shrunk to 0: {n_zero}  ->  {zero_vars}")
coef_lasso = pd.Series(lasso.coef_, index=all_pred)
print(coef_lasso.round(1))

# Q3 figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['tomato' if c == 0 else 'steelblue' for c in lasso.coef_]
axes[0].bar(range(len(all_pred)), lasso.coef_, color=colors, alpha=0.85)
axes[0].set_xticks(range(len(all_pred)))
axes[0].set_xticklabels(all_pred, rotation=45, ha='right', fontsize=8)
axes[0].set(ylabel='Coefficient (standardized)', title='Lasso Coefficients\n(Red = shrunk to zero)')
axes[0].axhline(0, color='black', lw=0.8)

axes[1].scatter(y_test/1000, lasso.predict(X_te_s)/1000, alpha=0.25, s=5, color='steelblue')
axes[1].plot([0,1500],[0,1500], 'r--', lw=1.5, label='Perfect prediction')
axes[1].set(xlabel='Actual ($K)', ylabel='Predicted ($K)',
            title=f'Lasso Regression\nPredicted vs. Actual (R²={r2_lasso_te:.3f})',
            xlim=(0,1500), ylim=(0,1500))
axes[1].legend()
plt.tight_layout()
plt.savefig('q3_figure.png', dpi=150, bbox_inches='tight')
plt.close()


# Question 4: Logistic Regression – gender pay gap

df_g = df[df['gender'].isin(['Male','Female'])].copy()
df_g['gender_bin'] = (df_g['gender'] == 'Female').astype(int)
controls = ['yearsofexperience', 'yearsatcompany', 'Age', 'Height', 'Zodiac', 'SAT', 'GPA']
y_g = df_g['gender_bin']

# Model 1: compensation only
X1 = df_g[['totalyearlycompensation']]
X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y_g, test_size=0.3, random_state=42)
sc1 = StandardScaler()
log1 = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log1.fit(sc1.fit_transform(X1_tr), y1_tr)
cm1 = confusion_matrix(y1_te, log1.predict(sc1.transform(X1_te)))
tn1,fp1,fn1,tp1 = cm1.ravel()

# Model 2: compensation + controls
X2 = df_g[['totalyearlycompensation'] + controls]
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y_g, test_size=0.3, random_state=42)
sc2 = StandardScaler()
log2 = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log2.fit(sc2.fit_transform(X2_tr), y2_tr)
cm2 = confusion_matrix(y2_te, log2.predict(sc2.transform(X2_te)))
tn2,fp2,fn2,tp2 = cm2.ravel()

print("\n=== Q4: Logistic Regression – Gender ===")
print(f"Model 1 coef (comp): {log1.coef_[0][0]:.4f}")
print(f"Model 1: Acc={accuracy_score(y1_te, log1.predict(sc1.transform(X1_te))):.3f}")
print(f"  Sens={tp1/(tp1+fn1):.3f}  Spec={tn1/(tn1+fp1):.3f}  Prec={tp1/(tp1+fp1):.3f}")
print(f"Model 2 coefs:")
for name, c in zip(['totalyearlycompensation']+controls, log2.coef_[0]):
    print(f"  {name}: {c:.4f}")
print(f"Model 2: Acc={accuracy_score(y2_te, log2.predict(sc2.transform(X2_te))):.3f}")
print(f"  Sens={tp2/(tp2+fn2):.3f}  Spec={tn2/(tn2+fp2):.3f}  Prec={tp2/(tp2+fp2):.3f}")

# Q4 figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, cm, title in zip(axes, [cm1, cm2],
    ['Model 1: Compensation Only\nConfusion Matrix',
     'Model 2: Compensation + Controls\nConfusion Matrix']):
    ax.imshow(cm, cmap='Blues')
    ax.set(title=title, xlabel='Predicted', ylabel='Actual',
           xticks=[0,1], yticks=[0,1])
    ax.set_xticklabels(['Male','Female']); ax.set_yticklabels(['Male','Female'])
    for i in range(2):
        for j in range(2):
            ax.text(j,i,cm[i,j],ha='center',va='center',fontsize=14,
                   color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.tight_layout()
plt.savefig('q4_figure.png', dpi=150, bbox_inches='tight')
plt.close()


# Questino 5: Logistic Regression – high vs low earner

median_comp = df['totalyearlycompensation'].median()
df['high_earner'] = (df['totalyearlycompensation'] > median_comp).astype(int)
q5_preds = ['yearsofexperience', 'Age', 'Height', 'SAT', 'GPA']
X5 = df[q5_preds]
y5 = df['high_earner']

X5_tr, X5_te, y5_tr, y5_te = train_test_split(X5, y5, test_size=0.3, random_state=42)
sc5 = StandardScaler()
log5 = LogisticRegression(max_iter=1000, random_state=42)
log5.fit(sc5.fit_transform(X5_tr), y5_tr)
y5_pred = log5.predict(sc5.transform(X5_te))
cm5 = confusion_matrix(y5_te, y5_pred)
tn5,fp5,fn5,tp5 = cm5.ravel()
acc5 = accuracy_score(y5_te, y5_pred)
sens5 = tp5/(tp5+fn5); spec5 = tn5/(tn5+fp5)
prec5 = tp5/(tp5+fp5); f1_5 = 2*prec5*sens5/(prec5+sens5)

print("\n=== Q5: High vs Low Earner ===")
print(f"Median split: ${median_comp:,.0f}")
for name, c in zip(q5_preds, log5.coef_[0]):
    print(f"  {name}: {c:.4f}")
print(f"Acc={acc5:.3f}  Sens={sens5:.3f}  Spec={spec5:.3f}  Prec={prec5:.3f}  F1={f1_5:.3f}")
print(f"CM:\n{cm5}")

# Q5 figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(cm5, cmap='Blues')
axes[0].set(title='High vs. Low Earner\nConfusion Matrix',
            xlabel='Predicted', ylabel='Actual', xticks=[0,1], yticks=[0,1])
axes[0].set_xticklabels(['Low','High']); axes[0].set_yticklabels(['Low','High'])
for i in range(2):
    for j in range(2):
        axes[0].text(j,i,cm5[i,j],ha='center',va='center',fontsize=14,
                    color='white' if cm5[i,j]>cm5.max()/2 else 'black')

axes[1].bar(q5_preds, log5.coef_[0],
            color=['steelblue' if c>0 else 'tomato' for c in log5.coef_[0]], alpha=0.85)
axes[1].set(title='Q5 Logistic Regression Coefficients\n(Standardized Predictors)',
            ylabel='Coefficient')
axes[1].axhline(0, color='black', lw=0.8)
plt.tight_layout()
plt.savefig('q5_figure.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll figures saved.")
