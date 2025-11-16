import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Titanic-Dataset.csv')

print(df.describe().T)

print(df.isnull().sum())

sns.set(style="whitegrid")

# Numeric columns 
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']

# Histograms 
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Boxplots 
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Correlation matrix 
corr_matrix = df[numeric_cols + ['Survived', 'Pclass']].corr()
print(corr_matrix)

# Pairplot 
sns.pairplot(df[numeric_cols + ['Survived', 'Pclass']].dropna(), hue='Survived')
plt.show()
