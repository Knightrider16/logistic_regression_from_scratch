import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("data.csv")

# activation function
def sigmoid(z):
    return 1/(1+np.exp(-z))

for i in range(len(df)):
    cured=df.loc[i,'Cured']
    total=df.loc[i,'Total Patients']
    pcured= cured/total
    odds=pcured / (1 - pcured)
    logit=np.log(odds)
    df.loc[i,"logits"]=logit
print(df)


x = df['Medication Dosage'].values
y= df['logits'].values

x_mean = np.mean(x)
y_mean = np.mean(y)

numerator = 0
denominator = 0

for index, row in df.iterrows():
    xi = row['Medication Dosage']
    yi = row['logits']
    numerator += (xi - x_mean) * (yi - y_mean)
    denominator += (xi - x_mean) ** 2

# Linear regression is of the form y = b0 + b1x
# general solution is given by (X'.X)^-1.X'y
# calculated b0 and b1

b1 = numerator / denominator
b0 = y_mean - (b1 * x_mean)

# calculate probability for every row based on determined values of b0 and b1
for i in range(len(df)):
    xi = df.loc[i, 'Medication Dosage']
    y = sigmoid(b1 * xi + b0)
    print("prob", y)

    df.loc[i, 'predicted_probability'] = y
    print("cured" if y > 0.5 else "not cured")


# plot decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(df['Medication Dosage'], df['predicted_probability'], color='blue', label='Predicted Probability')
plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
x_line = np.linspace(df['Medication Dosage'].min(), df['Medication Dosage'].max(), 100)
y_line = sigmoid(b1 * x_line + b0)
plt.plot(x_line, y_line, color='green', label='Logistic Regression Fit')
plt.title('Medication Dosage vs Predicted Probability of Being Cured')
plt.xlabel('Medication Dosage')
plt.ylabel('Predicted Probability')
plt.legend()
plt.grid()
plt.show()
