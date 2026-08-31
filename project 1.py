import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_excel('study_hours_vs_score.xlsx')
x = df[['Hours Studied (X)']]
y = df[['Exam Score (Y)']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

print('Gradient of model is: ',model.coef_,'Y intercept of model is: ',model.intercept_)

prediction_input = pd.DataFrame([[8]], columns=['Hours Studied (X)'])
print('Model Score when x is 8: ', model.predict(prediction_input))

plt.plot(x, y)
plt.xlabel('Hours Studied (X)')
plt.ylabel('Exam Score (Y)')
plt.title('Study Hours vs Exam Score')
plt.show()