# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
# Date:
# Developed by : ANUSHARON.S
# Register no: 212222240010
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
df = pd.read_csv('/content/Paris 2024 Olympics Nations Medals.csv')

y = df['Total'].values
x = np.arange(len(y))

A = np.vstack([x, np.ones(len(x))]).T
linear_trend = np.linalg.lstsq(A, y, rcond=None)[0]

y_linear_trend = linear_trend[0] * x + linear_trend[1]

degree = 2
coeffs = np.polyfit(x, y, degree)
y_poly_trend = np.polyval(coeffs, x)

plt.figure(figsize=(12, 6))
plt.plot(x, y, label='Original Data', marker='o')
plt.plot(x, y_linear_trend, label='Linear Trend', linestyle='--')
plt.plot(x, y_poly_trend, label='Polynomial Trend', linestyle='--')
plt.title('Linear and Polynomial Trend Lines')
plt.xlabel('Index')
plt.ylabel('Total Medals')
plt.legend()
plt.show()

print("Linear Trend Equation: y = {:.2f}x + {:.2f}".format(linear_trend[0], linear_trend[1]))
print("Polynomial Trend Equation: y = {:.2f}x^2 + {:.2f}x + {:.2f}".format(coeffs[0], coeffs[1], coeffs[2]))

```

### OUTPUT

![Screenshot 2024-08-28 093314](https://github.com/user-attachments/assets/f34179c7-d1f7-4e64-88be-2408508ab597)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
