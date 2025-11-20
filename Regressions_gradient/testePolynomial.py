from statistics import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def fake_data(nb_points):
    x_data = np.random.uniform(low=-3.0, high=10.0, size=nb_points)
    
    y_veritable = 10 * np.sin(x_data) / x_data

    bruit_gaussien = np.random.normal(loc=0.0, scale=1.0, size=nb_points)
    
    y_data_bruitee = y_veritable + bruit_gaussien
    
    return x_data, y_data_bruitee

x, y = fake_data(500)
x_reshaped = x.reshape(-1, 1)

poly_features = PolynomialFeatures(degree=8, include_bias=False)

x_poly = poly_features.fit_transform(x_reshaped)

sgd_reg=LinearRegression()
sgd_reg.fit(x_poly, y)

x_min = x.min()
x_max = x.max()

x_line = np.linspace(x_min, x_max)
x_line_reshaped = x_line.reshape(-1, 1)
x_line_poly = poly_features.fit_transform(x_line_reshaped)

y_line=sgd_reg.predict(x_line_poly)

fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(x, y)
ax.plot(x_line, y_line, 'r')


ax.set_title(f'Génération de 15 points bruités pour l\'ensemble d\'apprentissage')
ax.grid(True, alpha=0.5)
plt.show()