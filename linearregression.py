from sklearn.linear_model import LinearRegression

from sklearn import datasets

from sklearn.model_selection import train_test_split
 
import matplotlib.pylab as plt
import numpy as np



diabetes = datasets.load_diabetes()
print(diabetes.data.shape)

print(diabetes.target.shape)

#print(diabetes.feature_names)

X_train, X_test, y_train , y_test = train_test_split(diabetes.data, diabetes.target, test_size =.2, random_state=0)


model = LinearRegression()

model.fit(X_train,y_train)


print(model.score(X_test,y_test))
print("Coefficients:")
print(model.coef_)

print("Intercept")
print(model.intercept_)

y_pred = model.predict(X_test)

plt.plot(y_test,y_pred,'.')

x = np.linspace(0,330,100)

y=x

plt.plot(x,y)
plt.show()


