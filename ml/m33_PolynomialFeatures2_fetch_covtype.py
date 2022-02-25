from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(datasets.feature_names)
print(datasets.DESCR)
print(x.shape, y.shape)


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=66)

# model = LinearRegression()
model = make_pipeline(StandardScaler(), LinearRegression())

model.fit(x_train, y_train)

print(model.score(x_test, y_test))  
# 0.32318370424905796

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=7, scoring='r2')
print(scores)
#[0.32356495 0.31735982 0.31296177 0.31751387 0.31760645 0.3195332
#  0.319905  ]

# import sklearn
# print(sklearn.metrics.SCORERS.keys())

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)
xp = pf.fit_transform(x)
print(xp.shape)    # (581012, 1540)

x_train, x_test, y_train, y_test = train_test_split(xp, y, test_size=0.1, random_state=66)

# model = LinearRegression()
model = make_pipeline(StandardScaler(), LinearRegression())

model.fit(x_train, y_train)

print(model.score(x_test, y_test))  
# 0.48548642075872117

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=7, scoring='r2')
print(scores)
