from sklearn.datasets import load_boston,fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

datasets = load_boston()
# datasets = fetch_california_housing()

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
# 0.7795056314949791 그냥
# 0.77950563149498   pipeline

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=7, scoring='r2')
print(scores)
# [0.83841344 0.81105121 0.65897081 0.63406181 0.71122933 0.51831124
#  0.73634677]

# import sklearn
# print(sklearn.metrics.SCORERS.keys())

################################ PolynomialFeatures 후 #################################################


from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)
xp = pf.fit_transform(x)
print(xp.shape)    # (506, 105)

x_train, x_test, y_train, y_test = train_test_split(xp, y, test_size=0.1, random_state=66)

# model = LinearRegression()
model = make_pipeline(StandardScaler(), LinearRegression())

model.fit(x_train, y_train)

print(model.score(x_test, y_test))  
# 0.7795056314949791 그냥
# 0.77950563149498   pipeline

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=7, scoring='r2')
print(scores)
