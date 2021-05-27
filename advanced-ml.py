import sys
sys.path.insert(0, 'data')
from data.data_description import feature_names
import pandas as pd
from sklearn.model_selection import train_test_split

# feature_names = []
# print(feature_names)
data = pd.read_csv('./data/MI.data', sep=",", names=feature_names, index_col=False)


# print(data.shape)
print(data.describe())
print(data.head(10))
# print(data.columns)
data = pd.DataFrame(data)

features = feature_names[1:112]
output = feature_names[119]
print(features)
print(output)
X = data[features].copy()
y = data[output].copy()

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)