import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# データの読み込み
boston = load_boston()

# 訓練データ、テストデータに分割
X, Xtest, y, ytest = train_test_split(boston['data'], boston['target'], test_size=0.2, random_state=2)

# 交差検証データのジェネレーター
def gen_cv():
    m_train = np.floor(len(y)*0.75).astype(int)#このキャストをintにしないと後にハマる
    train_indices = np.arange(m_train)
    test_indices = np.arange(m_train, len(y))
    yield (train_indices, test_indices)

# 訓練データを基準に標準化（平均、標準偏差で標準化）
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
# テストデータも標準化
Xtest_norm = scaler.transform(Xtest)
# ハイパーパラメータのチューニング
# 計算に時間がかかるのである程度パラメーターを絞っておいた
# （1e-2～1e4まで12×12でやって最適値が'C': 0.123, 'epsilon': 1.520）
params_cnt = 20
params = {"C":np.logspace(0,1,params_cnt), "epsilon":np.logspace(-1,1,params_cnt)}
gridsearch = GridSearchCV(SVR(kernel="linear"), params, cv=gen_cv(), scoring="r2", return_train_score=True)
gridsearch.fit(X_norm, y)
print("C, εのチューニング")
print("最適なパラメーター =", gridsearch.best_params_)
print("精度 =", gridsearch.best_score_)
print()

# チューニングしたハイパーパラメーターをフィット
regr = SVR(kernel="linear", C=gridsearch.best_params_["C"], epsilon=gridsearch.best_params_["epsilon"])
train_indices = next(gen_cv())[0]
valid_indices = next(gen_cv())[1]
regr.fit(X_norm[train_indices, :], y[train_indices])
print("切片と係数")
print(regr.intercept_)
print(regr.coef_)
print()
# テストデータの精度を計算
print("テストデータにフィット")
print("テストデータの精度 =", regr.score(Xtest_norm, ytest))
print()
print("※参考")
print("訓練データの精度 =", regr.score(X_norm[train_indices, :], y[train_indices]))
print("交差検証データの精度 =", regr.score(X_norm[valid_indices, :], y[valid_indices]))
