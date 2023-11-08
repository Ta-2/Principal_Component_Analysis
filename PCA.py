import numpy as np

#データの取得
X = np.loadtxt('data.csv')
row = 5.0
col = 5.0

#取得データの確認
print("inputed data:")
print(X)

#平均ベクトルと偏差行列を計算
m = np.mean(X, axis=0)
Xbar = X - m
print("mean vector:")
print(Xbar)

#分散共分散行列を計算
Cov = np.dot(Xbar.T, Xbar)/row
print("Cov matrix:")
print(Cov)

#Covの固有値と固有ベクトルを取得
eig_val,  eig_vec = np.linalg.eig(Cov)
eig_vec = eig_vec.T
#固有値が全て正の値になるように調節
for idx, (val, vec) in enumerate(zip(eig_val, eig_vec)):
    if val < 0.0:
        eig_val[idx] *= -1
        eig_vec[idx] *= -1

print("eigen value:")
print(eig_val)
print("eigen vector:")
print(eig_vec)

#固有値の値が降順になるようにインデックスをソート
sorted_index = sorted(enumerate(eig_val), key=lambda x: x[1], reverse=True)
index = [pair[0] for pair in sorted_index]
print("index:")
print(index)

#寄与率を計算
eig_val_sum = np.sum(eig_val)
Proportion_of_Variance = [eig_val[idx]/eig_val_sum for idx in index]
#累積寄与率を計算
Cumulative_Proportion = []
CPsum = 0.0
for PoV in Proportion_of_Variance:
    Cumulative_Proportion.append(PoV+CPsum)
    CPsum = PoV+CPsum

print("Proportion of Variance: ")
print(Proportion_of_Variance)
print("Cumulative Proportion: ")
print(Cumulative_Proportion)

#新しい基底での座標を計算する
basis_num = 2   #用意する基底の数
new_basis = np.r_[[eig_vec[idx] for idx in index[0:basis_num]]]
print(new_basis)

#新しい基底を用いて次元削減された座標を計算
new_data = np.dot(X, new_basis.T)
print("new_basis:")
print(new_data)