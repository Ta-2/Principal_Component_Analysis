import numpy as np
import illustrator as ill
import eigen_module as em

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
eig_val, eig_vec = em.calc_sorter_eigen(Cov)

print("eigen value:")
print(eig_val)
print("eigen vector:")
print(eig_vec)

#寄与率を計算
eig_val_sum = np.sum(eig_val)
Proportion_of_Variance = eig_val/eig_val_sum
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
new_basis = eig_vec[0:basis_num]
print(new_basis)

#新しい基底を用いて次元削減された座標を計算
new_data = np.dot(X, new_basis.T)
print("new_basis:")
print(new_data)

ill.illustrate(new_data)