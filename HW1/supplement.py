#!/usr/bin/env python
# encoding: utf-8
# 叶子凌锋 1600011337 物理学院

import numpy as np
import matplotlib.pyplot as plt


# 阶乘
def fac(i):
    if i==0:
        return 1
    else:
        return i*(fac(i-1))

def transpose(A):
    # To transpose a matrix.
    # return type: List
    m = len(A); n = len(A[0])
    B = []; tmp = []
    
    for i in range(n):
        for j in range(m):
            tmp.append(A[j][i])
        B.append(tmp)
        tmp = []
    return B

def findMax(a,startingIdx):
    # 用于列支点遴选，用于找到列最大元以及其下标, max_{startingIdx<=r<=n} a_{ri}
    # arg type : a - list,  startingIdx - int
    # return type:   maxidx - int, maxnum - float(same as data in a)
    n = len(a)
    maxnum = -1
    maxidx = -1
    for i in range(startingIdx,n):
        if a[i]>maxnum:
            maxnum=a[i]
            maxidx=i;
    return maxidx,maxnum

def swap(vector,i,j):
    # 交换两个向量的值
    tmp = vector[i]
    vector[i] = vector[j]
    vector[j] = tmp

def shape(A):
    # 得到一个矩阵的尺寸
    m = len(A)
    n = len(A[0])
    return m,n

def mdot(A,B):
    # 矩阵乘法,如果不符合矩阵要求则报错
    # 注意： 若输入矩阵为一维，则需要将每一维的元素从int转为list才能正常进行运算
    Am, An = shape(A)
    Bm, Bn = shape(B)
    C = []
    if(An!=Bm):
        print("Wrong dimension.\n")
        return
    else:
        for i in range(Am):
            tmplist = []
            for j in range(Bn):
                tmp = 0
                for k in range(An):
                    tmp += A[i][k]*B[k][j]
                tmplist.append(tmp)
            C.append(tmplist)
        return C

def multiply(scale,array):
    # 矢量数乘
    n =len(array)
    ans = []
    for i in range(n):
        ans.append(array[i] * scale)
    return ans

def add(va,vb):
    # 向量加法，element-wise add
    n =len(va)
    ans = []
    for i in range(n):
        ans.append(va[i] + vb[i])
    return ans

def solveL(L,b):
    # 解下三角线性方程组，要求L为下三角方阵，Lx=b，返回x
    # return type: list
    n = len(L)
    for i in range(n-1):
        b[i] /= L[i][i]
        for j in range(i+1,n):
            b[j] -= b[i]*L[j][i]
    b[n-1] /= L[n-1][n-1]
    return b
    
def solveU(U,b):
    # 回代法求解上三角线性方程组，要求U为上三角方阵，Ux=b，返回x
    # return type: list
    n = len(U)
    i = n-1
    while i >= 0:
        b[i] /= U[i][i]
        for j in range(i):
            b[j] -= U[j][i]*b[i]
        i -= 1 
    b[0] /= U[0][0]
    return b
    
def GEM(_A,_b):
    # 带部分指点遴选的高斯消元法解Ax=b
    # return type: x - list
    A = _A; b = _b;
    n = len(b)
    x = b
    j = 0
    for i in range(n):
        idx, pivot = findMax(transpose(A)[i],i)
        if idx!=i:
            swap(A,i,idx)
            swap(b,i,idx)
        for j in range(i+1,n):
            ratio = A[j][i]/pivot
            A[j] = add(A[j] , multiply(- A[j][i]/pivot , A[i]))
            b[j] = b[j] - (ratio)*b[i]
    return solveU(A,b)

def Cholesky(_A,_b):
    # Cholesky 分解杰Ax=b
    # return type: x - list
    A = _A; b = _b
    n= len(A)
    x = b
    j = 0
    for i in range(n):
        A[i][i] = A[i][i] ** 0.5
        for j in range(i+1,n):
            A[j][i] /= A[i][i]
        for j in range(i+1,n):
            A[i][j] = 0.
            for l in range(j,n):
                    A[l][j] = A[l][j] - A[l][i] * A[j][i]
    L = A
    return solveU(transpose(L),solveL(L,b))

def constructH(dim):
    # 构造Hilbert matrix 和b= [1,1,...,1]
    # return type: H - hilbert matrix, list
    #              b - [1,1,...,1], list
    H = []
    b = []
    for i in range(dim):
        tmp = []
        for j in range(dim):
            tmp.append(1/float(i+j+1))
        H.append(tmp)
        b.append(1)
    return H,b


# 估算det(H)，利用递推关系就可以，不需要使用对数
tmp = 1
for i in range(0,11):
    print("Dimension:"+str(i),"det(H)="+str(tmp))
    tmp = (fac(i)**4) /(fac(2*i + 1)* fac(2*i))* tmp


# n 为维数，可以改变n的赋值语句来考察不同维数下得到的解
# ans_gem 为用高斯消元法得到的结果
# ans_cho 为用Cholesky分解得到的结果
n = 10
ans_gem = []
ans_cho = []
A,b = constructH(n)
ans_gem = GEM(A,b)
A,b = constructH(n)
ans_cho = Cholesky(A,b)
print(ans_gem,ans_cho)



NUM = 100


# 以下部分计算了维数由1到100变化，两种方法得到的结果的差的矢量的模长（L2模）
# 存储在res中。
res = np.zeros(NUM)
for i in range(1,NUM):
    n = i
    ans_gem = []
    ans_cho = []
    A,b = constructH(n)
    ans_gem = GEM(A,b)
    A,b = constructH(n)
    ans_cho = Cholesky(A,b)
    res[i-1] = np.linalg.norm(np.array(ans_gem)-np.array(ans_cho))
print(res)

# 以下部分计算了维数由1到100变化，两种方法得到的结果运行的时间差（每个维度运行10次取平均值）
# 存储在runtimeGEM和runtimeCho中。

import time
NUM = 100
runtimeGEM = np.zeros(NUM)
runtimeCho = np.zeros(NUM)
for i in range(1,NUM):
    rep = 10
    n = i 
    tmp = 0.
    ans_gem = []
    ans_cho = []
    for j in range(rep):
        A,b = constructH(n)
        start = time.time()
        ans_gem = GEM(A,b)
        end = time.time()
        tmp += (end-start) 
    runtimeGEM[i] = tmp/rep
    
    for j in range(rep):
        A,b = constructH(n)
        start = time.time()
        ans_cho = Cholesky(A,b)
        end = time.time()
        tmp += (end-start) 
    runtimeCho[i] = tmp/rep

print(runtimeGEM,runtimeCho)


plt.plot(range(NUM), np.log(res))
plt.title('Difference between solutions given by the two methods.')
plt.xlabel('Dimension')
plt.ylabel('Log(||x1-x2||)')

plt.show()

plt.plot(range(NUM),runtimeGEM,label='GEM runtime')
plt.plot(range(NUM),runtimeCho,label='Cholesky runtime')

plt.title('Difference between run time of the two methods.')
plt.xlabel('Dimension')
plt.ylabel('time/s')
plt.legend()
plt.show()