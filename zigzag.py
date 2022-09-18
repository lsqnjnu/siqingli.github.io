from scipy import linalg
import numpy as np
from numpy import pi
# from numpy import 
import cmath
import matplotlib.pyplot as plt

def hamiltonian(k, N, M, t1):  # graphene哈密顿量（N是条带的宽度参数）
    # 初始化为零矩阵
    h00 = np.zeros((4*N, 4*N), dtype=complex)
    h01 = np.zeros((4*N, 4*N), dtype=complex)

    # 原胞内的跃迁h00
    for i in range(N):
        h00[i*4+0, i*4+0] = M
        h00[i*4+1, i*4+1] = -M
        h00[i*4+2, i*4+2] = M
        h00[i*4+3, i*4+3] = -M

        # 最近邻
        h00[i*4+0, i*4+1] = t1
        h00[i*4+1, i*4+0] = t1
        h00[i*4+1, i*4+2] = t1
        h00[i*4+2, i*4+1] = t1
        h00[i*4+2, i*4+3] = t1
        h00[i*4+3, i*4+2] = t1
    for i in range(N-1):
        # 最近邻
        h00[i*4+3, (i+1)*4+0] = t1
        h00[(i+1)*4+0, i*4+3] = t1

    # 原胞间的跃迁h01
    for i in range(N):
        # 最近邻
        h01[i*4+1, i*4+0] = t1
        h01[i*4+2, i*4+3] = t1

    matrix = h00 + h01*cmath.exp(1j*k) + h01.transpose().conj()*cmath.exp(-1j*k)
    return matrix

def main():
    
    X = []
    Y = []
    for k in np.linspace(-pi,pi,500):
        H = hamiltonian(k,50,0,1)
        l,v = linalg.eig(H)
        E = list(l)
        
        for i in range(len(E)):
            X.append(k)
            Y.append(E[i])
        print(k)
    
    plt.scatter(X,Y,c= 'purple', s = 0.1)
    plt.xlabel('kx')
    plt.ylabel('E')
    plt.show()


if __name__ =='__main__':
    main()