# -*- encoding:utf-8 -*-
import numpy as np
from numpy import linalg
import sys
import time
import pickle
reload(sys)
sys.setdefaultencoding('utf8')


class GMM():
    def __init__(self, k=3, filepath='../data/cluster.pkl'):
        self.samples = self.loadData(filepath)
        # K个聚类点component
        self.K = k
        # 数据预处理
        self.random_points = np.zeros([len(self.samples)*len(self.samples[0][0]), 2])
        i = 0
        for smp in self.samples:
            smp_len = len(smp[0])
            self.random_points[i:i+smp_len,0] = smp[0]
            self.random_points[i:i+smp_len,1] = smp[1]
            i += smp_len
        # M条数据,每条数据N个特征
        self.M, self.N = self.random_points.shape

    # 用k均值的思路初始化所有参数
    def initial(self):
        centroids = np.random.choice(range(self.M), self.K)
        centroids = self.random_points[centroids]

        # 1*K,即各个聚类的概率
        ppi = np.zeros([self.K])
        # K*N,即各个聚类点坐标
        pmiu = centroids
        # K*N*N,即各个聚类点的方差
        psigma = np.zeros([self.K, self.N, self.N])

        # M*K,距离矩阵
        distmat = np.tile(np.sum(self.random_points**2, 1).reshape([-1,1]), (1, self.K)) + \
                    np.tile(np.sum(pmiu**2, 1).reshape([1,-1]), (self.M, 1)) - \
                    2*self.random_points.dot(np.transpose(pmiu))

        # 初始化参数
        distind = distmat.argmin(1)

        for k in range(self.K):
            # 得到每个类的索引
            cur_ind = (distind==k).nonzero()[0]
            # 得到每个类的数据比例
            ppi[k] = float(len(cur_ind))/len(distind)
            # 得到每个类的协方差矩阵,N*N
            psigma[k,:,:] = np.cov(self.random_points[cur_ind].transpose())

        return ppi, pmiu, psigma



    def gmm(self):
        ppi, pmiu, psigma = self.initial()

        # 门限
        threshold = 1e-10

        Loss = -np.inf

        iter = 1
        while True:
            # M*K
            Pc = self.getProb(ppi, pmiu, psigma)

            # M*K
            gama = Pc*ppi
            gama = gama / np.sum(gama, 1).reshape(-1,1)

            # K*1
            Nk = np.sum(gama, 0).flatten()


            # 先计算协方差矩阵
            for k in range(self.K):
                shift = self.random_points-pmiu[k,:]
                psigma[k,:,:] = (np.transpose(shift).dot(np.diag(gama[:,k]).dot(shift)))/Nk[k]

            # 计算pi
            ppi = Nk/self.M
            # 计算miu
            pmiu = (np.dot(np.transpose(gama), self.random_points))/Nk.reshape([-1,1])

            # 也不一定是损失,只要使用一个会收敛的数值进行收敛判断就行,比如各个点到各自聚类点的距离之和
            curLoss = -np.sum(np.log(Pc.dot(ppi.reshape(self.K, 1))))

            # 判定是否收敛
            if abs(curLoss-Loss)<threshold:
                break
            Loss = curLoss
            print "Cur Loss:",Loss
            iter += 1

        self.Pc = Pc
        self.miu = pmiu
        self.gama = gama
        self.pi = ppi

        return Pc

    def getProb(self, ppi, pmiu, psigma):
        # N(x|pMiu,pSigma) = 1/((2pi)^(N/2))*(1/(abs(sigma))^0.5)*exp(-1/2*(x-pmiu)'psigma^(-1)*(x-pmiu))
        Pc = np.zeros([self.M, self.K])
        pi = np.pi
        N = self.N

        # 根据n维高斯公式计算概率密度矩阵
        for k in range(self.K):
            curp = psigma[k,:,:]
            # 求伪逆pinv,效率比较低.也可以求逆inv,但可能会遭遇奇异矩阵问题
            sigma_inverse = linalg.pinv(curp)
            sigma_det = linalg.det(sigma_inverse)
            if sigma_det < 0:
                sigma_det = 0.
            shift = self.random_points - pmiu[k,:]
            Pc[:,k] = (1./((2*pi)**(N/2)))*(np.sqrt(np.abs(sigma_det)))*np.exp(-0.5*np.sum(np.dot(shift,sigma_inverse)*shift, 1)).flatten()
        # M*K
        return Pc

    # 加载数据
    def loadData(self, filepath):
        with open(filepath) as fr:
            return pickle.load(fr)

    # 绘制图形
    def draw(self, dataxy):
        from matplotlib import pyplot as plt
        plt.figure(1)
        plt.scatter(dataxy[:,0],dataxy[:,1])
        plt.show()

    # 绘制聚类图形
    def drawCluster(self):
        from matplotlib import pyplot as plt
        plt.figure(1)
        all_ind = np.argmax(self.Pc, 1)
        color = ['r', 'g', 'b']
        for k in range(self.K):
            print (all_ind==k).nonzero()[0]
            x, y = self.random_points[(all_ind==k).nonzero()[0],0],self.random_points[(all_ind==k).nonzero()[0],1]
            plt.scatter(x, y , c=color[k])
        plt.show()




if __name__ == '__main__':
    gmm = GMM()
    gmm.gmm()
    gmm.draw(gmm.random_points)
    gmm.drawCluster()