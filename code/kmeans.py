# -*- encoding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import pickle
from copy import deepcopy
import random

class Kmeans():
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

    # 加载数据
    def loadData(self, filepath):
        with open(filepath) as fr:
            return pickle.load(fr)

    # k均值
    def kmeans(self, K, all=None):
        centroids = self.initial(K, all)
        if all is None:
            all = range(0, self.M)
        all = np.array(all)
        dist = np.zeros([len(all), K])
        cost = np.inf
        threshold = 1e-3
        while True:
            cur_cost = 0.
            split_parts = list()
            split_cost = list()
            split_dist_ind = list()
            for k in range(K):
                dist[:,k] = np.sqrt(np.sum(np.power(self.random_points[all,:]-centroids[k], 2), 1))
            for k in range(K):
                all_ind = np.argmin(dist, 1)
                dist_ind = (all_ind==k).nonzero()[0]
                k_ind = all[dist_ind]
                split_parts.append(k_ind)
                k_points = self.random_points[k_ind,:]
                centroids[k] = np.average(k_points, 0)
                cur_cost += np.sum(dist[dist_ind, k])
                split_cost.append(np.sum(dist[dist_ind, k]))
                split_dist_ind.append(dist_ind)

            print cur_cost
            if abs(cur_cost-cost)<threshold:
                break

            cost = cur_cost
            del split_parts
            del split_cost

        # self.centroids = centroids
        # self.dist = dist
        return dist, centroids, split_parts, cur_cost, split_cost

    def bikmeans(self, K):
        cur_num = 1
        centroids = [list(np.average(self.random_points, 0))]
        cost = [np.sum(np.sqrt(np.sum(np.power(self.random_points-centroids[0], 2), 1)))]
        all = [range(0, self.M)]
        for _ in range(K-1):
            min_cost = np.inf
            for i in range(cur_num):
                _, cur_centroids, split_parts, cur_cost, split_cost = self.kmeans(2, all[i])
                if i==0 or cost[i]-cur_cost > cost[min_ind]-min_cost:
                    min_cost = cur_cost
                    min_ind = i
                    min_centroids = cur_centroids
                    min_split_cost = split_cost
                    min_split_parts = split_parts
            centroids[min_ind] = min_centroids[0]
            centroids.append(min_centroids[1])
            all[min_ind] = min_split_parts[0]
            all.append(min_split_parts[1])
            cost[min_ind] = min_split_cost[0]
            cost.append(min_split_cost[1])

            cur_num += 1

        return all


    def kmedoids(self, K):
        centroids = self.initial(K)
        dist = np.zeros([self.M, K])
        cost = np.inf
        threshold = 1e-3
        while True:
            cur_cost = 0.
            for k in range(K):
                dist[:,k] = np.sqrt(np.sum(np.power(self.random_points-centroids[k], 2), 1))
            for k in range(K):
                all_ind = np.argmin(dist, 1)
                dist_ind = (all_ind==k).nonzero()[0]
                k_points = self.random_points[dist_ind,:]
                dist_sum = np.zeros([len(k_points)])
                for i, point in enumerate(k_points):
                    dist_sum[i] = np.sum(np.sqrt(np.sum(np.power((self.random_points[dist_ind,:]-point), 2),1)))
                best = dist_sum.argmin()
                centroids[k] = k_points[best]
                cur_cost += np.sum(dist_sum[best])

            print cur_cost
            if abs(cur_cost-cost)<threshold:
                break

            cost = cur_cost

        # self.centroids = centroids
        # self.dist = dist
        return dist, centroids


    def initial(self, K, all=None):
        if all is None:
            return self.random_points[np.random.choice(range(0,self.M), K, replace=False)]
        else:
            return self.random_points[np.random.choice(all, K, replace=False)]

    def draw(self, dist):
        plt.figure(1)
        all_ind = np.argmin(dist, 1)
        color = ['r', 'g', 'b']
        for k in range(self.K):
            x, y = self.random_points[(all_ind==k).nonzero()[0],0],self.random_points[(all_ind==k).nonzero()[0],1]
            plt.scatter(x, y , c=color[k])
        plt.show()

    def drawBiKmeans(self, points):
        plt.figure(1)
        color = ['r', 'g', 'b']
        for k in range(self.K):
            x, y = self.random_points[points[k],0],self.random_points[points[k],1]
            plt.scatter(x, y , c=color[k])
        plt.show()


if __name__ == '__main__':
    km = Kmeans()
    dist, centroids, _, _1, _2 = km.kmeans(3)
    km.draw(dist)
    points = km.bikmeans(3)
    km.drawBiKmeans(points)
    dist, _ = km.kmedoids(3)
    km.draw(dist)