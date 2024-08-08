from typing import List

import networkx as nx
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as mpl

from pert.pert import PERT
from .fuzzy_model import saisFuzzy_353_611 as saisFuzzy, saisFuzzyInfecting_353_611 as saisFuzzyInfecting, saisFuzzyLearning_353_611 as saisFuzzyLearning, saveFuzzyCache
from .disease_model import generateDisease

import sys
import os
# TODO: improve performance
# from multiprocessing import Pool
# from concurrent.futures import ProcessPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


step3d = 20
dx, dy = 1, 1.0/step3d
# step3d+=1


def calculate_link_strength(l, i1, i2, a1, a2):
    am = a1 + a2  # 0-2
    im = max(0, i1 - 0.25)+max(0, i2 - 0.25)  # 0-1.5
    return max(0, l-(am/8)-(im/3))  # 0,l-0.75
    # return max(0, l-(am/2)-(im))  # 0,l-0.75


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))


def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())

# a = np.array([1,2,3,4,5,6,7,8,9])
# b=a+5
# print(a.sum()/len(a),geo_mean(a), b.sum()/len(b), geo_mean(b))

# SAIRS model


class FuzzyRandomSimulator:
    def __init__(self, G: nx.Graph, name, shortName, time, p0_inf, p0_aware, diseasePertList: List[PERT],
                 timeStep=1, beta=0.2, learning_rate=0.2, forgetting_rate=0.2,
                 imagePath=None, diseaseDelay=(1, 5), diseaseRecovered=(50, 200),reduce_connection=True) -> None:
        self.G = G
        G.name = name
        self.name = name
        self.shortName = shortName
        self.timeStep = timeStep
        self.beta = beta
        self.beta_max = beta*10
        self.beta_step = self.beta_max/step3d
        self.diseasePertList = diseasePertList
        self.diseaseDelay = diseaseDelay
        self.diseaseRecovered = diseaseRecovered
        self.learning_rate = learning_rate
        self.forgetting_rate = forgetting_rate
        self.time = time
        self.aThreshold = 0.5
        self.imagePath = imagePath
        self.reduce_connection = reduce_connection
        self.time_x = np.arange(0, time+dx, dx)
        self.result_y = np.arange(0, 1.0 + dy, dy)
        self.result_y1 = np.arange(
            0, self.beta_max + self.beta_step, self.beta_step)
        self.result_y2 = np.arange(-1.0, 1.0 + dy, dy)
        self.edgeCount = G.number_of_edges()
        self.nodeCount = G.number_of_nodes()
        self.X, self.Y = np.meshgrid(self.time_x, self.result_y)
        self.X1, self.Y1 = np.meshgrid(self.time_x, self.result_y1)
        self.X2, self.Y2 = np.meshgrid(self.time_x, self.result_y2)

        self.st_i = np.zeros(time+dx)  # per H/L
        self.st_a = np.zeros(time+dx)
        self.st_bn = np.zeros(time+dx)
        self.st_tn = np.zeros(time+dx)
        self.st_bl = np.zeros(time+dx)
        self.st_tl = np.zeros(time+dx)

        self.old_su = np.zeros(time+dx)
        self.old_sa = np.zeros(time+dx)
        self.old_iu = np.zeros(time+dx)
        self.old_ia = np.zeros(time+dx)
        self.old_ru = np.zeros(time+dx)
        self.old_ra = np.zeros(time+dx)

        self.total_case = 0
        self.total_removed = 0
        self.total_recovered = 0
        self.total_case_time = np.zeros(time+dx)
        self.new_case_time = np.zeros(time+dx)
        self.total_removed_time = np.zeros(time+dx)
        self.total_recovered_time = np.zeros(time+dx)

        # 3d
        self.Zi = np.zeros((len(self.result_y), len(self.time_x),))
        self.Za = np.zeros((len(self.result_y), len(self.time_x),))
        self.Zbn = np.zeros((len(self.result_y1), len(self.time_x),))
        self.Ztn = np.zeros((len(self.result_y2), len(self.time_x),))
        # self.Zic = np.zeros((len(self.result_y), len(self.time_x),))
        # self.Zac = np.zeros((len(self.result_y), len(self.time_x),))
        self.initSim(p0_inf, p0_aware)

    def nodeInfectingProbability(self, nodeData, neighbors):
        infecting = 1
        for v in neighbors:
            beta_e = neighbors[v]['be']
            # infecting = pow(1-beta_e, timeStep)
            # beta_e = 1 - pow(1-beta_e, timeStep)
            infecting *= (1 - beta_e)
        # infecting = pow(infecting, timeStep)
        nodeData['bn'] = (1 - infecting)

    def nodeLearningArithmeticMean(self, nodeData, neighbors):
        learning = 0
        num = len(neighbors)
        if num <= 0:
            nodeData['tn'] = -0.1
        else:
            for v in neighbors:
                learning += neighbors[v]['lf']
            nodeData['tn'] = learning/num

    def nodeLearningGeometricMean(self, nodeData, neighbors):
        learning = 1.0
        for v in neighbors:
            learning *= (2 + neighbors[v]['lf'])
        nodeData['tn'] = learning**(1.0/len(neighbors)) - 2

    def nodeLearningProbability(self, nodeData, neighbors):
        forgetting = 1
        for v in neighbors:
            tau_e = (1+neighbors[v]['lf'])/2
            forgetting *= (1 - tau_e)
        # infecting = pow(infecting, timeStep)
        nodeData['tn'] = ((1 - forgetting)*2)-1

    def nodeLearningMax(self, node, neighbors):
        learningMax = -1
        for v in neighbors:
            learningMax = max(learningMax, neighbors[v]['lf'])
        node['tn'] = learningMax

    def updateNodeInfection(self, nodeData, t):
        if nodeData['ti'] == -1:
            if np.random.random() < nodeData['bn']:
                nodeData['disease'] = disease = generateDisease(
                    self.diseasePertList,
                    self.diseaseDelay,
                    self.diseaseRecovered)
                nodeData['ti'] = t
                nodeData['ic'] = nodeData['ic'] + 1
                self.total_case += 1
        else:
            dt = t - nodeData['ti']
            disease = nodeData['disease']
            if dt >= 0:
                if dt >= len(disease):
                    nodeData['ti'] = -1
                    nodeData['i'] = 0
                    self.total_removed += 1
                else:
                    if dt > 0 and disease[dt] == 0 and disease[dt-1] != 0:
                        self.total_recovered += 1
                    nodeData['i'] = disease[dt]

    def updateNodeLearning(self, nodeData):
        da = 0
        a = nodeData['a']
        l = nodeData['tn']
        if l > 0:
            da = self.learning_rate * l * (1 - a)
        elif l < 0:
            da = self.forgetting_rate * l * (a)
        # da = 1 - pow(1-da,timeStep)
        nodeData['a'] = a + da  # *timeStep

    def initSim(self, p_inf, p_aware):
        for u, nodeData in self.G.nodes(True):
            # nodeData = nodes[u]
            nodeData['a'] = 1 if np.random.random() < p_aware else 0
            nodeData['i'] = 1 if np.random.random() < p_inf else 0
            if nodeData['i'] == 1:
                nodeData['disease'] = generateDisease(self.diseasePertList)
                nodeData['ti'] = 0
                self.total_case += 1
            else:
                nodeData['disease'] = []
                nodeData['ti'] = -1
            nodeData['ic'] = 0
            # np.random.choice([1.0, 0.0], p=[p_inf, 1-p_inf])
            # np.random.choice([1.0, 0.0], p=[p_aware, 1-p_aware])
        # TODO: apply connection strength dist
        for u, v, edgeData in self.G.edges(data=True):
            edgeData['l'] = 1

    def updateLinkState(self, t):
        infecting = 0
        learning = 0
        for u, v, edgeData in self.G.edges(data=True):
            l = edgeData['l']
            node1 = self.G.nodes[u]
            node2 = self.G.nodes[v]
            ti1 = node1['ti']
            i1 = node1['i']
            i2 = node2['i']
            a1 = node1['a']
            a2 = node2['a']
            new_l = calculate_link_strength(l, i1, i2, a1, a2) if self.reduce_connection else l
            if ti1 == -1:
                bl = saisFuzzyInfecting(new_l, i1, i2, a1, a2)
                lf = saisFuzzyLearning(l, i1, i2, a1, a2)
            else:
                bl = saisFuzzyInfecting(new_l, i2, i1, a2, a1)
                lf = saisFuzzyLearning(l, i2, i1, a2, a1)
            # bl, lf = saisFuzzy(c, i1, i2, a1, a2) if ti1 == -1 \
            #     else saisFuzzy(c, i2, i1, a2, a1)
            edgeData['be'] = bl * self.beta
            edgeData['lf'] = lf

            # add to stat
            infecting = infecting + bl
            learning = learning + lf
        self.st_bl[t] = infecting/self.edgeCount
        self.st_tl[t] = learning/self.edgeCount
        return (infecting, learning,)

    def updateNodeState(self, t):
        zi, za = self.Zi[:, t], self.Za[:, t]
        zbn, ztn = self.Zbn[:, t], self.Ztn[:, t]

        stat_i = 0  # per H/L
        stat_a = 0
        stat_bn = 0
        stat_tn = 0
        old_ia = 0
        old_iu = 0
        old_sa = 0
        old_su = 0
        old_ra = 0
        old_ru = 0

        n = len(self.G.nodes)
        for u, nodeData in self.G.nodes(True):
            # nodeData = self.G.nodes[u]
            neighbors = self.G[u]
            self.nodeInfectingProbability(nodeData, neighbors)
            # self.nodeLearningArithmeticMean(node, neighbors)
            # self.nodeLearningGeometricMean(node, neighbors)
            self.nodeLearningMax(nodeData, neighbors)

            # calculate new i & a & ti
            self.updateNodeInfection(nodeData, t)
            self.updateNodeLearning(nodeData)

            # add stat
            ti = nodeData['ti']
            i = nodeData['i']
            a = nodeData['a']
            bn = nodeData['bn']
            tn = nodeData['tn']
            stat_i += i
            stat_a += a
            stat_bn += bn
            stat_tn += tn
            zi[int(i*step3d)] += 1
            za[int(a*step3d)] += 1
            b = int(step3d*bn/self.beta_max)
            if b >= len(zbn):
                # print('Bug', b,bn,self.beta_max)
                zbn[-1] += 1
            else:
                zbn[b] += 1
            # zbn[int(step3d*bn/self.beta_max)] += 1
            ztn[int((tn+1)*step3d)] += 1

            if i == 0:
                if ti == -1:
                    if a > self.aThreshold:
                        old_sa += 1
                    else:
                        old_su += 1
                else:
                    if a > self.aThreshold:
                        old_ra += 1
                    else:
                        old_ru += 1
            else:
                if a > self.aThreshold:
                    old_ia += 1
                else:
                    old_iu += 1
        zi /= n
        za /= n
        zbn /= n
        ztn /= n
        # Zic[i][t] = zi[i]
        # Zac[i][t] = za[i]
        self.st_i[t] = stat_i/n
        self.st_a[t] = stat_a/n
        self.st_bn[t] = stat_bn/n
        self.st_tn[t] = stat_tn/n
        self.old_sa[t] = old_sa/n
        self.old_su[t] = old_su/n
        self.old_iu[t] = old_iu/n
        self.old_ia[t] = old_ia/n
        self.old_ra[t] = old_ra/n
        self.old_ru[t] = old_ru/n

        if t > 0:
            self.new_case_time[t] = self.total_case - self.total_case_time[t-1]
        self.total_case_time[t] = self.total_case
        self.total_removed_time[t] = self.total_removed
        self.total_recovered_time[t] = self.total_recovered

    def doSimAt(self, t):
        self.updateLinkState(t)
        self.updateNodeState(t)

    def doSim(self):
        print(self.shortName, ": ", end='')
        for t in self.time_x:
            if (t % 10) == 0:
                print(t, end=',')
            self.doSimAt(t)
        print()

    def plotAverageState(self):
        fig, ax = mpl.subplots(figsize=(6, 2))
        ax.plot(self.time_x, self.st_i, label='I')
        ax.plot(self.time_x, self.st_a, label='A')
        # ax0.set_title('Node')
        ax.set_xlabel("Time")
        ax.set_ylabel("Average")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, self.time)
        ax.legend(ncol=2)
        fig.tight_layout()
        if self.imagePath:
            fig.savefig(self.imagePath + '/sim-ai-' +
                        self.shortName+'-'+str(self.time)+'.pdf')

    def plotAverageIG(self):
        fig, ax = mpl.subplots(figsize=(6, 2))
        ax.plot(self.time_x, self.st_bn, label=r'$\beta_n$')
        ax.plot(self.time_x, self.st_bl, label=r'$\beta_l$')
        ax.plot(self.time_x, self.st_tn, label=r'$\tau_n$')
        # ax0.plot(self.time_x, st_tl, label='Links learning')
        # ax0.set_title('Node & Link')
        ax.set_ylim([-1, 1])
        ax.set_xlim(0, self.time)
        ax.set_xlabel("Time")
        ax.set_ylabel("Average")
        ax.legend(ncol=3)
        fig.tight_layout()
        if self.imagePath:
            fig.savefig(self.imagePath + '/sim-rates-avg-' +
                        self.shortName+'-'+str(self.time)+'.pdf')

    def plotInfectedHist(self):
        # figs, (axs0, axs1) = mpl.subplots(1,2, figsize=(6, 2))
        nbins = 25
        cmap = mpl.get_cmap('viridis')
        fig, ax = mpl.subplots(figsize=(6, 1.6))
        # norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=0.0, vmax=1.0, base=10)
        # norm = colors.BoundaryNorm(boundaries=np.linspace(0.0, 1.0, 20), ncolors=256, extend='both')
        # im = ax0.contourf(X[:-1, :-1] + dx/2., Y[:-1, :-1] + dy/2., z, cmap=cmap, levels=levels)  # norm=norm,
        # self.Zi[0,:]=0 # clear non infected from histogram
        levels = MaxNLocator(nbins=nbins).tick_values(
            self.Zi.min(), self.Zi.max())
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        im = ax.pcolormesh(self.X, self.Y, self.Zi,
                           shading='auto', cmap=cmap, norm=norm)
        ax.plot(self.time_x, self.st_i, 'w', linewidth=0.5)

        ax.set_xlim(0, self.time)
        ax.set_xlabel("Time")
        ax.set_ylabel("I intensity")
        # ax0.set_zlabel("Density")
        # ax0.set_title("Disease Population")
        fig.tight_layout()
        fig.colorbar(im, ax=ax, ticks=np.arange(0, 1.2, 0.2))
        fig.tight_layout()
        if self.imagePath:
            fig.savefig(self.imagePath + '/sim-i-hist-' +
                        self.shortName+'-'+str(self.time)+'.pdf')

    def plotAwarenessHist(self):
        nbins = 25
        cmap = mpl.get_cmap('plasma')
        fig, ax = mpl.subplots(figsize=(6, 1.6))
        levels = MaxNLocator(nbins=nbins).tick_values(
            self.Za.min(), self.Za.max())
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        im = ax.pcolormesh(self.X, self.Y, self.Za,
                           shading='auto', norm=norm, cmap=cmap)
        ax.plot(self.time_x, self.st_a, 'w', linewidth=0.5)

        ax.set_xlim(0, self.time)
        ax.set_xlabel("Time")
        ax.set_ylabel("A intensity")
        fig.colorbar(im, ax=ax, ticks=np.arange(0, 1.2, 0.2))
        fig.tight_layout()
        if self.imagePath:
            fig.savefig(self.imagePath + '/sim-a-hist-' +
                        self.shortName+'-'+str(self.time)+'.pdf')

    def plotInfectingHist(self):
        nbins = 25
        cmap = mpl.get_cmap('inferno')
        fig, ax = mpl.subplots(figsize=(6, 1.6))
        levels = MaxNLocator(nbins=nbins).tick_values(
            self.Za.min(), self.Za.max())
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        im = ax.pcolormesh(self.X1, self.Y1, self.Zbn, shading='auto',
                           norm=norm, cmap=cmap)
        ax.plot(self.time_x, self.st_bn, 'w', linewidth=0.5)

        ax.set_xlim(0, self.time)
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$\beta_n$ intensity")
        fig.colorbar(im, ax=ax, ticks=np.arange(0, 1.2, 0.2))
        fig.tight_layout()
        if self.imagePath:
            fig.savefig(self.imagePath + '/sim-bn-hist-' +
                        self.shortName+'-'+str(self.time)+'.pdf')

    def plotLearningHist(self):
        nbins = 25
        cmap = mpl.get_cmap('magma')
        fig, ax = mpl.subplots(figsize=(6, 1.8))
        levels = MaxNLocator(nbins=nbins).tick_values(
            self.Za.min(), self.Za.max())
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        im = ax.pcolormesh(self.X2, self.Y2, self.Ztn, shading='auto',
                           norm=norm, cmap=cmap)
        ax.plot(self.time_x, self.st_tn, 'w', linewidth=0.5)

        ax.set_xlim(0, self.time)
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$\tau_n$ intensity")
        fig.colorbar(im, ax=ax, ticks=np.arange(0, 1.2, 0.2))
        fig.tight_layout()
        if self.imagePath:
            fig.savefig(self.imagePath + '/sim-tn-hist-' +
                        self.shortName+'-'+str(self.time)+'.pdf')

    def plotCompatible(self):
        fig, ax = mpl.subplots(figsize=(5, 3.2))
        ax.plot(self.time_x, self.old_su + self.old_sa,
                color='b', label='S', linestyle='-')
        ax.plot(self.time_x, self.old_su, color='c', label='SU', linestyle=':')
        ax.plot(self.time_x, self.old_sa, color='g',
                label='SA', linestyle='--')
        ax.plot(self.time_x, self.old_iu + self.old_ia,
                color='r', label='I', linestyle='-')
        ax.plot(self.time_x, self.old_iu, color='darkred',
                label='IU', linestyle=':')
        ax.plot(self.time_x, self.old_ia, color='orange',
                label='IA', linestyle='--')
        ax.plot(self.time_x, self.old_ru + self.old_ra,
                color='k', label='R', linestyle='-')
        ax.plot(self.time_x, self.old_ru, color='gray',
                label='RU', linestyle=':')
        ax.plot(self.time_x, self.old_ra, color='darkgreen',
                label='RA', linestyle='--')

        # ax.set_xlim(0, self.time)
        ax.set_xlabel("Time")
        ax.set_ylabel("old compatibly")
        ax.set_ylim(0, 1)
        ax.set_title(self.shortName + "-" +
                     str(self.learning_rate) + "x" + str(self.forgetting_rate))
        ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.1))
        # ax.legend(ncol=3)
        fig.tight_layout()

    def plotTotalCase(self):
        fig, ax = mpl.subplots(figsize=(6, 1.8))
        ax.plot(self.time_x, self.new_case_time,
                color='c', label='New', linestyle='-')
        ax.plot(self.time_x, self.total_case_time,
                color='b', label='Case', linestyle='-')
        ax.plot(self.time_x, self.total_removed_time,
                color='r', label='Removed', linestyle='-')
        ax.plot(self.time_x, self.total_recovered_time,
                color='g', label='Recovered', linestyle='-')

        # ax.set_xlim(0, self.time)
        ax.set_xlabel("Time")
        ax.set_ylabel("Total case")
        ax.set_title(self.shortName + "-" +
                     str(self.learning_rate) + "x" + str(self.forgetting_rate))
        ax.legend(ncol=3)
        fig.tight_layout()

    def plotStatsHist(self):
        # self.plotAverageState()
        # self.plotAverageIG()
        self.plotInfectedHist()
        self.plotInfectingHist()
        self.plotAwarenessHist()
        self.plotLearningHist()
