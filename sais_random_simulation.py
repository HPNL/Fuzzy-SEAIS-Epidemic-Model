from multiprocessing.pool import Pool
from typing import List

import networkx as nx
from networkx.algorithms import community

from sais.fuzzy_simulation import FuzzyRandomSimulator
from sais.disease_model import pertList

import numpy as np
import matplotlib.pyplot as mpl
from matplotlib.colors import BoundaryNorm, TABLEAU_COLORS
from matplotlib.ticker import MaxNLocator
# from numba import jit
from scipy.signal import find_peaks

from datetime import datetime
import random
import os


np.random.seed(14000530)
random.seed(14000530)

startTime = datetime.now()


imagePath = None
# imagePath = './img'
# imagePath = './imgNew'

lineWidth = 1.5
if imagePath:
    if not os.path.exists(imagePath):
        os.mkdir(imagePath)

time = 200
n = 600
averageNeighbors = 6
p0_link = averageNeighbors/n
p0_inf = 10.0/n
p0_aware = 1.0
beta = 0.01

diseaseDelay = (1, 5)
diseaseRecovered = (50, 200)

step3d = 10
timeStep = 1
time = int(time / timeStep)
dx, dy = 1, 1.0/step3d

graph = [
    'ER1',
    'ER2',
    'ER3',
    'ER4',
    'ER5',

    'BA1',
    'BA2',
    'BA3',
    'BA4',
    'BA5',
]


simulators: List[FuzzyRandomSimulator] = []


def initSimulator():
    # G = nx.gnp_random_graph(n, p0_link)
    # G = nx.barabasi_albert_graph(n, averageNeighbors)
    if 'ER1' in graph:
        G = nx.gnp_random_graph(n, p0_link/2)
        simulators.append(FuzzyRandomSimulator(G, 'erdos_renyi', 'ER1', time,
                                               p0_inf, 1, pertList,
                                               timeStep, beta, 0.6, 0.006, imagePath,
                                               diseaseDelay, diseaseRecovered))
    if 'ER2' in graph:
        G = nx.gnp_random_graph(n, p0_link/2)
        simulators.append(FuzzyRandomSimulator(G, 'erdos_renyi', 'ER2', time,
                                               p0_inf, 0, pertList,
                                               timeStep, beta, 0.6, 0.06, imagePath,
                                               diseaseDelay, diseaseRecovered))
    if 'ER3' in graph:
        G = nx.gnp_random_graph(n, p0_link)
        simulators.append(FuzzyRandomSimulator(G, 'erdos_renyi', 'ER3', time,
                                               p0_inf, 1, pertList,
                                               timeStep, beta, 0.6, 0.006, imagePath,
                                               diseaseDelay, diseaseRecovered))
    if 'ER4' in graph:
        G = nx.gnp_random_graph(n, p0_link)
        simulators.append(FuzzyRandomSimulator(G, 'erdos_renyi', 'ER4', time,
                                               p0_inf, 0, pertList,
                                               timeStep, beta, 0.6, 0.06, imagePath,
                                               diseaseDelay, diseaseRecovered))
    if 'ER5' in graph:
        G = nx.gnp_random_graph(n, p0_link)
        simulators.append(FuzzyRandomSimulator(G, 'erdos_renyi', 'ER5', time,
                                               p0_inf, 0, pertList,
                                               timeStep, beta, 0, 0, imagePath,
                                               diseaseDelay, diseaseRecovered))
    if 'BA1' in graph:
        G = nx.barabasi_albert_graph(n, int(averageNeighbors/2))
        simulators.append(FuzzyRandomSimulator(G, 'barabasi_albert', 'BA1', time,
                                               p0_inf, 1, pertList,
                                               timeStep, beta, 0.6, 0.006, imagePath,
                                               diseaseDelay, diseaseRecovered))
    if 'BA2' in graph:
        G = nx.barabasi_albert_graph(n, int(averageNeighbors/2))
        simulators.append(FuzzyRandomSimulator(G, 'barabasi_albert', 'BA2', time,
                                               p0_inf, 0, pertList,
                                               timeStep, beta, 0.6, 0.06, imagePath,
                                               diseaseDelay, diseaseRecovered))
    if 'BA3' in graph:
        G = nx.barabasi_albert_graph(n, averageNeighbors)
        simulators.append(FuzzyRandomSimulator(G, 'barabasi_albert', 'BA3', time,
                                               p0_inf, 1, pertList,
                                               timeStep, beta, 0.6, 0.006, imagePath,
                                               diseaseDelay, diseaseRecovered))
    if 'BA4' in graph:
        G = nx.barabasi_albert_graph(n, averageNeighbors)
        simulators.append(FuzzyRandomSimulator(G, 'barabasi_albert', 'BA4', time,
                                               p0_inf, 0, pertList,
                                               timeStep, beta, 0.6, 0.06, imagePath,
                                               diseaseDelay, diseaseRecovered))
    if 'BA5' in graph:
        G = nx.barabasi_albert_graph(n, averageNeighbors)
        simulators.append(FuzzyRandomSimulator(G, 'barabasi_albert', 'BA5', time,
                                               p0_inf, 0, pertList,
                                               timeStep, beta, 0, 0, imagePath,
                                               diseaseDelay, diseaseRecovered))
    for sim in simulators:
        print(f"{sim.shortName}: node={len(sim.G.nodes)}   edge={len(sim.G.edges)}")


def doSim():
    # with Pool() as pool:
    #     pool.map(FuzzyRandomSimulator.doSim,simulators)
    #     pass
    for s in simulators:
        s.doSim()
    print()




# %%
def plotSub():
    fig, axes = mpl.subplots(1, 2, figsize=(6, 2))
    for s in simulators:
        axes[0].plot(s.time_x, s.st_i, label=s.shortName,
                     linewidth=lineWidth)
    for s in simulators:
        axes[1].plot(s.time_x, s.st_i, linewidth=lineWidth)
    axes[0].set_xlim(0, 200)
    axes[1].set_xlim(801, 1000)
    # fig.set_xlabel("Time")
    # fig.set_ylabel("Avg-I")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    fig.legend(ncol=len(simulators)/2, bbox_to_anchor=(
        0., 1.02, 1., .102), loc='upper center', borderaxespad=0.)
    fig.tight_layout()
    axes[0].spines.right.set_visible(False)
    axes[1].spines.left.set_visible(False)
    axes[0].yaxis.tick_left()
    # axes[1].tick_params(labelright='off')
    axes[1].axes.yaxis.set_ticklabels([])
    axes[1].yaxis.tick_right()
    fig.subplots_adjust(wspace=0.03, hspace=0, right=.95)
    d = 2  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    axes[0].plot([1, 1], [1, 0], transform=axes[0].transAxes, **kwargs)
    axes[1].plot([0, 0], [0, 1], transform=axes[1].transAxes, **kwargs)
    if imagePath:
        fig.savefig(imagePath + '/sim-i-all-two-sub-'+str(time)+'.pdf')


# %%
def plotTotalInfected():
    fig, ax0 = mpl.subplots(figsize=(5.5, 3))
    for s in simulators:
        ax0.plot(s.time_x, s.total_case_time,
                 label=s.shortName, linewidth=lineWidth)
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Total Infected")
    ax0.set_ylim(0, n)
    ax0.set_xlim(0, time)
    ax0.legend(ncol=len(simulators)/2, loc="lower center",
               bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    if imagePath:
        fig.savefig(imagePath + '/sim-i-total-all-'+str(time)+'.pdf')


def plotTotalInfectedAt(time: int):
    name = [sim.shortName for sim in simulators]
    x = np.arange(len(name))
    finalInfected = [s.total_case_time[time] for s in simulators]

    fig, ax = mpl.subplots(figsize=(5, 2))

    rect1 = ax.bar(x, finalInfected, color=TABLEAU_COLORS)
    ax.set_xticks(x)
    ax.set_xticklabels(name)
    # ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_ylabel('Total Infected')
    # ax.set_xlabel('Graph')
    ax.bar_label(rect1, padding=3, fmt='%2.3g', label_type='center')
    # fig.legend(ncol=len(simulators)/2, #bbox_to_anchor=(0.03, 1.0, 1.0, .08),
    #            loc='upper center', borderaxespad=0.)
    fig.tight_layout()
    if imagePath:
        fig.savefig(imagePath + '/sim-ni-'+str(time)+'.pdf')


def plotNewInfected():
    fig, ax0 = mpl.subplots(figsize=(5, 3.2))
    for s in simulators:
        ax0.plot(s.time_x, s.new_case_time,
                 label=s.shortName, linewidth=lineWidth)
    ax0.set_xlabel("Time")
    ax0.set_ylabel("New Infected")
    # ax0.set_ylim(0, 1)
    ax0.set_xlim(0, time)
    ax0.legend(ncol=len(simulators)/2, loc="lower center",
               bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    if imagePath:
        fig.savefig(imagePath + '/sim-ni-total-all-'+str(time)+'.pdf')


def plotTotalRemoved():
    fig, ax0 = mpl.subplots(figsize=(5, 3.2))
    for s in simulators:
        ax0.plot(s.time_x, s.total_removed_time,
                 label=s.shortName, linewidth=lineWidth)
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Total Removed")
    # ax0.set_ylim(0, 1)
    ax0.set_xlim(0, time)
    ax0.legend(ncol=len(simulators)/2, loc="lower center",
               bbox_to_anchor=(0.5, 1.1),)
    fig.tight_layout()
    if imagePath:
        fig.savefig(imagePath + '/sim-r-total-all-'+str(time)+'.pdf')


def plotTotalRecovered():
    fig, ax0 = mpl.subplots(figsize=(5, 3.2))
    for s in simulators:
        ax0.plot(s.time_x, s.total_recovered_time,
                 label=s.shortName, linewidth=lineWidth)
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Total Recover")
    # ax0.set_ylim(0, 1)
    ax0.set_xlim(0, time)
    ax0.legend(ncol=len(simulators)/2, loc="lower center",
               bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    if imagePath:
        fig.savefig(imagePath + '/sim-rec-total-all-'+str(time)+'.pdf')


def plotAverageInfected():
    fig, ax0 = mpl.subplots(figsize=(5.5, 3))
    for s in simulators:
        ax0.plot(s.time_x, s.st_i, label=s.shortName, linewidth=lineWidth)
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Avg-I")
    # ax0.set_ylim(0, 1)
    ax0.set_xlim(0, time)
    ax0.legend(ncol=len(simulators)/2, loc="lower center",
               bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    if imagePath:
        fig.savefig(imagePath + '/sim-i-avg-all-'+str(time)+'.pdf')


def plotAverageAwareness():
    fig, ax0 = mpl.subplots(figsize=(5.5, 3))
    for s in simulators:
        ax0.plot(s.time_x, s.st_a, label=s.shortName, linewidth=lineWidth)
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Avg-A")
    ax0.set_ylim(0, 1)
    ax0.set_xlim(0, time)
    ax0.legend(ncol=len(simulators)/2, loc="lower center",
               bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    if imagePath:
        fig.savefig(imagePath + '/sim-a-avg-all-'+str(time)+'.pdf')


def plotAverageBetaLink():
    fig, ax0 = mpl.subplots(figsize=(5, 3.2))
    for s in simulators:
        ax0.plot(s.time_x, s.st_bl,
                 label=s.shortName, linewidth=lineWidth)
    ax0.set_xlabel("Time")
    ax0.set_ylabel(r"Avg-$\beta_l$")
    ax0.set_ylim(0, 1)
    ax0.set_xlim(0, time)
    ax0.legend(ncol=len(simulators)/2, loc="lower center",
               bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    if imagePath:
        fig.savefig(imagePath + '/sim-bl-avg-all-'+str(time)+'.pdf')


def plotAverageBetaNode():
    fig, ax0 = mpl.subplots(figsize=(5.5, 3))
    for s in simulators:
        ax0.plot(s.time_x, s.st_bn,
                 label=s.shortName, linewidth=lineWidth)
    ax0.set_xlabel("Time")
    ax0.set_ylabel(r"Avg-$\beta_n$")
    # ax0.set_ylim(0, 1)
    ax0.set_xlim(0, time)
    ax0.legend(ncol=len(simulators)/2, loc="lower center",
               bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    if imagePath:
        fig.savefig(imagePath + '/sim-bn-avg-all-'+str(time)+'.pdf')


def plotAverageTauNode():
    fig, ax0 = mpl.subplots(figsize=(5.5, 3))
    for s in simulators:
        ax0.plot(s.time_x, s.st_tn,
                 label=s.shortName, linewidth=lineWidth)
    ax0.set_xlabel("Time")
    ax0.set_ylabel(r"Avg-$\tau_n$")
    ax0.set_ylim(-1, 1)
    ax0.set_xlim(0, time)
    ax0.legend(ncol=len(simulators)/2, loc="lower center",
               bbox_to_anchor=(0.5, 1.1))

    fig.tight_layout()
    if imagePath:
        fig.savefig(imagePath + '/sim-tn-avg-all-'+str(time)+'.pdf')


# %%
def plotPeakStat():
    width = 0.18
    name = [sim.shortName for sim in simulators]
    x = np.arange(len(name))
    prominence = 0.01

    xI = [sim.st_i for sim in simulators]
    xBl = [sim.st_bl for sim in simulators]
    xBn = [sim.st_bn for sim in simulators]
    xTn = [sim.st_tn for sim in simulators]
    xA = [sim.st_a for sim in simulators]

    peakI = [find_peaks(x, prominence=prominence) for x in xI]
    peakBl = [find_peaks(x, prominence=prominence) for x in xBl]
    peakBn = [find_peaks(x, prominence=prominence) for x in xBn]
    peakTn = [find_peaks(x, prominence=prominence) for x in xTn]
    peakA = [find_peaks(x, prominence=prominence) for x in xA]

    pI = [len(p[0]) for p in peakI]
    pBl = [len(p[0]) for p in peakBl]
    pBn = [len(p[0]) for p in peakBn]
    pTn = [len(p[0]) for p in peakTn]
    pA = [len(p[0]) for p in peakA]

    fig, ax0 = mpl.subplots(figsize=(6, 2))
    ax0.bar(x - width * 2, pI, color='red', width=width, label='I')
    ax0.bar(x - width * 1, pBl, color='orange',
            width=width, label=r"$\beta_l$")
    ax0.bar(x + width * 0, pBn, color='yellow',
            width=width, label=r"$\beta_n$")
    ax0.bar(x + width * 1, pTn, color='cyan', width=width, label=r"$\tau_n$")
    ax0.bar(x + width * 2, pA, color='green', width=width, label='A')
    ax0.set_xticks(x)
    ax0.set_xticklabels(name)
    ax0.set_ylabel('Peak count')
    ax0.set_xlabel('Graph')
    ax0.grid(axis='y')
    ax0.legend(ncol=5, bbox_to_anchor=(0., 1.02, 1., .102),
               loc='lower left')
    fig.tight_layout()
    if imagePath:
        fig.savefig(imagePath + '/sim-peak-count-all-'+str(time)+'.pdf')

    for peak, title in zip([peakI, peakBl, peakBn, peakTn, peakA], ['I', 'Bl', 'Bn', 'Tn', 'A']):
        fig, ax0 = mpl.subplots(figsize=(6, 2))
        for p, n in zip(peak, name):
            ax0.plot(p[0], p[1]['prominences'], label=n)
        ax0.legend(ncol=len(simulators)/2, bbox_to_anchor=(0., 1.02, 1., .102),
                   loc='lower left')
        ax0.set_ylabel('Promin-'+title)
        ax0.set_xlabel('Time')
        fig.tight_layout()
        if imagePath:
            fig.savefig(imagePath + '/sim-peak-trend-' +
                        title+'-'+str(time)+'.pdf')


# %%
def plotDegree():
    degreeMax = 30
    name = [sim.shortName for sim in simulators]
    x = np.arange(len(name))

    fig, ax = mpl.subplots(figsize=(4, 2))
    degrees = [[G.degree(n) for n in G.nodes]
               for G in [s.G for s in simulators]]
    degree_x = np.arange(0, degreeMax+1, 1)
    y, bins, _ = ax.hist(degrees, degree_x, label=name)
    ax.legend(ncol=len(simulators)/2, loc="lower center",
              bbox_to_anchor=(0.5, 1.1))
    ax.set_yscale('log')
    ax.set_ylabel('Count')
    ax.set_xlabel('Degree')
    ax.set_xlim(0, degreeMax)
    fig.tight_layout()
    if imagePath:
        fig.savefig(imagePath + '/sim-degree-hist.pdf')

    # fig, ax0 = mpl.subplots(figsize=(6, 2))
    # for i in range(len(y)):
    #     ax0.plot(y[i], label=name[i])
    # ax0.legend(ncol=len(simulators)/2, bbox_to_anchor=(0., 1.02, 1., .102),
    #            loc='lower left')
    # ax0.set_ylabel('Count')
    # ax0.set_xlabel('Degree')
    # fig.tight_layout()

    fig, axes = mpl.subplots(2, 1, figsize=(5, 3.2))
    ax = axes[0]
    for i in range(len(y)):
        ax.plot(y[i], label=name[i])
    ax.set_yscale('log')
    ax.set_ylabel('Count')
    ax.set_xlabel('Degree')
    ax.set_xlim(0, degreeMax)

    ax = axes[1]
    degreesAverage = np.average(degrees, axis=1)
    rect1 = ax.bar(x, degreesAverage, color=TABLEAU_COLORS)
    ax.set_xticks(x)
    ax.set_xticklabels(name)
    # ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_ylabel('Average Degree')
    ax.set_xlabel('Graph')
    ax.set_ylim(0, 15)
    ax.bar_label(rect1, padding=3, fmt='%2.3g')
    # fig.legend(ncol=len(simulators)/2, #bbox_to_anchor=(0.03, 1.0, 1.0, .08),
    #            loc='upper center', borderaxespad=0.)
    fig.tight_layout()
    if imagePath:
        fig.savefig(imagePath + '/sim-degree-two-'+str(time)+'.pdf')


# %%
def plotRangeHist(start, end):
    fig, axes = mpl.subplots(2, 2, figsize=(6, 3.2))
    # fig, ax0 = mpl.subplots(figsize=(6, 2))
    ax = axes[0, 0]
    for s in simulators:
        z = s.Zi[:, start:end]
        # ax0.errorbar(result_y+dx,z.mean(axis=1), z.std(axis=1),
        #  label=s.shortName, linewidth=linewidth,color=color)
        p = ax.plot(s.result_y, z.mean(axis=1),  linewidth=lineWidth)
        color = p[0].get_color()
        # ax0.plot(s.result_y, z.std(axis=1), linewidth=linewidth,
        #          color=color, linestyle=':')
        ax.plot(s.result_y, z.max(axis=1), linewidth=lineWidth,
                color=color, linestyle='--')
        ax.plot(s.result_y, z.min(axis=1), linewidth=lineWidth,
                color=color, linestyle=':')
        # ax0.fill_between(s.result_y,z.max(axis=1),z.min(axis=1),alpha=0.2)
    # ax0.set_yscale("log")
    # ax.set_xlabel("Intensity")
    ax.set_ylabel("I range dist")
    ax.set_ylim(0, np.ceil(z.max(axis=1)[1:].max()*10)/10.0)
    ax.set_xlim(0, 1)
    # ax0.legend(ncol=len(simulators)/2, bbox_to_anchor=(0., 1.02, 1., .102),
    #            loc='lower left')
    # fig.tight_layout()

    # fig, ax0 = mpl.subplots(figsize=(6, 2))
    ax = axes[0, 1]
    for s in simulators:
        z = s.Za[:, start:end]
        # ax0.errorbar(s.result_y+dx,z.mean(axis=1), z.std(axis=1), label=s.shortName, linewidth=linewidth,color=color)
        p = ax.plot(s.result_y, z.mean(axis=1), label=s.shortName,
                    linewidth=lineWidth)
        color = p[0].get_color()
        ax.plot(s.result_y, z.max(axis=1), linewidth=lineWidth,
                color=color, linestyle='--')
        ax.plot(s.result_y, z.min(axis=1), linewidth=lineWidth,
                color=color, linestyle=':')
        # ax0.fill_between(s.result_y,z.max(axis=1),z.min(axis=1),alpha=0.2)
    # ax.set_xlabel("Intensity")
    ax.set_ylabel("A range dist")
    # ax.set_ylim(0, 0.3)
    ax.set_xlim(0, 1)

    ax = axes[1, 0]
    for s in simulators:
        z = s.Zbn[:, start:end]
        p = ax.plot(s.result_y1, z.mean(axis=1),  linewidth=lineWidth)
        color = p[0].get_color()
        ax.plot(s.result_y1, z.max(axis=1), linewidth=lineWidth,
                color=color, linestyle='--')
        ax.plot(s.result_y1, z.min(axis=1), linewidth=lineWidth,
                color=color, linestyle=':')
    ax.set_xlabel("Intensity")
    ax.set_ylabel(r"$\beta_n$ range dist")
    # ax.set_ylim(0, 0.5)
    ax.set_xlim(0, 1)

    ax = axes[1, 1]
    for s in simulators:
        z = s.Ztn[:, start:end]
        p = ax.plot(s.result_y2, z.mean(axis=1), label=s.shortName,
                    linewidth=lineWidth)
        color = p[0].get_color()
        ax.plot(s.result_y2, z.max(axis=1), linewidth=lineWidth,
                color=color, linestyle='--')
        ax.plot(s.result_y2, z.min(axis=1), linewidth=lineWidth,
                color=color, linestyle=':')
    ax.set_xlabel("Intensity")
    ax.set_ylabel(r"$\tau_n$ range dist")
    # ax.set_ylim(0, 0.3)
    ax.set_xlim(-1, 1)
    fig.tight_layout()
    if imagePath:
        fig.savefig('%s/sim-aitb-hist-diff-%d-%d.pdf' %
                    (imagePath,  start, end))


# %%
def plotMixedNodeStats():
    fig, axes = mpl.subplots(2, 2, figsize=(11, 4.5))
    labels = [s.shortName for s in simulators]
    fig.set_linewidth(lineWidth)

    ax = axes[0, 0]
    lines = []
    for s in simulators:
        lines += ax.plot(s.time_x, s.st_i, label=s.shortName,
                         linewidth=lineWidth)
    # ax.set_xlabel("Time")
    ax.set_ylabel("Avg-I")
    # ax.set_ylim(0, 0.4)
    ax.set_xlim(0, time)
    ax.legend(lines[0:5], labels[0:5], ncols=5,
              loc="lower center", bbox_to_anchor=(0.5, 1.1))

    ax = axes[0, 1]
    lines = []
    for s in simulators:
        lines += ax.plot(s.time_x, s.st_a, label=s.shortName,
                         linewidth=lineWidth)
    # ax.set_xlabel("Time")
    ax.set_ylabel("Avg-A")
    # ax.set_ylim(0, 1)
    ax.set_xlim(0, time)
    ax.legend(lines[5:10], labels[5:10], ncols=5,
              loc="lower center", bbox_to_anchor=(0.5, 1.1))
    # ax.legend(lines[0:5],labels[0:5], loc="center left",bbox_to_anchor= (1.1, 0.5))

    ax = axes[1, 0]
    lines = []
    for s in simulators:
        lines += ax.plot(s.time_x, s.st_bn, label=s.shortName,
                         linewidth=lineWidth)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Avg-$\beta_n$")
    # ax.set_ylim(0, 1)
    ax.set_xlim(0, time)

    ax = axes[1, 1]
    lines = []
    for s in simulators:
        lines += ax.plot(s.time_x, s.st_tn, label=s.shortName,
                         linewidth=lineWidth)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Avg-$\tau_n$")
    ax.set_ylim(-1, 0.5)
    ax.set_xlim(0, time)
    # ax.legend(lines[5:10],labels[5:10], loc="center left",bbox_to_anchor= (1.1, 0.5))

    # fig.legend(lines,labels,ncol=len(simulators), loc="lower center",bbox_to_anchor=(0.5, 1.))
    # fig.suptitle("Status summary of all simulation scenarios")
    fig.tight_layout()
    if imagePath:
        fig.savefig(imagePath + '/sim-avg-all-'+str(time)+'.pdf')


# %%
def plotHistDiff(start: int, end: int, graph: list):
    sim = [s for s in simulators if s.shortName in graph]
    simLen = len(sim)
    gridspec = {'width_ratios': [1]*(simLen-1)+[1.2],
                'height_ratios': [1, 1, 1, 1, 1],
                'wspace': 0.2,
                'hspace': 0.2}
    fig, axes = mpl.subplots(
        5, simLen, figsize=(12, 5.3), gridspec_kw=gridspec)
    shading = 'gouraud'  # auto, gouraud, nearest
    nbins = 25
    for i in range(simLen):
        s = sim[i]
        axTC = ax = axes[0, i]
        axNew = ax.twinx()
        ax.plot(s.time_x, s.new_case_time, 'r', linewidth=0.75)
        axNew.plot(s.time_x, s.total_case_time, 'b', linewidth=0.75)
        ax.set_xlim(start, end)
        ax.set_ylim(0, n/30)
        axNew.set_xlim(start, end)
        axNew.set_ylim(0, n)
        axNew.set_yticks(np.arange(0, n+1, n/2, dtype=int))
        labelLen = len(ax.get_xticklabels())
        empty_string_labels = ['']*labelLen
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(empty_string_labels)
        ax.set_xlabel(s.shortName)
        ax.xaxis.set_label_position('top')
        if i > 0:
            ax.set_ylim(0, n/30)
            labelLen = len(ax.get_yticklabels())
            empty_string_labels = ['']*labelLen
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(empty_string_labels)
            ax.set_ylim(0, n/30)
            pass
        else:
            ax.set_ylabel('New case')
        if i != simLen - 1:
            labelLen = len(axNew.get_yticklabels())
            empty_string_labels = ['']*labelLen
            axNew.set_yticks(axNew.get_yticks())
            axNew.set_yticklabels(empty_string_labels)
            axNew.set_ylim(0, n)
        else:
            axNew.set_ylabel('Total case')

        axZi = ax = axes[1, i]
        z = s.Zi
        cmap = mpl.get_cmap('viridis')
        levels = MaxNLocator(nbins=nbins).tick_values(0, 0.1)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        imZi = ax.pcolormesh(s.X, s.Y, z, shading=shading,
                             cmap=cmap, norm=norm)
        ax.plot(s.time_x, s.st_i, 'w', linewidth=0.5)
        ax.set_xlim(start, end)
        ax.set_ylim(0, 1)
        labelLen = len(ax.get_xticklabels())
        empty_string_labels = ['']*labelLen
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(empty_string_labels)
        # ax.set_xlabel(s.shortName)
        # ax.xaxis.set_label_position('top')
        if i > 0:
            labelLen = len(ax.get_yticklabels())
            empty_string_labels = ['']*labelLen
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(empty_string_labels)
        else:
            ax.set_ylabel('Infected')

        axZa = ax = axes[2, i]
        z = s.Za
        cmap = mpl.get_cmap('plasma')
        levels = MaxNLocator(nbins=nbins).tick_values(0, 0.2)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        imZa = ax.pcolormesh(s.X, s.Y, z, shading=shading,
                             cmap=cmap, norm=norm)
        ax.plot(s.time_x, s.st_a, 'w', linewidth=0.5)
        ax.set_xlim(start, end)
        ax.set_ylim(0, 1)
        labelLen = len(ax.get_xticklabels())
        empty_string_labels = ['']*labelLen
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(empty_string_labels)
        if i > 0:
            labelLen = len(ax.get_yticklabels())
            empty_string_labels = ['']*labelLen
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(empty_string_labels)
        else:
            ax.set_ylabel('Awareness')

        axZbn = ax = axes[3, i]
        z = s.Zbn
        cmap = mpl.get_cmap('inferno')
        levels = MaxNLocator(nbins=nbins).tick_values(0, 0.3)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        imZbn = ax.pcolormesh(s.X, s.Y1, z, shading=shading,
                              cmap=cmap, norm=norm)
        ax.plot(s.time_x, s.st_bn, 'w', linewidth=0.5)
        ax.set_xlim(start, end)
        ax.set_ylim(0, s.beta_max)
        labelLen = len(ax.get_xticklabels())
        empty_string_labels = ['']*labelLen
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(empty_string_labels)
        if i > 0:
            labelLen = len(ax.get_yticklabels())
            empty_string_labels = ['']*labelLen
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(empty_string_labels)
        else:
            ax.set_ylabel(r'$\beta_n$')

        axZtn = ax = axes[4, i]
        z = s.Ztn
        cmap = mpl.get_cmap('magma')
        levels = MaxNLocator(nbins=nbins).tick_values(0, 0.2)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        imZtn = ax.pcolormesh(
            s.X2, s.Y2, z, shading=shading, cmap=cmap, norm=norm)
        ax.plot(s.time_x, s.st_tn, 'w', linewidth=0.5)
        ax.set_xlim(start, end)
        ax.set_ylim(-1, 1)
        # ax.set_xlabel(s.shortName)
        labelLen = len(ax.get_xticklabels())
        empty_string_labels = ['']*labelLen
        # ax.set_xticks(ax.get_xticks())
        # ax.set_xticklabels(empty_string_labels)
        if i > 0:
            labelLen = len(ax.get_yticklabels())
            empty_string_labels = ['']*labelLen
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(empty_string_labels)
        else:
            ax.set_ylabel(r'$\tau_n$')
    # axes[3,1].set_xlabel("Time")

    fig.colorbar(imZi, ax=axZi, ticks=np.arange(0, 0.4, 0.1))
    fig.colorbar(imZa, ax=axZa, ticks=np.arange(0, 0.4, 0.1))
    fig.colorbar(imZbn, ax=axZbn, ticks=np.arange(0, 0.6, 0.2))
    fig.colorbar(imZtn, ax=axZtn, ticks=np.arange(0, 0.4, 0.1))
    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0.1, hspace=0.1)
    if imagePath:
        fig.savefig('%s/sim-aitb-hist-%s-diff-%d-%d.pdf' %
                    (imagePath, "+".join(graph), start, end))


# %%
def plotHistDiffContour(start: int, end: int, graph: list):
    sim = [s for s in simulators if s.shortName in graph]
    simLen = len(sim)
    gridspec = {'width_ratios': [1]*(simLen-1)+[1.2],
                'height_ratios': [1, 1, 1, 1],
                'wspace': 0.2,
                'hspace': 0.2}
    fig, axes = mpl.subplots(4, simLen, figsize=(6, 4), gridspec_kw=gridspec)
    nbins = 10
    for i in range(simLen):
        s = sim[i]
        axZi = ax = axes[0, i]
        z = s.Zi
        cmap = mpl.get_cmap('viridis')
        imZi = ax.contourf(s.X, s.Y, z, np.linspace(
            0, 0.4, 11), cmap=cmap, extend='max')
        ax.set_xlim(start, end)
        ax.set_ylim(0, 1)
        labelLen = len(ax.get_xticklabels())
        empty_string_labels = ['']*labelLen
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(empty_string_labels)
        ax.set_xlabel(s.shortName)
        ax.xaxis.set_label_position('top')
        if i > 0:
            labelLen = len(ax.get_yticklabels())
            empty_string_labels = ['']*labelLen
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(empty_string_labels)
        else:
            ax.set_ylabel('I')

        axZa = ax = axes[1, i]
        z = s.Za
        cmap = mpl.get_cmap('plasma')
        imZa = ax.contourf(s.X, s.Y, z, np.linspace(
            0, 0.4, 11), cmap=cmap, extend='max')
        ax.set_xlim(start, end)
        ax.set_ylim(0, 1)
        labelLen = len(ax.get_xticklabels())
        empty_string_labels = ['']*labelLen
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(empty_string_labels)
        if i > 0:
            labelLen = len(ax.get_yticklabels())
            empty_string_labels = ['']*labelLen
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(empty_string_labels)
        else:
            ax.set_ylabel('A')

        axZbn = ax = axes[2, i]
        z = s.Zbn
        cmap = mpl.get_cmap('inferno')
        imZbn = ax.contourf(s.X, s.Y1, z, np.linspace(
            0, 0.6, 13), cmap=cmap, extend='max')
        ax.set_xlim(start, end)
        ax.set_ylim(0, 1)
        labelLen = len(ax.get_xticklabels())
        empty_string_labels = ['']*labelLen
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(empty_string_labels)
        if i > 0:
            labelLen = len(ax.get_yticklabels())
            empty_string_labels = ['']*labelLen
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(empty_string_labels)
        else:
            ax.set_ylabel(r'$\beta_n$')

        axZtn = ax = axes[3, i]
        z = s.Ztn
        cmap = mpl.get_cmap('magma')
        imZtn = ax.contourf(s.X2, s.Y2, z, np.linspace(
            0, 0.4, 11), cmap=cmap, extend='max')
        ax.set_xlim(start, end)
        ax.set_ylim(-1, 1)
        # ax.set_xlabel(s.shortName)
        labelLen = len(ax.get_xticklabels())
        empty_string_labels = ['']*labelLen
        # ax.set_xticks(ax.get_xticks())
        # ax.set_xticklabels(empty_string_labels)
        if i > 0:
            labelLen = len(ax.get_yticklabels())
            empty_string_labels = ['']*labelLen
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(empty_string_labels)
        else:
            ax.set_ylabel(r'$\tau_n$')
    # axes[3,1].set_xlabel("Time")
    fig.colorbar(imZi, ax=axZi, ticks=np.arange(0, 0.5, 0.1))
    fig.colorbar(imZa, ax=axZa, ticks=np.arange(0, 0.5, 0.1))
    fig.colorbar(imZbn, ax=axZbn, ticks=np.arange(0, 0.7, 0.2))
    fig.colorbar(imZtn, ax=axZtn, ticks=np.arange(0, 0.5, 0.1))
    fig.tight_layout()
    # fig.subplots_adjust(wspace=0.1, hspace=0.1)
    if imagePath:
        fig.savefig('%s/sim-aitb-hist-%s-diff-c-%d-%d.pdf' %
                    (imagePath, "+".join(graph), start, end))


# %%
def plotCommunity():
    communities = []
    for s in simulators:
        communities.append(community.girvan_newman(s.G))
    allCom = []
    for C in communities:
        allCom.append([])
        for com in next(C):
            allCom[-1].append(list(com))


# %%
def main():
    initSimulator()
    doSim()
    # import cProfile
    # cProfile.run("doSim()",sort='cumulative')
    print(datetime.now() - startTime)
    # plotDisease()
    for s in simulators:
        # s.plotStatsHist()
        # s.plotInfectedHist()
        # s.plotInfectingHist()
        # s.plotCompatible()
        # s.plotTotalCase()
        pass
    plotTotalInfected()
    # plotNewInfected()
    # plotTotalRemoved()
    # plotTotalRecovered()
    plotAverageInfected()
    plotAverageBetaNode()
    # plotAverageTauNode()
    # plotAverageAwareness()
    # plotPeakStat()
    # plotDegree()

    # plotTotalInfectedAt(int(time/2))
    # plotTotalInfectedAt(time)

    plotMixedNodeStats()
    # plotRangeHist(start, end)
    plotHistDiff(0, time, graph)
    # plotHistDiffContour(start, end, graph)

    mpl.tight_layout()
    mpl.show()


if __name__ == '__main__':
    main()
