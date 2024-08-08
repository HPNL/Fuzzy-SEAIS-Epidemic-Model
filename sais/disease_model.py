from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm

from .fuzzy_model import boltzmann_list
from pert import PERT

xMin = 0
xMax = 200
x = np.arange(xMin, xMax + 1)
yMax = boltzmann_list(
    x,
    [0, 5, 10, 15, 20, 30, 45, 60],
    [0, 0.6, 0.9, 1, 1, 0.65, 0.15, 0],
)
yMin = boltzmann_list(x, [0, 5, 10, 15, 20], [0, 0.1, 0.15, 0.1, 0])
yMean = (yMax + yMin) / 2
yRange = yMax-yMin


def generatePertList(x, yMax, yMean, yMin) -> List[PERT]:
    pertList: List[PERT] = []
    # ySum = np.zeros(len(x))
    for i in x:
        if np.any(np.multiply(np.isclose(yMax[i], yMean[i]),np.isclose(yMean[i], yMin[i]))):
            continue
        pert = PERT(yMin[i], yMean[i], yMax[i])
        pertList.append(pert)
        # ySum[i] = np.sum(pdf)
    return pertList


pertList = generatePertList(x, yMax, yMean, yMin)


def generateDisease(pertList: List[PERT], delay=(1, 5), recovered=(200, 300)):
    # MCMC algorithm
    disease = np.zeros(len(pertList))
    for t in range(1, len(pertList)):
        if t >= len(pertList):
            continue
        pert = pertList[t]
        oldPert = pertList[t-1]
        increasing = pert.b > oldPert.b
        oldD = disease[t-1]
        if t > 5 and pert.a < 0.01 and oldD < 0.05:
            # fully recovered
            continue

        def acceptanceFunction(oldPert: PERT, pert: PERT, old, new, increasing) -> bool:
            oldP = pert.pdf(old)
            if oldP == 0:
                oldP = 0.1
            pdfRatio = pert.pdf(new)/oldP
            oldC, newC = pert.cdf(
                [old, new]) if increasing else pert.sf([old, new])
            if oldC == 0:
                oldC = 0.1
            finalRatio = pdfRatio * newC/oldC
            if finalRatio > 1:
                return True
            return np.random.rand() < finalRatio

        def simpleAcceptanceFunction(pert: PERT, old, new, diffRate=0.2) -> bool:
            diffMax = pert.range * diffRate
            return np.abs(new-old) <= diffMax

        newD = pert.rvs()[0]
        maxTry = 2
        while maxTry > 0 and not acceptanceFunction(oldPert, pert, oldD, newD, increasing):
            newD = pert.rvs()[0]
            maxTry -= 1
        if maxTry == 0:
            newD = oldD
        disease[t] = newD
    return np.concatenate((np.zeros(np.random.randint(delay[0], delay[1])) , disease , np.zeros(np.random.randint(recovered[0], recovered[1]))))


def plotDisease(x, disease, imagePath: str | None = None):
    y = np.linspace(0, 1, 41)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(y), len(x)))

    for i in range(len(pertList)):
        pert = pertList[i]
        pdf = pert.pdf(y)
        Z[:, i] = pdf

    fig, ax = plt.subplots(figsize=(5, 3.2))

    # shading='gouraud'
    pcm = ax.pcolormesh(X, Y, Z, norm=colors.PowerNorm(
        0.25), shading="auto", cmap="Blues")
    fig.colorbar(pcm, ax=ax)
    ax.plot(x, disease, 'k', linewidth=1, label="Disease")
    # ax.plot(x, yMean, linewidth=1, label="Mean")

    ax.hlines(0.0, xmin=xMin, xmax=xMax, colors="g",
              linewidth=1, label="S", ls=":")
    ax.hlines(0.25, xmin=xMin, xmax=xMax, colors="y",
              linewidth=1, label="E", ls=":")
    ax.hlines(0.5, xmin=xMin, xmax=xMax, colors="orange",
              linewidth=1, label="I", ls=":")
    ax.hlines(0.75, xmin=xMin, xmax=xMax, colors="r",
              linewidth=1, label="H", ls=":")
    ax.hlines(1.0, xmin=xMin, xmax=xMax, colors="k",
              linewidth=1, label="D", ls=":")

    ax.vlines(0, ymin=0, ymax=1, colors="r", linewidth=1, label=r"$t_I$")
    ax.vlines(70, ymin=0, ymax=1, colors="g", linewidth=1, label=r"$t_R$")
    ax.vlines(100, ymin=0, ymax=1, colors="orange",
              linewidth=1, label=r"$t_S$")

    ax.set_xlabel("Time")
    ax.set_ylabel("Severity distribution")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=5)

    fig.tight_layout()

    if imagePath:
        fig.savefig(imagePath+'/sim-disease-boundary-with-disease.pdf')
