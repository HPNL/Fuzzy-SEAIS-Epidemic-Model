import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# plot disease


def boltzmann(x, xMid, tau):
    """
    evaluate the boltzmann function with midpoint xMid and time constant tau
    over x
    """
    return 1. / (1. + np.exp(-(x-xMid)/tau))


def boltzmann_double(x, xMid1, tau1, xMid2, tau2):
    return boltzmann(x, xMid1, tau1) * (1 - boltzmann(x, xMid2, tau2))


def fuzz_boltzmann_double(x, abcd, fix=0.25):
    a, b, c, d = abcd
    return boltzmann_double(x, (a+b)/2, fix * (b-a), (c+d)/2, fix * (d-c))


def boltzmann3(x, xMid1, tau1, yMid2, xMid2, tau2, xMid3, tau3):
    return boltzmann(x, xMid1, tau1) * (yMid2 + (1-yMid2)*boltzmann(x, xMid2, tau2)) * (1 - boltzmann(x, xMid3, tau3))


def fuzz_boltzmann3(x, abcdef, yMid=0.5, fix=0.25):
    a, b, c, d, e, f = abcdef
    return boltzmann3(x, (a+b)/2, fix * (b-a), yMid, (c+d)/2, fix * (d-c), (e+f)/2, fix * (f-e))


def boltzmann_list(x, px, py, fix=0.25):
    r = py[0]
    for i in range( len(px)-1):
        dy = py[i+1]-py[i]
        if dy == 0:
            continue
        xm = (px[i]+px[i+1])/2
        dx = px[i+1]-px[i]
        # ym = (py[i]+py[i+1])/2
        r += dy * boltzmann(x, xm, fix*dx)
    return r


x_disease = np.arange(0, 12.1, 0.1)
y_disease0 = fuzz.trapmf(x_disease, [1, 2, 5, 11])
y_disease1 = fuzz_boltzmann_double(x_disease, [1, 2, 5, 11], fix=0.2)
y_disease2 = fuzz.trapmf(x_disease, [2, 5, 6, 11])
y_disease3 = fuzz_boltzmann_double(x_disease, [2, 5, 6, 11], fix=0.2)
y_disease4 = fuzz.trapmf(x_disease, [1, 2, 10, 11])
y_disease5 = fuzz_boltzmann_double(x_disease, [1, 2, 10, 11], fix=0.2)

# # next plot
# fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(9, 3))
# # fig = plt.figure()

# ax0.plot(x_disease, y_disease0, label='trap')
# ax0.plot(x_disease, y_disease1, label='bd')
# ax1.plot(x_disease, y_disease2, label='trap')
# ax1.plot(x_disease, y_disease3, label='bd')
# ax2.plot(x_disease, y_disease4, label='trap')
# ax2.plot(x_disease, y_disease5, label='bd')

# fig.tight_layout()

fig, ax0 = plt.subplots(figsize=(9, 3))

x_disease = np.arange(0, 20.1, 0.1)
xs = len(x_disease)
xsm = int(xs/2)
yd1 = fuzz.trapmf(x_disease, [2, 3, 10, 11])
yd2 = yd1/2
yd3 = fuzz.trapmf(x_disease, [9, 11, 15, 19])
yd4 = np.concatenate((yd2[0:xsm+1], yd3[xsm+1:]))
# yd5 = fuzz_boltzmann3(x_disease, [2, 3, 10, 11, 15, 19],yMid=0.5, fix=0.2)
yd5 = boltzmann_list(x_disease, [2, 3, 10, 11, 15, 19], [0, 0.5, 0.5, 1, 1, 0])
# ax0.plot(x_disease, yd1)
# ax0.plot(x_disease, yd2)
# ax0.plot(x_disease, yd3)
ax0.plot(x_disease, yd4)
ax0.plot(x_disease, yd5)


plt.tight_layout(h_pad=1)
plt.show()
