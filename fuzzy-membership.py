import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

from sais.fuzzy_model import x01, x11

nrows = 1
ncols = 1
isVertical = False
if isVertical:
    nrows = 2
else:
    ncols = 2

# Learning or forgetting membership function
x_learning = x11
forgetting_h = fuzz.trimf(x_learning, [-1, -1, -0.75])
forgetting_s = fuzz.trimf(x_learning, [-1, -0.8, -0.6])
forgetting_m = fuzz.trimf(x_learning, [-0.8, -0.6, -0.4])
forgetting_w = fuzz.trimf(x_learning, [-0.6, -0.4, -0.2])
forgetting_l = fuzz.trimf(x_learning, [-0.4, -0.2, 0])
memorization = fuzz.trimf(x_learning, [-0.2, 0, 0.2])
learning_l = fuzz.trimf(x_learning, [0, 0.2, 0.4])
learning_w = fuzz.trimf(x_learning, [0.2, 0.4, 0.6])
learning_m = fuzz.trimf(x_learning, [0.4, 0.6, 0.8])
learning_s = fuzz.trimf(x_learning, [0.6, 0.8, 1])
learning_h = fuzz.trimf(x_learning, [0.8, 1, 1])
learning_h[-1] = 1


# Visualize these universes and membership functions
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(
    6, 2.2), gridspec_kw={'width_ratios': [2, 1]})

ax0.plot(x_learning, forgetting_h, '#e00', label='FH')
ax0.plot(x_learning, forgetting_s, '#e30', label='FS')
ax0.plot(x_learning, forgetting_m, '#e60', label='FM')
ax0.plot(x_learning, forgetting_w, '#e90', label='FW')
ax0.plot(x_learning, forgetting_l, '#eb0', label='FL')
ax0.plot(x_learning, memorization, '#ee0', label='MM')
ax0.plot(x_learning, learning_l, '#be0', label='LL')
ax0.plot(x_learning, learning_w, '#9e0', label='LW')
ax0.plot(x_learning, learning_m, '#6e0', label='LM')
ax0.plot(x_learning, learning_s, '#3e0', label='LS')
ax0.plot(x_learning, learning_h, '#0e0', label='LH')
ax0.set_xlabel(r'$\tau_l$')
ax0.set_ylabel('Membership grade')
ax0.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=6, mode="expand", borderaxespad=0., handlelength=1)
# fig.tight_layout()

# Generate fuzzy membership functions for infecting
x_infecting = x01
infecting_z = fuzz.trimf(x_infecting, [0, 0, 0.2])
infecting_l = fuzz.trimf(x_infecting, [0, 0.2, 0.4])
infecting_w = fuzz.trimf(x_infecting, [0.2, 0.4, 0.6])
infecting_m = fuzz.trimf(x_infecting, [0.4, 0.6, .8])
infecting_s = fuzz.trimf(x_infecting, [0.6, 0.8, 1])
infecting_h = fuzz.trimf(x_infecting, [0.8, 1, 1])

# ax1.plot([0], [1], '#000', marker='*', label='Z')
ax1.plot(x_infecting, infecting_z, '#066', label='Z')
ax1.plot(x_infecting, infecting_l, '#0b0', label='L')
ax1.plot(x_infecting, infecting_w, '#6b0', label='W')
ax1.plot(x_infecting, infecting_m, '#bb0', label='M')
ax1.plot(x_infecting, infecting_s, '#b60', label='S')
ax1.plot(x_infecting, infecting_h, '#b00', label='H')
ax1.set_xlabel(r'$\beta_l$')
# ax1.set_ylabel('Membership grade')
# ax1.set_title('Infecting')
ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0., handlelength=1)
fig.tight_layout()

# Turn off top/right axes
for ax in (ax0, ax1,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


# fig.savefig('./img/infecting-6-learn-forget-11.pdf')

# Visualize these universes and membership functions
# fig, (ax0, ax1,) = plt.subplots(nrows, ncols, figsize=(3, 2))

fig, axs = plt.subplots(1, 3, figsize=(6, 2.2))


# Generate fuzzy membership functions for infected
x_link = x01
disconnected = fuzz.trimf(x_link, [0, 0, 1])
connected = fuzz.trimf(x_link, [0, 1, 1])
connected_d = fuzz.trimf(x_link, [0, 0, 0.5])  # equal to disconnected
connected_w = fuzz.trimf(x_link, [0, 0.5, 1])
connected_s = fuzz.trimf(x_link, [0.5, 1, 1])

ax0 = axs[0]

# ax0.plot(x_link, disconnected, 'r', label='S')
# ax0.plot(x_link, connected,    'g', label='W')

ax0.plot(x_link, connected_d, 'r', label='D')
ax0.plot(x_link, connected_w, 'y', label='W')
ax0.plot(x_link, connected_s, 'g', label='S')


ax0.set_xlabel('Link')
ax0.set_ylabel('Membership grade')


# Generate fuzzy membership functions for infected
x_infected = x01
susceptible = fuzz.trimf(x_infected, [0, 0, 1])
infected = fuzz.trimf(x_infected, [0, 1, 1])

infected_s = fuzz.trimf(x_infected, [0, 0, 0.25])
infected_e = fuzz.trimf(x_infected, [0, 0.25, 0.5])
infected_i = fuzz.trimf(x_infected, [0.25, 0.5, 0.75])
infected_h = fuzz.trimf(x_infected, [0.5, 0.75, 1])
infected_d = fuzz.trimf(x_infected, [0.75, 1, 1])


# fig, ax1 = plt.subplots(figsize=(3, 2))
ax1 = axs[1]

# ax1.plot(x_infected, susceptible, 'g', label='L')
# ax1.plot(x_infected, infected,    'r', label='H')

ax1.plot(x_infected, infected_s, 'g', label='S')
ax1.plot(x_infected, infected_e, 'y', label='E')
ax1.plot(x_infected, infected_i, 'orange', label='I')
ax1.plot(x_infected, infected_h, 'r', label='H')
ax1.plot(x_infected, infected_d, 'k', label='D')

ax1.set_xlabel('Infected')
# ax1.set_ylabel('Membership grade')

# Generate fuzzy membership functions for alerting
x_alerted = x01
unwary = fuzz.trimf(x_alerted, [0, 0, 1])
alerted = fuzz.trimf(x_alerted, [0, 1, 1])
alerted_l = fuzz.trimf(x_alerted, [0, 0, 0.5])
alerted_m = fuzz.trimf(x_alerted, [0, 0.5, 1])
alerted_h = fuzz.trimf(x_alerted, [0.5, 1, 1])
# alerting, aware, informed

# fig, ax2 = plt.subplots(figsize=(3, 2))
ax2 = axs[2]

# ax2.plot(x_alerted, unwary, 'r', label='L')
# ax2.plot(x_alerted, alerted, 'g', label='H')

ax2.plot(x_alerted, alerted_l, 'r', label='L')
ax2.plot(x_alerted, alerted_m, 'y', label='M')
ax2.plot(x_alerted, alerted_h, 'g', label='H')

ax2.set_xlabel('Awareness')
# ax2.set_ylabel('Membership grade')
fig.tight_layout()

# Turn off top/right axes
for ax in (ax0, ax1, ax2,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
              ncol=3, mode="expand", borderaxespad=0., handlelength=1)

fig.tight_layout()
# plt.show()

# fig.savefig('./img/link3-infected5-awareness3.pdf')

plt.tight_layout(h_pad=1)
plt.show()
