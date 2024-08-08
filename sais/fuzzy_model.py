from typing import List, Tuple

import atexit
from skfuzzy import control as ctrl
import numpy as np
import pickle


def getLevels(count):
    return [i for i in range(count)]


x01 = np.arange(0, 1.05, 0.05)
x11 = np.arange(-1, 1.05, 0.05)


# for 2 layer input parameter

inputNames2 = ['L', 'H']
inputNames3 = ['L', 'M', 'H']
inputNames5 = ['L', 'W', 'M', 'S', 'H']

alertedName3 = ['L', 'M', 'H']
infectingNames6 = ['Z', 'L', 'W', 'M', 'S', 'H']

linkName2 = ['D', 'S']
linkName3 = ['D', 'W', 'S']

infectedName2 = ['L', 'H']
infectedName3 = ['L', 'M', 'H']
infectedName4 = ['S', 'E', 'I', 'H']
infectedName5 = ['S', 'E', 'I', 'H', 'D']


learningName5 = ['FH', 'FL', 'MM', 'LL', 'LH']
learningName7 = ['FH', 'FM', 'FL', 'MM', 'LL', 'LM', 'LH']
learningNames9 = ['FH', 'FS', 'FM', 'FW', 'MM', 'LW', 'LM', 'LS', 'LH']
learningNames11 = ['FH', 'FS', 'FM', 'FW',
                   'FL', 'MM', 'LL', 'LW', 'LM', 'LS', 'LH']


linkMember2 = ctrl.Antecedent(x01, 'Link')
linkMember2.automf(2, names=inputNames2)

linkMember3 = ctrl.Antecedent(x01, 'Link')
linkMember3.automf(3, names=linkName3)

infected1Member2 = ctrl.Antecedent(x01, 'Infected1')
infected1Member2.automf(2, names=inputNames2)

infected2Member2 = ctrl.Antecedent(x01, 'Infected2')
infected2Member2.automf(2, names=inputNames2)

infected1Member3 = ctrl.Antecedent(x01, 'Infected1')
infected1Member3.automf(3, names=infectedName3)

infected2Member3 = ctrl.Antecedent(x01, 'Infected2')
infected2Member3.automf(3, names=infectedName3)

infected1Member4 = ctrl.Antecedent(x01, 'Infected1')
infected1Member4.automf(4, names=infectedName4)

infected2Member4 = ctrl.Antecedent(x01, 'Infected2')
infected2Member4.automf(4, names=infectedName4)

infected1Member5 = ctrl.Antecedent(x01, 'Infected1')
infected1Member5.automf(5, names=infectedName5)

infected2Member5 = ctrl.Antecedent(x01, 'Infected2')
infected2Member5.automf(5, names=infectedName5)

alerted1Member2 = ctrl.Antecedent(x01, 'Alerted1')
alerted1Member2.automf(2, names=inputNames2)

alerted2Member2 = ctrl.Antecedent(x01, 'Alerted2')
alerted2Member2.automf(2, names=inputNames2)

alerted1Member3 = ctrl.Antecedent(x01, 'Alerted1')
alerted1Member3.automf(3, names=alertedName3)

alerted2Member3 = ctrl.Antecedent(x01, 'Alerted2')
alerted2Member3.automf(3, names=alertedName3)


# lom/som/mom/centroid/bisector
DE_FUZZ_METHOD = 'centroid'

# Learning or forgetting membership function
learningMember5 = ctrl.Consequent(x11, 'Learning', DE_FUZZ_METHOD)
learningMember5.automf(5, names=learningName5)

learningMember7 = ctrl.Consequent(x11, 'Learning', DE_FUZZ_METHOD)
learningMember7.automf(7, names=learningName7)

learningMember9 = ctrl.Consequent(x11, 'Learning', DE_FUZZ_METHOD)
learningMember9.automf(9, names=learningNames9)

learningMember11 = ctrl.Consequent(x11, 'Learning', DE_FUZZ_METHOD)
learningMember11.automf(11, names=learningNames11)

# Generate fuzzy membership functions for infecting
infectingMember3 = ctrl.Consequent(x01, 'Infecting', DE_FUZZ_METHOD)
infectingMember3.automf(3, names=inputNames3)

infectingMember5 = ctrl.Consequent(x01, 'Infecting', DE_FUZZ_METHOD)
infectingMember5.automf(5, names=inputNames5)

infectingMember6 = ctrl.Consequent(x01, 'Infecting', DE_FUZZ_METHOD)
infectingMember6.automf(6, names=infectingNames6)


def generateSAISFuzzyModel_222_35():
    # Link Infecting Rules
    l = linkMember2
    i1, i2 = infected1Member2, infected2Member2
    a1, a2 = alerted1Member2, alerted2Member2

    ruleInfectingL = ctrl.Rule(
        l['L'] |
        (
            l['H'] &
            (
                (i1['L'] & i2['L']) |
                (i1['H'] & i2['H']) |
                (i1['H'] & i2['L'] & a1['H'] & a2['H']) |
                (i1['L'] & i2['H'] & a1['H'] & a2['H'])
            )
        ),
        infectingMember3['L'])
    ruleInfectingM = ctrl.Rule(
        l['H'] &
        (
            (i1['H'] & i2['L'] & a1['H'] & a2['L']) |
            (i1['H'] & i2['L'] & a1['L'] & a2['H']) |
            (i1['L'] & i2['H'] & a1['H'] & a2['L']) |
            (i1['L'] & i2['H'] & a1['L'] & a2['H'])
        ),
        infectingMember3['M'])
    ruleInfectingH = ctrl.Rule(
        l['H'] & a1['L'] & a2['L'] &
        (
            (i1['H'] & i2['L']) |
            (i1['L'] & i2['H'])
        ),
        infectingMember3['H'])

    ruleLearningFH = ctrl.Rule(
        l['L'] |
        (
            l['H'] &
            (
                (a1['L'] & a2['L'] & i1['L'] & i2['L']) |
                (a1['H'] & a2['L'] & i1['L'] & i2['L']) |
                (a1['L'] & a2['H'] & i1['L'] & i2['L'])
            )
        ),
        learningMember5['FH'])
    ruleLearningFL = ctrl.Rule(
        l['H'] & (a1['H'] & a2['H'] & i1['L'] & i2['L']),
        learningMember5['FL'])
    ruleLearningM = ctrl.Rule(
        l['H'] &
        (
            a1['H'] & a2['H'] &
            (
                (i1['H'] & i2['L']) |
                (i1['L'] & i2['H'])
            )
        ),
        learningMember5['MM'])
    ruleLearningLL = ctrl.Rule(
        l['H'] &
        (
            (a1['L'] & a2['L'] & (~(i1['L'] & i2['L']))) |
            (a1['L'] & a2['H'] & i1['L'] & i2['H']) |
            (a1['H'] & a2['L'] & i1['H'] & i2['L']) |
            (a1['H'] & a2['H'] & i1['H'] & i2['H'])
        ),
        learningMember5['LL'])
    ruleLearningLH = ctrl.Rule(
        l['H'] &
        (
            (a1['L'] & a2['H'] & i1['H']) |
            (a1['H'] & a2['L'] & i2['H'])
        ),
        learningMember5['LH'])

    saisFuzzyCtrl = ctrl.ControlSystem([ruleInfectingL, ruleInfectingM,
                                        ruleInfectingH, ruleLearningFH,
                                        ruleLearningFL, ruleLearningM,
                                        ruleLearningLL, ruleLearningLH])
    return ctrl.ControlSystemSimulation(saisFuzzyCtrl)


def generateSAISFuzzyModel_333_57():
    # Link Infecting Rules
    l = linkMember3
    i1, i2 = infected1Member3, infected2Member3
    a1, a2 = alerted1Member3, alerted2Member3

    rules = []
    rules.append(ctrl.Rule(
        l['D'] |
        (i1['L'] & i2['L']) |
        (i1['M'] & i2['M']) |
        (i1['H'] & i2['H']) |
        (i1['H'] & i2['M']) |
        (i1['M'] & i2['H']) |
        (l['W'] & (
            (i1['L'] & i2['M'] & ((a1['L'] & a2['H']) |
                                  (a1['M'] & a2['M']) |
                                  (a1['M'] & a2['H']) |
                                  (a1['H'] & a2['H']))) |
            (i2['L'] & i1['M'] & ((a2['L'] & a1['H']) |
                                  (a2['M'] & a1['M']) |
                                  (a2['M'] & a1['H']) |
                                  (a2['H'] & a1['H']))) |
            (i1['L'] & i2['H'] & ((a1['M'] & a2['H']) |
                                  (a1['H'] & a2['H']))) |
            (i2['L'] & i1['H'] & ((a2['M'] & a1['H']) |
                                  (a2['H'] & a1['H'])))
        )) |
        (l['S'] & (
            (i1['L'] & i2['M'] & ((a1['M'] & a2['H']) |
                                  (a1['H'] & a2['H']))) |
            (i2['L'] & i1['M'] & ((a2['M'] & a1['H']) |
                                  (a2['H'] & a1['H']))) |
            (i2['L'] & i1['H'] & a2['H'] & a1['H'])
        )),
        infectingMember5['L']))
    rules.append(ctrl.Rule(
        (l['W'] & (
            (i1['L'] & i2['M'] & a1['L'] & a2['M']) |
            (i1['L'] & i2['H'] & ((a1['L'] & a2['H']) |
                                  (a1['M'] & a2['M']))) |
            (i2['L'] & i1['M'] & a2['L'] & a1['M']) |
            (i2['L'] & i1['H'] & ((a2['L'] & a1['H']) |
                                  (a2['M'] & a1['M'])))
        )) |
        (l['S'] & (
            (i1['L'] & i2['M'] & ((a1['L'] & a2['H']) |
                                  (a1['M'] & a2['M']))) |
            (i1['L'] & i2['H'] & a1['M'] & a2['H']) |
            (i2['L'] & i1['M'] & ((a2['L'] & a1['H']) |
                                  (a2['M'] & a1['M']))) |
            (i2['L'] & i1['H'] & a2['M'] & a1['H'])
        )),
        infectingMember5['W']))
    rules.append(ctrl.Rule(
        (l['W'] & (
            (((i1['M'] & i2['L']) |
              (i1['L'] & i2['M'])) &
                a1['L'] & a2['L']) |
            (i1['L'] & i2['H'] & a1['L'] & a2['M']) |
            (i2['L'] & i1['H'] & a2['L'] & a1['M'])
        )) |
        (l['S'] & (
            (i1['L'] & i2['M'] & a1['L'] & a2['M']) |
            ((i1['L'] & i2['H']) & ((a1['L'] & a2['H']) |
                                    (a1['M'] & a2['M']))) |
            (i2['L'] & i1['M'] & a2['L'] & a1['M']) |
            ((i2['L'] & i1['H']) & ((a2['L'] & a1['H']) |
                                    (a2['M'] & a1['M'])))
        )),
        infectingMember5['M']))
    rules.append(ctrl.Rule(
        (l['W'] & (
            (((i1['H'] & i2['L']) |
              (i1['L'] & i2['H'])) &
                a1['L'] & a2['L'])
        )) |
        (l['S'] & (
            (((i1['M'] & i2['L']) |
              (i1['L'] & i2['M'])) &
                a1['L'] & a2['L']) |
            (i1['L'] & i2['H'] & a1['L'] & a2['M']) |
            (i2['L'] & i1['H'] & a2['L'] & a1['M'])
        )),
        infectingMember5['S']))
    rules.append(ctrl.Rule(
        (l['S'] & (
            (((i1['H'] & i2['L']) |
              (i1['L'] & i2['H'])) &
                a1['L'] & a2['L'])
        )),
        infectingMember5['H']))

    rules.append(ctrl.Rule(
        l['D'] |
        (l['W'] & (
            # (l['M'] | l['H']) &
            ((a1['L'] | a2['L']) & i1['L'] & i2['L'])
        ) |
            (l['S'] & (
                a1['L']
            ))
        ),
        learningMember7['FH']))
    rules.append(ctrl.Rule(
        l['W'] & (
            (a1['M'] & a2['M'] & i1['L'] & i2['L'])
        ),
        learningMember7['FM']))
    rules.append(ctrl.Rule(
        l['W'] & (
            (a1['M'] & a2['M'] & i1['L'] & i2['L'])
        ),
        learningMember7['FL']))
    rules.append(ctrl.Rule(
        l['W'] & (
            a1['M'] & a2['M'] &
            ((i1['M'] & i2['L']) | (i1['L'] & i2['M']))
        ),
        learningMember7['MM']))
    rules.append(ctrl.Rule(
        l['W'] & (
            (a1['L'] & a2['L'] & (~(i1['L'] & i2['L']))) |
            (a1['L'] & a2['M'] & i1['L'] & i2['M']) |
            (a1['M'] & a2['L'] & i1['M'] & i2['L']) |
            (a1['M'] & a2['M'] & i1['M'] & i2['M'])
        ),
        learningMember7['LL']))
    rules.append(ctrl.Rule(
        l['W'] & (
            (a1['L'] & a2['M'] & i1['M']) |
            (a1['M'] & a2['L'] & i2['M'])
        ),
        learningMember7['LM']))
    rules.append(ctrl.Rule(
        l['W'] & (
            (a1['L'] & a2['M'] & i1['M']) |
            (a1['M'] & a2['L'] & i2['M'])
        ),
        learningMember7['LH']))

    saisFuzzyCtrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(saisFuzzyCtrl)


def generateSAISFuzzyModel_333_611():
    # Link Infecting Rules
    l = linkMember3
    i1, i2 = infected1Member3, infected2Member3
    a1, a2 = alerted1Member3, alerted2Member3

    rules = []
    rules.append(ctrl.Rule(
        i1['M'] | i1['H'] | l['D'] | i2['L'],
        infectingMember6['Z']))
    rules.append(ctrl.Rule(
        i1['L'] & (
            (l['S'] & a1['H'] & i2['M'] & a2['H']) |
            (l['W'] & (
                (i2['M'] & (a1['H'] | a2['H'] |
                            (a1['M'] & a2['M']))) |
                (i2['H'] & ((a1['H'] & a2['M']) |
                            (a1['M'] & a2['H']) |
                            (a1['H'] & a2['H'])
                            ))
            ))
        ),
        infectingMember6['Z']))
    rules.append(ctrl.Rule(
        i1['L'] & (
            (l['S'] & (
                (i2['M'] & a1['H'] & a2['M']) |
                (i2['M'] & a1['M'] & a2['H']) |
                (i2['H'] & a1['H'] & a2['H'])
            )) |
            (l['W'] & (
                (i2['M'] & a1['L'] & a2['M']) |
                (i2['M'] & a1['M'] & a2['L']) |
                (i2['H'] & a1['L'] & a2['H']) |
                (i2['H'] & a1['M'] & a2['M']) |
                (i2['H'] & a1['H'] & a2['L'])
            ))
        ),
        infectingMember6['L']))
    rules.append(ctrl.Rule(
        i1['L'] & (
            (l['S'] & (
                (i2['M'] & a1['L'] & a2['H']) |
                (i2['M'] & a1['M'] & a2['M']) |
                (i2['M'] & a1['H'] & a2['L']) |
                (i2['H'] & a1['H'] & a2['M']) |
                (i2['H'] & a1['M'] & a2['H'])
            )) |
            (l['W'] & (
                (i2['M'] & a1['L'] & a2['L']) |
                (i2['H'] & a1['L'] & a2['M']) |
                (i2['H'] & a1['M'] & a2['L'])
            ))
        ),
        infectingMember6['W']))
    rules.append(ctrl.Rule(
        i1['L'] & (
            (l['S'] & (
                (i2['M'] & a1['L'] & a2['M']) |
                (i2['M'] & a1['M'] & a2['L']) |
                (i2['H'] & a1['L'] & a2['H']) |
                (i2['H'] & a1['M'] & a2['M']) |
                (i2['H'] & a1['H'] & a2['L'])
            )) |
            (l['W'] & i2['H'] & a1['L'] & a2['L'])
        ),
        infectingMember6['M']))
    rules.append(ctrl.Rule(
        i1['L'] & (
            l['S'] & (
                (i2['M'] & a1['L'] & a2['L']) |
                (i2['H'] & a1['L'] & a2['M']) |
                (i2['H'] & a1['M'] & a2['L'])
            )
        ),
        infectingMember6['S']))
    rules.append(ctrl.Rule(
        i1['L'] & l['S'] & i2['H'] & a1['L'] & a2['L'],
        infectingMember6['H']))

    rules.append(ctrl.Rule(
        l['D'] | l['W'],
        learningMember11['FH']))
    rules.append(ctrl.Rule(
        l['S'] & i1['L'] & i2['L'] & (
            (a1['L'] & a2['L']) |
            (a1['L'] & a2['M']) |
            (a1['M'] & a2['L'])
        ),
        learningMember11['FH']))
    rules.append(ctrl.Rule(
        l['S'] & i1['L'] & i2['L'] & (
            (a1['L'] & a2['H']) |
            (a1['H'] & a2['M']) |
            (a1['H'] & a2['L'])
        ),
        learningMember11['FS']))
    rules.append(ctrl.Rule(
        l['S'] & i1['L'] & (
            (i2['L'] & a1['M'] & a2['H']) |
            (i2['L'] & a1['H'] & a2['M']) |
            (i2['M'] & a1['L'] & a2['L'])
        ),
        learningMember11['FM']))
    rules.append(ctrl.Rule(
        l['S'] & (
            (i1['L'] & i2['L'] & (a1['H'] & a2['H'])) |
            (i1['L'] & i2['M'] & ((a1['M'] & a2['L']) | (a1['L'] & a2['M']))) |
            (i1['M'] & i2['L'] & (a1['L'] & a2['L']))
        ),
        learningMember11['FW']))
    rules.append(ctrl.Rule(
        l['S'] & (
            (i1['L'] & i2['M'] & ((a1['H'] & a2['L']) | (a1['M'] & a2['M']) | (a1['L'] & a2['H']))) |
            (i1['M'] & i2['L'] & ((a1['M'] & a2['L']) | (a1['L'] & a2['M']))) |
            (i1['M'] & i2['M'] & (a1['L'] & a2['L']))
        ),
        learningMember11['FL']))

    rules.append(ctrl.Rule(
        l['S'] & (
            (i1['L'] & i2['M'] & ((a1['H'] & a2['M']) | (a1['M'] & a2['H']))) |
            (i1['L'] & i2['H'] & (a1['L'] & a2['L'])) |
            (i1['M'] & i2['L'] & ((a1['H'] & a2['L']) | (a1['M'] & a2['M']) | (a1['L'] & a2['H']))) |
            (i1['M'] & i2['M'] & ((a1['M'] & a2['L']) | (a1['L'] & a2['M'])))
        ),
        learningMember11['MM']))

    rules.append(ctrl.Rule(
        l['S'] & (
            (i1['L'] & i2['M'] & (a1['H'] & a2['H'])) |
            (i1['L'] & i2['H'] & ((a1['L'] & a2['M']) | (a1['M'] & a2['L']))) |
            (i1['M'] & i2['L'] & ((a1['H'] & a2['M']) | (a1['M'] & a2['H']))) |
            (i1['M'] & i2['M'] & ((a1['H'] & a2['L']) | (a1['M'] & a2['M']) | (a1['L'] & a2['H']))) |
            (i1['M'] & i2['H'] & (a1['L'] & a2['L'])) |
            (i1['H'] & i2['L'] & (a1['L'] & a2['L'] | (a1['L'] & a2['M']) | (a1['M'] & a2['L']))) |
            (i1['H'] & i2['M'] & (a1['L'] & a2['L']))
        ),
        learningMember11['LL']))
    rules.append(ctrl.Rule(
        l['S'] & (
            (i1['L'] & i2['H'] & ((a1['H'] & a2['L']) | (a1['M'] & a2['M']) | (a1['L'] & a2['H']))) |
            (i1['M'] & i2['L'] & (a1['H'] & a2['H'])) |
            (i1['M'] & i2['M'] & ((a1['H'] & a2['M']) | (a1['M'] & a2['H']))) |
            (i1['M'] & i2['H'] & ((a1['L'] & a2['M']) | (a1['M'] & a2['L']) | (a1['M'] & a2['M']))) |
            (i1['H'] & i2['L'] & ((a1['H'] & a2['L']) | (a1['M'] & a2['M']) | (a1['L'] & a2['H']))) |
            (i1['H'] & i2['M'] & (a1['M'] & a2['L'])) |
            (i1['H'] & i2['H'] & (a1['H'] & a2['L']))
        ),
        learningMember11['LW']))
    rules.append(ctrl.Rule(
        l['S'] & (
            (i1['L'] & i2['H'] & ((a1['H'] & a2['M']) | (a1['M'] & a2['H']))) |
            (i1['M'] & i2['M'] & (a1['H'] & a2['H'])) |
            (i1['M'] & i2['H'] & ((a1['L'] & a2['H']) | (a1['M'] & a2['H']) | (a1['H'] & a2['L']) | (a1['H'] & a2['M']))) |
            (i1['H'] & i2['L'] & ((a1['H'] & a2['M']) | (a1['M'] & a2['H']))) |
            (i1['H'] & i2['M'] & ((a1['H'] & a2['L']) | (a1['M'] & a2['M']) | (a1['H'] & a2['H']))) |
            (i1['H'] & i2['H'] & ((a1['H'] & a2['M']) | (a1['M'] & a2['H'])))
        ),
        learningMember11['LM']))
    rules.append(ctrl.Rule(
        l['S'] & (
            (i1['L'] & i2['H'] & (a1['H'] & a2['H'])) |
            (i1['M'] & i2['H'] & (a1['H'] & a2['H'])) |
            (i1['H'] & i2['L'] & (a1['H'] & a2['H'])) |
            (i1['M'] & i2['H'] & ((a1['L'] & a2['M']) | (a1['M'] & a2['H']) | (a1['H'] & a2['M']) | (a1['H'] & a2['M']))) |
            (i1['H'] & i2['H'] & ((a1['L'] & a2['L']) |
             (a1['M'] & a2['M']) | (a1['H'] & a2['H'])))
        ),
        learningMember11['LS']))
    rules.append(ctrl.Rule(
        l['S'] & (
            (i1['H'] & i2['M'] & (a1['L'] & a2['H'])) |
            (i1['H'] & i2['H'] & ((a1['L'] & a2['M']) |
             (a1['L'] & a2['H']) | (a1['M'] & a2['H'])))
        ),
        learningMember11['LH']))

    saisFuzzyCtrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(saisFuzzyCtrl)


def SAISFuzzyInfectingRules_353_611()->List[ctrl.Rule]:
    # Link Infecting Rules
    # ['D', 'W', 'S']
    l = linkMember3
    # ['S', 'E', 'I', 'H', 'D']
    Ii, Ij = infected1Member5, infected2Member5
    # ['L', 'M', 'H']
    Ai, Aj = alerted1Member3, alerted2Member3
    # ['Z', 'L', 'W', 'M', 'S', 'H']
    # ['FH', 'FS', 'FM', 'FW', 'FL', 'MM', 'LL', 'LW', 'LM', 'LS', 'LH']

    rules:List[ctrl.Rule] = []
    rules.append(ctrl.Rule(
        l['D'] |
        Ij['S'] |
        Ii['E'] | Ii['I'] | Ii['H'] | Ii['D'],
        infectingMember6['Z']))
    rules.append(ctrl.Rule(
        Ii['S'] & (
            (l['S'] & Ai['H'] & (Ij['E'] | Ij['I'] | Ij['D']) & Aj['H']) |
            (l['W'] & (
                ((Ij['E'] | Ij['I']) & (Ai['H'] | Aj['H'] |
                                        (Ai['M'] & Aj['M']))) |
                (Ij['H'] & ((Ai['H'] & Aj['M']) |
                            (Ai['M'] & Aj['H']) |
                            (Ai['H'] & Aj['H'])
                            )) |
                (Ij['D'])
            ))
        ),
        infectingMember6['Z']))
    rules.append(ctrl.Rule(
        Ii['S'] & (
            (l['S'] & (
                ((Ij['E'] | Ij['I']) & Ai['H'] & Aj['M']) |
                ((Ij['E'] | Ij['I']) & Ai['M'] & Aj['H']) |
                (Ij['D'] & Ai['M']) |
                (Ij['H'] & Ai['H'] & Aj['H'])
            )) |
            (l['W'] & (
                ((Ij['E'] | Ij['I']) & Ai['L'] & Aj['M']) |
                ((Ij['E'] | Ij['I']) & Ai['M'] & Aj['L']) |
                (Ij['H'] & Ai['L'] & Aj['H']) |
                (Ij['H'] & Ai['M'] & Aj['M']) |
                (Ij['H'] & Ai['H'] & Aj['L'])
            ))
        ),
        infectingMember6['L']))
    rules.append(ctrl.Rule(
        Ii['S'] & (
            (l['S'] & (
                ((Ij['E'] | Ij['I']) & Ai['L'] & Aj['H']) |
                ((Ij['E'] | Ij['I']) & Ai['M'] & Aj['M']) |
                ((Ij['E'] | Ij['I']) & Ai['H'] & Aj['L']) |
                (Ij['H'] & Ai['H'] & Aj['M']) |
                (Ij['H'] & Ai['M'] & Aj['H']) |
                (Ij['D'] & Ai['L'])
            )) |
            (l['W'] & (
                ((Ij['E'] | Ij['I']) & Ai['L'] & Aj['L']) |
                (Ij['H'] & Ai['L'] & Aj['M']) |
                (Ij['H'] & Ai['M'] & Aj['L'])
            ))
        ),
        infectingMember6['W']))
    rules.append(ctrl.Rule(
        Ii['S'] & (
            (l['S'] & (
                ((Ij['E'] | Ij['I']) & Ai['L'] & Aj['M']) |
                ((Ij['E'] | Ij['I']) & Ai['M'] & Aj['L']) |
                (Ij['H'] & Ai['L'] & Aj['H']) |
                (Ij['H'] & Ai['M'] & Aj['M']) |
                (Ij['H'] & Ai['H'] & Aj['L'])
            )) |
            (l['W'] & Ij['H'] & Ai['L'] & Aj['L'])
        ),
        infectingMember6['M']))
    rules.append(ctrl.Rule(
        Ii['S'] & (
            l['S'] & (
                ((Ij['E'] | Ij['I']) & Ai['L'] & Aj['L']) |
                (Ij['H'] & Ai['L'] & Aj['M']) |
                (Ij['H'] & Ai['M'] & Aj['L'])
            )
        ),
        infectingMember6['S']))
    rules.append(ctrl.Rule(
        Ii['S'] & l['S'] & Ij['H'] & Ai['L'] & Aj['L'],
        infectingMember6['H']))
    return rules


def SAISFuzzyLearningRules_353_611()->List[ctrl.Rule]:
    # ['D', 'W', 'S']
    l = linkMember3
    # ['S', 'E', 'I', 'H', 'D']
    Ii, Ij = infected1Member5, infected2Member5
    # ['L', 'M', 'H']
    Ai, Aj = alerted1Member3, alerted2Member3
    # ['Z', 'L', 'W', 'M', 'S', 'H']
    # ['FH', 'FS', 'FM', 'FW', 'FL', 'MM', 'LL', 'LW', 'LM', 'LS', 'LH']
    rules:List[ctrl.Rule] = []
    rules.append(ctrl.Rule(
        l['D'] | l['W'],
        learningMember11['FH']))
    rules.append(ctrl.Rule(
        l['S'] & (Ii['S'] | Ii['E']) & (Ij['S'] | Ij['E']) & (
            (Ai['L'] & Aj['L']) |
            (Ai['L'] & Aj['M']) |
            (Ai['M'] & Aj['L'])
        ),
        learningMember11['FH']))
    rules.append(ctrl.Rule(
        l['S'] & (Ii['S'] | Ii['E']) & (Ij['S'] | Ij['E']) & (
            (Ai['L'] & Aj['H']) |
            (Ai['H'] & Aj['M']) |
            (Ai['H'] & Aj['L'])
        ),
        learningMember11['FS']))
    rules.append(ctrl.Rule(
        l['S'] & (Ii['S'] | Ii['E']) & (
            ((Ij['S'] | Ij['E']) & Ai['M'] & Aj['H']) |
            ((Ij['S'] | Ij['E']) & Ai['H'] & Aj['M']) |
            (Ij['I'] & Ai['L'] & Aj['L'])
        ),
        learningMember11['FM']))
    rules.append(ctrl.Rule(
        l['S'] & (
            ((Ii['S'] | Ii['E']) & (Ij['S'] | Ij['E']) & (Ai['H'] & Aj['H'])) |
            ((Ii['S'] | Ii['E']) & Ij['I'] & ((Ai['M'] & Aj['L']) | (Ai['L'] & Aj['M']))) |
            (Ii['I'] & (Ij['S'] | Ij['E']) & (Ai['L'] & Aj['L']))
        ),
        learningMember11['FW']))
    rules.append(ctrl.Rule(
        l['S'] & (
            ((Ii['S'] | Ii['E']) & Ij['I'] & ((Ai['H'] & Aj['L']) | (Ai['M'] & Aj['M']) | (Ai['L'] & Aj['H']))) |
            (Ii['I'] & (Ij['S'] | Ij['E']) & ((Ai['M'] & Aj['L']) | (Ai['L'] & Aj['M']))) |
            (Ii['I'] & Ij['I'] & (Ai['L'] & Aj['L']))
        ),
        learningMember11['FL']))

    rules.append(ctrl.Rule(
        l['S'] & (
            ((Ii['S'] | Ii['E']) & Ij['I'] & ((Ai['H'] & Aj['M']) | (Ai['M'] & Aj['H']))) |
            ((Ii['S'] | Ii['E']) & (Ij['H'] | Ij['D']) & (Ai['L'] & Aj['L'])) |
            (Ii['I'] & (Ij['S'] | Ij['E']) & ((Ai['H'] & Aj['L']) | (Ai['M'] & Aj['M']) | (Ai['L'] & Aj['H']))) |
            (Ii['I'] & Ij['I'] & ((Ai['M'] & Aj['L']) | (Ai['L'] & Aj['M']))) |
            (Ii['D'])
        ),
        learningMember11['MM']))

    rules.append(ctrl.Rule(
        l['S'] & (
            ((Ii['S'] | Ii['E']) & Ij['I'] & (Ai['H'] & Aj['H'])) |
            ((Ii['S'] | Ii['E']) & (Ij['H'] | Ij['D']) & ((Ai['L'] & Aj['M']) | (Ai['M'] & Aj['L']))) |
            (Ii['I'] & (Ij['S'] | Ij['E']) & ((Ai['H'] & Aj['M']) | (Ai['M'] & Aj['H']))) |
            (Ii['I'] & Ij['I'] & ((Ai['H'] & Aj['L']) | (Ai['M'] & Aj['M']) | (Ai['L'] & Aj['H']))) |
            (Ii['I'] & (Ij['H'] | Ij['D']) & (Ai['L'] & Aj['L'])) |
            (Ii['H'] & (Ij['S'] | Ij['E']) & (Ai['L'] & Aj['L'] | (Ai['L'] & Aj['M']) | (Ai['M'] & Aj['L']))) |
            (Ii['H'] & Ij['I'] & (Ai['L'] & Aj['L']))
        ),
        learningMember11['LL']))
    rules.append(ctrl.Rule(
        l['S'] & (
            ((Ii['S'] | Ii['E']) & (Ij['H'] | Ij['D']) & ((Ai['H'] & Aj['L']) | (Ai['M'] & Aj['M']) | (Ai['L'] & Aj['H']))) |
            (Ii['I'] & (Ij['S'] | Ij['E']) & (Ai['H'] & Aj['H'])) |
            (Ii['I'] & Ij['I'] & ((Ai['H'] & Aj['M']) | (Ai['M'] & Aj['H']))) |
            (Ii['I'] & (Ij['H'] | Ij['D']) & ((Ai['L'] & Aj['M']) | (Ai['M'] & Aj['L']) | (Ai['M'] & Aj['M']))) |
            (Ii['H'] & (Ij['S'] | Ij['E']) & ((Ai['H'] & Aj['L']) | (Ai['M'] & Aj['M']) | (Ai['L'] & Aj['H']))) |
            (Ii['H'] & Ij['I'] & (Ai['M'] & Aj['L'])) |
            (Ii['H'] & (Ij['H'] | Ij['D']) & (Ai['H'] & Aj['L']))
        ),
        learningMember11['LW']))
    rules.append(ctrl.Rule(
        l['S'] & (
            ((Ii['S'] | Ii['E']) & (Ij['H'] | Ij['D']) & ((Ai['H'] & Aj['M']) | (Ai['M'] & Aj['H']))) |
            (Ii['I'] & Ij['I'] & (Ai['H'] & Aj['H'])) |
            (Ii['I'] & (Ij['H'] | Ij['D']) & ((Ai['L'] & Aj['H']) | (Ai['M'] & Aj['H']) | (Ai['H'] & Aj['L']) | (Ai['H'] & Aj['M']))) |
            (Ii['H'] & (Ij['S'] | Ij['E']) & ((Ai['H'] & Aj['M']) | (Ai['M'] & Aj['H']))) |
            (Ii['H'] & Ij['I'] & ((Ai['H'] & Aj['L']) | (Ai['M'] & Aj['M']) | (Ai['H'] & Aj['H']))) |
            (Ii['H'] & (Ij['H'] | Ij['D']) &
             ((Ai['H'] & Aj['M']) | (Ai['M'] & Aj['H'])))
        ),
        learningMember11['LM']))
    rules.append(ctrl.Rule(
        l['S'] & (
            ((Ii['S'] | Ii['E']) & (Ij['H'] | Ij['D']) & (Ai['H'] & Aj['H'])) |
            (Ii['I'] & (Ij['H'] | Ij['D']) & (Ai['H'] & Aj['H'])) |
            (Ii['H'] & (Ij['S'] | Ij['E']) & (Ai['H'] & Aj['H'])) |
            (Ii['I'] & (Ij['H'] | Ij['D']) & ((Ai['L'] & Aj['M']) | (Ai['M'] & Aj['H']) | (Ai['H'] & Aj['M']) | (Ai['H'] & Aj['M']))) |
            (Ii['H'] & (Ij['H'] | Ij['D']) & ((Ai['L'] & Aj['L'])
             | (Ai['M'] & Aj['M']) | (Ai['H'] & Aj['H'])))
        ),
        learningMember11['LS']))
    rules.append(ctrl.Rule(
        l['S'] & (
            (Ii['H'] & Ij['I'] & (Ai['L'] & Aj['H'])) |
            (Ii['H'] & (Ij['H'] | Ij['D']) & ((Ai['L'] & Aj['M'])
             | (Ai['L'] & Aj['H']) | (Ai['M'] & Aj['H'])))
        ),
        learningMember11['LH']))
    return rules


def SAISFuzzyModel_353_611_rules()->List[ctrl.Rule]:
    rules = SAISFuzzyInfectingRules_353_611()
    rules += SAISFuzzyLearningRules_353_611()
    return rules


saisFuzzyModel_222_35 = generateSAISFuzzyModel_222_35()
saisFuzzyModel_333_57 = generateSAISFuzzyModel_333_57()
saisFuzzyModel_333_611 = generateSAISFuzzyModel_333_611()
saisFuzzyInfectingModel_353_611 = ctrl.ControlSystemSimulation(
    ctrl.ControlSystem(SAISFuzzyInfectingRules_353_611()))
saisFuzzyLearningModel_353_611 = ctrl.ControlSystemSimulation(
    ctrl.ControlSystem(SAISFuzzyLearningRules_353_611()))
saisFuzzyModel_353_611 = ctrl.ControlSystemSimulation(
    ctrl.ControlSystem(SAISFuzzyModel_353_611_rules()))

sk = 32
sk0 = sk**0  # = 1
sk1 = sk**1
sk2 = sk**2
sk3 = sk**3
sk4 = sk**4


saisFuzzyCache_222_35 = [(-1, -1,)]*(sk4*sk)
saisFuzzyCacheChange_222_35 = False

saisFuzzyCache_333_611 = [(-1, -1,)]*(sk4*sk)
saisFuzzyCacheChange_333_611 = False

saisFuzzyInfectingCache_353_611 = [-1]*(sk4*sk)
saisFuzzyLearningCache_353_611 = [-2]*(sk4*sk)
saisFuzzyCache_353_611 = [(-1, -1,)]*(sk4*sk)
saisFuzzyCacheChange_353_611 = False


def saveFuzzyCache():
    if saisFuzzyCacheChange_222_35:
        with open("saisFuzzy_222_35.Cache", "wb") as fp:  # Pickling
            pickle.dump(saisFuzzyCache_222_35, fp)
    if saisFuzzyCacheChange_333_611:
        with open("saisFuzzy_333_611.Cache", "wb") as fp:  # Pickling
            pickle.dump(saisFuzzyCache_333_611, fp)
    if saisFuzzyCacheChange_333_611:
        with open("saisFuzzy_353_611.Cache", "wb") as fp:  # Pickling
            pickle.dump(saisFuzzyCache_353_611, fp)
        with open("saisFuzzyInfecting_353_611.Cache", "wb") as fp:  # Pickling
            pickle.dump(saisFuzzyInfectingCache_353_611, fp)
        with open("saisFuzzyLearning_353_611.Cache", "wb") as fp:  # Pickling
            pickle.dump(saisFuzzyLearningCache_353_611, fp)


atexit.register(saveFuzzyCache)

try:
    with open("saisFuzzy_222_35.Cache", "rb") as fp:   # Unpickling
        saisFuzzyCache_222_35 = pickle.load(fp)
except FileNotFoundError:
    pass
try:
    with open("saisFuzzy_333_611.Cache", "rb") as fp:   # Unpickling
        saisFuzzyCache_333_611 = pickle.load(fp)
except FileNotFoundError:
    pass
try:
    with open("saisFuzzy_353_611.Cache", "rb") as fp:   # Unpickling
        saisFuzzyCache_353_611 = pickle.load(fp)
except FileNotFoundError:
    pass
try:
    with open("saisFuzzyInfecting_353_611.Cache", "rb") as fp:   # Unpickling
        saisFuzzyInfectingCache_353_611 = pickle.load(fp)
except FileNotFoundError:
    pass
try:
    with open("saisFuzzyLearning_353_611.Cache", "rb") as fp:   # Unpickling
        saisFuzzyLearningCache_353_611 = pickle.load(fp)
except FileNotFoundError:
    pass


def saisFuzzy_222_35(link, inf1, inf2, alert1, alert2) -> Tuple[float, float]:
    global saisFuzzyCacheChange_222_35
    # k = "%d.%d.%d.%d" % (int(inf1*sk),int(inf2*sk),int(alert1*sk),int(alert2*sk))
    # k = int(inf1*sk*1000000),int(inf2*sk*10000),int(alert1*sk*100),int(alert2*sk)
    # if k in saisFuzzy235Cache:
    #     return saisFuzzy235Cache[k]
    k = (int(inf1*sk1) << (5*3))+(int(inf2*sk1) << (5*2)) + \
        (int(alert1*sk1) << 5)+int(alert2*sk1)
    r = saisFuzzyCache_222_35[k]
    if r != (-1, -1,):
        return r

    saisFuzzyModel_222_35.input['Link'] = link
    saisFuzzyModel_222_35.input['Infected1'] = inf1
    saisFuzzyModel_222_35.input['Infected2'] = inf2
    saisFuzzyModel_222_35.input['Alerted1'] = alert1
    saisFuzzyModel_222_35.input['Alerted2'] = alert2

    # Crunch the numbers
    saisFuzzyModel_222_35.compute()

    result = (saisFuzzyModel_222_35.output["Infecting"],
              saisFuzzyModel_222_35.output['Learning'],)
    saisFuzzyCache_222_35[k] = result
    saisFuzzyCacheChange_222_35 = True
    return result


def saisFuzzyPr_222_35(l, i1, i2, a1, a2):
    l = [0, 1]  # 'l': 0, 'h': 1

    L, H, HI = getLevels(3)
    M = H
    FH, FL, MEM, LL, LH = getLevels(5)

    linkInfecting = [0] * 3
    linkInfecting[L] = (l[L] + (
        l[H] * ((i1[L] * i2[L]) +
                (i1[H] * i2[H]) +
                (i1[H] * i2[L] * a1[H] * a2[H]) +
                (i1[L] * i2[H] * a1[H] * a2[H]))))
    linkInfecting[M] = (l[H] * (
        (i1[H] * i2[L] * a1[H] * a2[L]) +
        (i1[H] * i2[L] * a1[L] * a2[H]) +
        (i1[L] * i2[H] * a1[H] * a2[L]) +
        (i1[L] * i2[H] * a1[L] * a2[H])))
    linkInfecting[HI] = (l[H] * a1[L] * a2[L] *
                         ((i1[H] * i2[L]) + (i1[L] * i2[H])))

    linkLearning = [0] * 5
    linkLearning[FH] = (l[L] + (l[H] * (
        (a1[L] * a2[L] * i1[L] * i2[L]) +
        (a1[H] * a2[L] * i1[L] * i2[L]) +
        (a1[L] * a2[H] * i1[L] * i2[L]))))
    linkLearning[FL] = (l[H] *
                        (a1[H] * a2[H] * i1[L] * i2[L]))
    linkLearning[MEM] = (l[H] * (
        a1[H] * a2[H] *
        ((i1[H] * i2[L]) + (i1[L] * i2[H]))))
    linkLearning[LL] = (l[H] * (
        (a1[L] * a2[L] * (1-(i1[L] * i2[L]))) +
        (a1[L] * a2[H] * i1[L] * i2[H]) +
        (a1[H] * a2[L] * i1[H] * i2[L]) +
        (a1[H] * a2[H] * i1[H] * i2[H])))
    linkLearning[LH] = (l[H] * (
        (a1[L] * a2[H] * i1[H]) +
        (a1[H] * a2[L] * i2[H])))

    return (linkInfecting, linkLearning,)


def saisFuzzy_333_57(link, inf1, inf2, alert1, alert2) -> Tuple[float, float]:
    saisFuzzyModel_333_57.input['Link'] = link
    saisFuzzyModel_333_57.input['Infected1'] = inf1
    saisFuzzyModel_333_57.input['Infected2'] = inf2
    saisFuzzyModel_333_57.input['Alerted1'] = alert1
    saisFuzzyModel_333_57.input['Alerted2'] = alert2

    # Crunch the numbers
    saisFuzzyModel_333_57.compute()

    return (saisFuzzyModel_333_57.output["Infecting"],
            saisFuzzyModel_333_57.output['Learning'],)


def saisFuzzy_333_611(link, inf1, inf2, alert1, alert2) -> Tuple[float, float]:
    global saisFuzzyCacheChange_333_611
    # k = "%d.%d.%d.%d" % (int(inf1*sk),int(inf2*sk),int(alert1*sk),int(alert2*sk))
    # k = int(inf1*sk*1000000),int(inf2*sk*10000),int(alert1*sk*100),int(alert2*sk)
    # if k in saisFuzzy3611Cache:
    #     return saisFuzzy3611Cache[k]
    k = (int(inf1*sk1) << (5*3))+(int(inf2*sk1) << (5*2)) + \
        (int(alert1*sk1) << 5)+int(alert2*sk1)
    r = saisFuzzyCache_333_611[k]
    if r != (-1, -1):
        return r

    saisFuzzyModel_333_611.input['Link'] = link
    saisFuzzyModel_333_611.input['Infected1'] = inf1
    saisFuzzyModel_333_611.input['Infected2'] = inf2
    saisFuzzyModel_333_611.input['Alerted1'] = alert1
    saisFuzzyModel_333_611.input['Alerted2'] = alert2

    # Crunch the numbers
    saisFuzzyModel_333_611.compute()

    result = (saisFuzzyModel_333_611.output["Infecting"],
              saisFuzzyModel_333_611.output['Learning'],)
    saisFuzzyCache_333_611[k] = result
    saisFuzzyCacheChange_333_611 = True
    return result


def saisFuzzy_353_611(link, inf1, inf2, alert1, alert2) -> Tuple[float, float]:
    global saisFuzzyCacheChange_353_611
    k = (int(inf1*sk1) << (5*3))+(int(inf2*sk1) << (5*2)) + \
        (int(alert1*sk1) << 5)+int(alert2*sk1)
    r = saisFuzzyCache_353_611[k]
    if r != (-1, -1):
        return r

    saisFuzzyModel_353_611.input['Link'] = link
    saisFuzzyModel_353_611.input['Infected1'] = inf1
    saisFuzzyModel_353_611.input['Infected2'] = inf2
    saisFuzzyModel_353_611.input['Alerted1'] = alert1
    saisFuzzyModel_353_611.input['Alerted2'] = alert2

    # Crunch the numbers
    saisFuzzyModel_353_611.compute()

    result = (saisFuzzyModel_353_611.output["Infecting"],
              saisFuzzyModel_353_611.output['Learning'],)
    saisFuzzyCache_353_611[k] = result
    saisFuzzyCacheChange_353_611 = True
    return result



def saisFuzzyInfecting_353_611(link, inf1, inf2, alert1, alert2) -> float:
    global saisFuzzyCacheChange_353_611
    k = (int(inf1*sk1) << (5*3))+(int(inf2*sk1) << (5*2)) + \
        (int(alert1*sk1) << 5)+int(alert2*sk1)
    r = saisFuzzyInfectingCache_353_611[k]
    if r != -1:
        return r

    saisFuzzyInfectingModel_353_611.input['Link'] = link
    saisFuzzyInfectingModel_353_611.input['Infected1'] = inf1
    saisFuzzyInfectingModel_353_611.input['Infected2'] = inf2
    saisFuzzyInfectingModel_353_611.input['Alerted1'] = alert1
    saisFuzzyInfectingModel_353_611.input['Alerted2'] = alert2

    # Crunch the numbers
    saisFuzzyInfectingModel_353_611.compute()

    result = saisFuzzyInfectingModel_353_611.output["Infecting"]
    saisFuzzyInfectingCache_353_611[k] = result
    saisFuzzyCacheChange_353_611 = True
    return result



def saisFuzzyLearning_353_611(link, inf1, inf2, alert1, alert2) -> float:
    global saisFuzzyCacheChange_353_611
    k = (int(inf1*sk1) << (5*3))+(int(inf2*sk1) << (5*2)) + \
        (int(alert1*sk1) << 5)+int(alert2*sk1)
    r = saisFuzzyLearningCache_353_611[k]
    if r != -2:
        return r

    saisFuzzyLearningModel_353_611.input['Link'] = link
    saisFuzzyLearningModel_353_611.input['Infected1'] = inf1
    saisFuzzyLearningModel_353_611.input['Infected2'] = inf2
    saisFuzzyLearningModel_353_611.input['Alerted1'] = alert1
    saisFuzzyLearningModel_353_611.input['Alerted2'] = alert2

    # Crunch the numbers
    saisFuzzyLearningModel_353_611.compute()

    result = saisFuzzyLearningModel_353_611.output['Learning']
    saisFuzzyLearningCache_353_611[k] = result
    saisFuzzyCacheChange_353_611 = True
    return result


def boltzmann(x, xMid, tau):
    """
    evaluate the boltzmann function with midpoint xMid and time constant tau
    over x
    """
    return 1. / (1. + np.exp(-(x-xMid)/tau))


def boltzmann_list(x, px, py, fix=0.25):
    r = py[0]
    for i in range(len(px)-1):
        dy = py[i+1]-py[i]
        if dy == 0:
            continue
        xm = (px[i]+px[i+1])/2
        dx = px[i+1]-px[i]
        # ym = (py[i]+py[i+1])/2
        r += dy * boltzmann(x, xm, fix*dx)
    return r
