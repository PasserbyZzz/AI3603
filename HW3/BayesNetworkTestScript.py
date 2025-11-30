# -*- coding:utf-8 -*-

from BayesianNetworks import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#############################
## Example Tests from Bishop `Pattern Recognition and Machine Learning` textbook on page 377
#############################
BatteryState = readFactorTable(['battery'], [0.9, 0.1], [[1, 0]])
FuelState = readFactorTable(['fuel'], [0.9, 0.1], [[1, 0]])
GaugeBF = readFactorTable(['gauge', 'battery', 'fuel'], [0.8, 0.2, 0.2, 0.1, 0.2, 0.8, 0.8, 0.9], [[1, 0], [1, 0], [1, 0]])

carNet = [BatteryState, FuelState, GaugeBF]  # carNet is a list of factors
## Notice that different order of operations give the same answer
## (rows/columns may be permuted)
joinFactors(joinFactors(BatteryState, FuelState), GaugeBF)
joinFactors(joinFactors(GaugeBF, FuelState), BatteryState)

marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'gauge')
joinFactors(marginalizeFactor(GaugeBF, 'gauge'), BatteryState)

joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState)
marginalizeFactor(joinFactors(joinFactors(GaugeBF, FuelState), BatteryState), 'battery')

marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'gauge')
marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'fuel')

evidenceUpdateNet(carNet, ['fuel', 'battery'], [1, 0])

# inference
print("inference starts")
print(inference(carNet, ['battery', 'fuel'], [], []))  ## chapter 8 equation (8.30)
print(inference(carNet, ['battery'], ['fuel'], [0]))  ## chapter 8 equation (8.31)
print(inference(carNet, ['battery'], ['gauge'], [0]))  ##chapter 8 equation  (8.32)
print(inference(carNet, [], ['gauge', 'battery'], [0, 0]))  ## chapter 8 equation (8.33)
print("inference ends")
###########################################################################
# RiskFactor Data Tests
###########################################################################
riskFactorNet = pd.read_csv('RiskFactorsData.csv')

# Create factors

income = readFactorTablefromData(riskFactorNet, ['income'])
smoke = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
bmi = readFactorTablefromData(riskFactorNet, ['bmi', 'income'])
diabetes = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])

## you need to create more factor tables

risk_net = [income, smoke, long_sit, stay_up, exercise, bmi, diabetes]
print("income dataframe is ")
print(income)
factors = riskFactorNet.columns

# example test p(diabetes|smoke=1,exercise=2,long_sit=1)

margVars = list(set(factors) - {'diabetes', 'smoke', 'exercise', 'long_sit'})
obsVars = ['smoke', 'exercise', 'long_sit']
obsVals = [1, 2, 1]

p = inference(risk_net, margVars, obsVars, obsVals)
print(p)


###########################################################################
# Written Part
###########################################################################

# 1. Create the Bayesian Network
print("--- Task 1: Creating Bayesian Network ---")

# Define structure (Child: [Parents])
structure = {
    'income': [],
    'exercise': ['income'],
    'long_sit': ['income'],
    'stay_up': ['income'],
    'smoke': ['income'],
    'bmi': ['income', 'exercise', 'long_sit'],
    'cholesterol': ['exercise', 'income', 'smoke', 'stay_up'],
    'bp': ['income', 'exercise', 'long_sit', 'stay_up', 'smoke'],
    'diabetes': ['bmi'],
    'stroke': ['bmi', 'bp', 'cholesterol'],
    'attack': ['bmi', 'bp', 'cholesterol'],
    'angina': ['bmi', 'bp', 'cholesterol']
}

bayes_net = []
network_size = 0
full_joint_size = 1

# build Bayesian Network and calculate size
for child, parents in structure.items():
    varnames = [child] + parents
    factor = readFactorTablefromData(riskFactorNet, varnames)
    bayes_net.append(factor)
    
    # Calculate size (number of entries in CPT)
    cpt_size = len(factor)
    network_size += cpt_size
    # print(f"Factor {child}|{parents} size: {cpt_size}")

# What is the size (in terms of the number of probabilities needed) of this network?
print(f"Bayesian Network Size (sum of CPT sizes): {network_size}")

# calculate full joint distribution size
domain_sizes = {}
for col in riskFactorNet.columns:
    if col != 'Unnamed: 0':
        domain_sizes[col] = get_domain_size(riskFactorNet, col)
        full_joint_size *= domain_sizes[col]

# What is the total number of probabilities needed to store the full joint distribution?
print(f"Full Joint Distribution Size: {full_joint_size}")

# 2. Queries
print("\n--- Task 2: Health Outcome Queries ---")
outcomes = ['diabetes', 'stroke', 'attack', 'angina']

# 2(a) Bad Habits vs Good Habits
# Bad: smoke=1, exercise=2, long_sit=1, stay_up=1
# Good: smoke=2, exercise=1, long_sit=2, stay_up=2
bad_habits = {'smoke': 1, 'exercise': 2, 'long_sit': 1, 'stay_up': 1}
good_habits = {'smoke': 2, 'exercise': 1, 'long_sit': 2, 'stay_up': 2}

print("2(a) Probability of outcome=1 given habits:")
print(f"{'Outcome':<10} | {'Bad Habits':<12} | {'Good Habits':<12}")
for out in outcomes:
    # Bad Habits
    # We need to marginalize everything else.
    # Hidden vars = all vars - outcome - evidence vars
    all_vars = set(structure.keys())
    evidence_vars = list(bad_habits.keys())
    evidence_vals = list(bad_habits.values())
    hidden_vars = list(all_vars - set([out]) - set(evidence_vars))
    
    res_bad = inference(bayes_net, hidden_vars, evidence_vars, evidence_vals)
    prob_bad = res_bad[res_bad[out] == 1]['probs'].values[0] if 1 in res_bad[out].values else 0

    # Good Habits
    evidence_vars = list(good_habits.keys())
    evidence_vals = list(good_habits.values())
    hidden_vars = list(all_vars - set([out]) - set(evidence_vars))
    
    res_good = inference(bayes_net, hidden_vars, evidence_vars, evidence_vals)
    prob_good = res_good[res_good[out] == 1]['probs'].values[0] if 1 in res_good[out].values else 0
    
    # What is the probability of the outcome if I have bad habits? How about if I have good habits?
    print(f"{out:<10} | {prob_bad:.6f}     | {prob_good:.6f}")


# 2(b) Poor Health vs Good Health
# Poor: bp=1, cholesterol=1, bmi=3 (overweight)
# Good: bp=3, cholesterol=2, bmi=2 (normal)
poor_health = {'bp': 1, 'cholesterol': 1, 'bmi': 3}
good_health = {'bp': 3, 'cholesterol': 2, 'bmi': 2}

print("\n2(b) Probability of outcome=1 given health status:")
print(f"{'Outcome':<10} | {'Poor Health':<12} | {'Good Health':<12}")
for out in outcomes:
    # Poor Health
    evidence_vars = list(poor_health.keys())
    evidence_vals = list(poor_health.values())
    hidden_vars = list(all_vars - set([out]) - set(evidence_vars))
    
    res_poor = inference(bayes_net, hidden_vars, evidence_vars, evidence_vals)
    prob_poor = res_poor[res_poor[out] == 1]['probs'].values[0] if 1 in res_poor[out].values else 0

    # Good Health
    evidence_vars = list(good_health.keys())
    evidence_vals = list(good_health.values())
    hidden_vars = list(all_vars - set([out]) - set(evidence_vars))
    
    res_good = inference(bayes_net, hidden_vars, evidence_vars, evidence_vals)
    prob_good = res_good[res_good[out] == 1]['probs'].values[0] if 1 in res_good[out].values else 0
    
    # What is the probability of the outcome if I have poor health? How about if I have good health?
    print(f"{out:<10} | {prob_poor:.6f}     | {prob_good:.6f}")


# 3. Effect of Income
print("\n--- Task 3: Effect of Income ---")
print("Probability of outcome=1 given income level (1-8):")
print(f"{'Income':<6} | {'diabetes':<10} | {'stroke':<10} | {'attack':<10} | {'angina':<10}")

income_levels = range(1, 9)
probs_by_outcome = {out: [] for out in outcomes}

for i in income_levels:
    probs = []
    for out in outcomes:
        evidence_vars = ['income']
        evidence_vals = [i]
        hidden_vars = list(all_vars - set([out]) - set(evidence_vars))
        
        res = inference(bayes_net, hidden_vars, evidence_vars, evidence_vals)
        p = res[res[out] == 1]['probs'].values[0] if 1 in res[out].values else 0
        probs.append(p)
        probs_by_outcome[out].append(p)
    
    # Evaluate the effect a person’s income has on their probability of having one of the four health outcomes.
    print(f"{i:<6} | {probs[0]:.6f}   | {probs[1]:.6f}   | {probs[2]:.6f}   | {probs[3]:.6f}")

# Plotting
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^', 'D']
for idx, out in enumerate(outcomes):
    plt.plot(income_levels, probs_by_outcome[out], marker=markers[idx], label=out)

plt.xlabel('Income Level (1-8)')
plt.ylabel('Probability of Outcome = 1 (Yes)')
plt.title('Effect of Income on Health Outcomes')
plt.legend()
plt.grid(True)
plt.savefig('screenshots/income_effect_plot.png')
print("Plot saved to screenshots/income_effect_plot.png")


# 4. Test Assumptions (Edges from Habits to Outcomes)
print("\n--- Task 4: Testing Assumptions (Adding edges from habits to outcomes) ---")
# Create second network
structure_q4 = structure.copy()
# Add edges: smoke -> outcomes, exercise -> outcomes
# This means outcomes now have smoke and exercise as parents
# Original parents:
# diabetes: [bmi] -> [bmi, smoke, exercise]
# stroke: [bmi, bp, cholesterol] -> [bmi, bp, cholesterol, smoke, exercise]
# attack: [bmi, bp, cholesterol] -> [bmi, bp, cholesterol, smoke, exercise]
# angina: [bmi, bp, cholesterol] -> [income, bp, cholesterol, smoke, exercise]

structure_q4['diabetes'] = ['bmi', 'smoke', 'exercise']
structure_q4['stroke'] = ['bmi', 'bp', 'cholesterol', 'smoke', 'exercise']
structure_q4['attack'] = ['bmi', 'bp', 'cholesterol', 'smoke', 'exercise']
structure_q4['angina'] = ['bmi', 'bp', 'cholesterol', 'smoke', 'exercise']

bayes_net_q4 = []
for child, parents in structure_q4.items():
    varnames = [child] + parents
    factor = readFactorTablefromData(riskFactorNet, varnames)
    bayes_net_q4.append(factor)

print("Redoing Query 2(a) with new network:")
print(f"{'Outcome':<10} | {'Bad Habits':<12} | {'Good Habits':<12}")
for out in outcomes:
    # Bad Habits
    evidence_vars = list(bad_habits.keys())
    evidence_vals = list(bad_habits.values())
    hidden_vars = list(all_vars - set([out]) - set(evidence_vars))
    
    res_bad = inference(bayes_net_q4, hidden_vars, evidence_vars, evidence_vals)
    prob_bad = res_bad[res_bad[out] == 1]['probs'].values[0] if 1 in res_bad[out].values else 0

    # Good Habits
    evidence_vars = list(good_habits.keys())
    evidence_vals = list(good_habits.values())
    hidden_vars = list(all_vars - set([out]) - set(evidence_vars))
    
    res_good = inference(bayes_net_q4, hidden_vars, evidence_vars, evidence_vals)
    prob_good = res_good[res_good[out] == 1]['probs'].values[0] if 1 in res_good[out].values else 0
    
    # What assumption is this making about the effects of smoking and exercise on health problems?
    # Redo 2(a)
    print(f"{out:<10} | {prob_bad:.6f}     | {prob_good:.6f}")

print("\nRedoing Query 2(b) with new network:")
print(f"{'Outcome':<10} | {'Poor Health':<12} | {'Good Health':<12}")
for out in outcomes:
    # Poor Health
    evidence_vars = list(poor_health.keys())
    evidence_vals = list(poor_health.values())
    hidden_vars = list(all_vars - set([out]) - set(evidence_vars))
    
    res_poor = inference(bayes_net_q4, hidden_vars, evidence_vars, evidence_vals)
    prob_poor = res_poor[res_poor[out] == 1]['probs'].values[0] if 1 in res_poor[out].values else 0

    # Good Health
    evidence_vars = list(good_health.keys())
    evidence_vals = list(good_health.values())
    hidden_vars = list(all_vars - set([out]) - set(evidence_vars))
    
    res_good = inference(bayes_net_q4, hidden_vars, evidence_vars, evidence_vals)
    prob_good = res_good[res_good[out] == 1]['probs'].values[0] if 1 in res_good[out].values else 0
    
    # What assumption is this making about the effects of smoking and exercise on health problems?
    # Redo 2(b)
    print(f"{out:<10} | {prob_poor:.6f}     | {prob_good:.6f}")


# 5. Test Assumptions (Interaction between Health Problems)
print("\n--- Task 5: Testing Assumptions (Adding edge diabetes -> stroke) ---")
# Start from network in Q4
structure_q5 = structure_q4.copy()
# Add edge diabetes -> stroke
# stroke parents: [bmi, bp, cholesterol, smoke, exercise] -> [bmi, bp, cholesterol, smoke, exercise, diabetes]
structure_q5['stroke'] = ['bmi', 'bp', 'cholesterol', 'smoke', 'exercise', 'diabetes']

bayes_net_q5 = []
for child, parents in structure_q5.items():
    varnames = [child] + parents
    factor = readFactorTablefromData(riskFactorNet, varnames)
    bayes_net_q5.append(factor)

# Evaluate P(stroke=1 | diabetes=1) and P(stroke=1 | diabetes=3)
# For both networks (Q4 and Q5)

print("Comparing P(stroke=1 | diabetes) in Network Q4 vs Q5")
print(f"{'Condition':<20} | {'Network Q4':<12} | {'Network Q5':<12}")

# Condition 1: diabetes=1
cond1_vars = ['diabetes']
cond1_vals = [1]
hidden_vars = list(all_vars - set(['stroke']) - set(cond1_vars))

res_q4_c1 = inference(bayes_net_q4, hidden_vars, cond1_vars, cond1_vals)
p_q4_c1 = res_q4_c1[res_q4_c1['stroke'] == 1]['probs'].values[0] if 1 in res_q4_c1['stroke'].values else 0

res_q5_c1 = inference(bayes_net_q5, hidden_vars, cond1_vars, cond1_vals)
p_q5_c1 = res_q5_c1[res_q5_c1['stroke'] == 1]['probs'].values[0] if 1 in res_q5_c1['stroke'].values else 0

# What was the effect, and was the assumption about the interaction between diabetes and stroke valid?
print(f"{'diabetes=1':<20} | {p_q4_c1:.6f}     | {p_q5_c1:.6f}")

# Condition 2: diabetes=3
cond2_vars = ['diabetes']
cond2_vals = [3]

res_q4_c2 = inference(bayes_net_q4, hidden_vars, cond2_vars, cond2_vals)
p_q4_c2 = res_q4_c2[res_q4_c2['stroke'] == 1]['probs'].values[0] if 1 in res_q4_c2['stroke'].values else 0

res_q5_c2 = inference(bayes_net_q5, hidden_vars, cond2_vars, cond2_vals)
p_q5_c2 = res_q5_c2[res_q5_c2['stroke'] == 1]['probs'].values[0] if 1 in res_q5_c2['stroke'].values else 0

# What was the effect, and was the assumption about the interaction between diabetes and stroke valid?
print(f"{'diabetes=3':<20} | {p_q4_c2:.6f}     | {p_q5_c2:.6f}")