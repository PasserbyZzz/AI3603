# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from functools import reduce

###########################################################################
# Coding Part
###########################################################################

## Function to create a conditional probability table
## Conditional probability is of the form p(x1 | x2, ..., xk)
## varnames: vector of variable names (strings) first variable listed
##           will be x_i, remainder will be parents of x_i, p1, ..., pk
## probs: vector of probabilities for the flattened probability table
## outcomesList: a list containing a vector of outcomes for each variable
## factorTable is in the type of pandas dataframe
## See the test file for examples of how this function works
def readFactorTable(varnames, probs, outcomesList):
    factorTable = pd.DataFrame({'probs': probs})

    totalfactorTableLength = len(probs)
    numVars = len(varnames)

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(totalfactorTableLength / (k * numLevs))
        k = k * numLevs
    return factorTable


## Build a factorTable from a data frame using frequencies
## from a data frame of data to generate the probabilities.
## data: data frame read using pandas read_csv
## varnames: specify what variables you want to read from the table
## factorTable is in the type of pandas dataframe
def readFactorTablefromData(data, varnames):
    numVars = len(varnames)
    outcomesList = []

    for i in range(0, numVars):
        name = varnames[i]
        outcomesList = outcomesList + [list(set(data[name]))]

    lengths = list(map(lambda x: len(x), outcomesList))
    m = reduce(lambda x, y: x * y, lengths)

    factorTable = pd.DataFrame({'probs': np.zeros(m)})

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    numLevels = len(outcomesList[0])

    # creates the vector called fact to index probabilities
    # using matrix multiplication with the data frame
    fact = np.zeros(data.shape[1])
    lastfact = 1
    for i in range(len(varnames) - 1, -1, -1):
        fact = np.where(np.isin(list(data), varnames[i]), lastfact, fact)
        lastfact = lastfact * len(outcomesList[i])

    # Compute unnormalized counts of subjects that satisfy all conditions
    a = (data - 1).dot(fact) + 1
    for i in range(0, m):
        factorTable.at[i, 'probs'] = sum(a == (i + 1))

    # normalize the conditional probabilities
    skip = int(m / numLevels)
    for i in range(0, skip):
        normalizeZ = 0
        for j in range(i, m, skip):
            normalizeZ = normalizeZ + factorTable['probs'][j]
        for j in range(i, m, skip):
            if normalizeZ != 0:
                factorTable.at[j, 'probs'] = factorTable['probs'][j] / normalizeZ

    return factorTable


## Join of two factors
## Factor1, Factor2: two factor tables
##
## Should return a factor table that is the join of factor 1 and 2.
## You can assume that the join of two factors is a valid operation.
## Hint: You can look up pd.merge for mergin two factors
def joinFactors(Factor1, Factor2):
    # Get the variable names (columns excluding 'probs')
    vars1 = [c for c in Factor1.columns if c != 'probs']
    vars2 = [c for c in Factor2.columns if c != 'probs']
    
    # Find common variables to merge on
    commonVars = list(set(vars1).intersection(set(vars2)))
    
    if commonVars:
        # Merge on common variables
        merged = pd.merge(Factor1, Factor2, on=commonVars)
    else:
        # If no common variables, perform a cross join (Cartesian product)
        # Using a temporary key for cross join compatible with older pandas versions
        f1_temp = Factor1.copy()
        f2_temp = Factor2.copy()
        f1_temp['key'] = 1
        f2_temp['key'] = 1
        merged = pd.merge(f1_temp, f2_temp, on='key').drop('key', axis=1)
    
    # Calculate the new probabilities
    # pd.merge creates 'probs_x' and 'probs_y' for the 'probs' column
    merged['probs'] = merged['probs_x'] * merged['probs_y']
    
    # Drop the old probability columns
    merged = merged.drop(['probs_x', 'probs_y'], axis=1)
    
    return merged


## Marginalize a variable from a factor
## table: a factor table in dataframe
## hiddenVar: a string of the hidden variable name to be marginalized
##
## Should return a factor table that marginalizes margVar out of it.
## Assume that hiddenVar is on the left side of the conditional.
## Hint: you can look can pd.groupby
def marginalizeFactor(factorTable, hiddenVar):
    # Check if the hidden variable is in the table
    if hiddenVar not in factorTable.columns:
        return factorTable
    
    # Identify variables to keep (all columns except hiddenVar and 'probs')
    vars_to_keep = [c for c in factorTable.columns if c != hiddenVar and c != 'probs']
    
    # Group by the remaining variables and sum the probabilities
    if vars_to_keep:
        marginalized = factorTable.groupby(vars_to_keep)['probs'].sum().reset_index()
    else:
        # If no variables left (marginalizing the last variable), sum all probs
        marginalized = pd.DataFrame({'probs': [factorTable['probs'].sum()]})
        
    return marginalized


## Update BayesNet for a set of evidence variables
## bayesnet: a list of factor and factor tables in dataframe format
## evidenceVars: a list of variable names in the evidence list
## evidenceVals: a list of values for corresponding variables (in the same order)
##
## Set the values of the evidence variables. Other values for the variables
## should be removed from the tables. You do not need to normalize the factors
def evidenceUpdateNet(bayesnet, evidenceVars, evidenceVals):
    updated_net = []
    
    # Create a dictionary for easier lookup of evidence
    evidence_dict = dict(zip(evidenceVars, evidenceVals))
    
    for factor in bayesnet:
        new_factor = factor.copy()
        
        # Check each column in the factor
        for col in new_factor.columns:
            if col in evidence_dict:
                # Filter rows where the column value matches the evidence
                val = evidence_dict[col]
                new_factor = new_factor[new_factor[col] == val]
        
        updated_net.append(new_factor)

    return updated_net


## Run inference on a Bayesian network
## bayesnet: a list of factor tables and each table iin dataframe type
## hiddenVars: a list of variable names to be marginalized
## evidenceVars: a list of variable names in the evidence list
## evidenceVals: a list of values for corresponding variables (in the same order)
##
## This function should run variable elimination algorithm by using
## join and marginalization of the sets of variables.
## The order of the elimiation can follow hiddenVar ordering
## It should return a single joint probability table. The
## variables that are hidden should not appear in the table. The variables
## that are evidence variable should appear in the table, but only with the single
## evidence value. The variables that are not marginalized or evidence should
## appear in the table with all of their possible values. The probabilities
## should be normalized to sum to one.
def inference(bayesnet, hiddenVars, evidenceVars, evidenceVals):
    # Step 1: Update the network with evidence
    current_net = evidenceUpdateNet(bayesnet, evidenceVars, evidenceVals)
    
    # Step 2: Eliminate hidden variables one by one
    for hidden in hiddenVars:
        factors_with_hidden = []
        other_factors = []
        
        # Separate factors that contain the hidden variable
        for factor in current_net:
            if hidden in factor.columns:
                factors_with_hidden.append(factor)
            else:
                other_factors.append(factor)
        
        if factors_with_hidden:
            # Join all factors containing the hidden variable
            combined_factor = factors_with_hidden[0]
            for i in range(1, len(factors_with_hidden)):
                combined_factor = joinFactors(combined_factor, factors_with_hidden[i])
            
            # Marginalize the hidden variable
            marginalized_factor = marginalizeFactor(combined_factor, hidden)
            
            # Add the new factor back to the list
            other_factors.append(marginalized_factor)
            
        current_net = other_factors
    
    # Step 3: Join all remaining factors
    if not current_net:
        return pd.DataFrame({'probs': [1.0]}) # Should not happen in valid net
        
    final_factor = current_net[0]
    for i in range(1, len(current_net)):
        final_factor = joinFactors(final_factor, current_net[i])
        
    # Step 4: Normalize the probabilities
    total_prob = final_factor['probs'].sum()
    if total_prob > 0:
        final_factor['probs'] = final_factor['probs'] / total_prob
        
    return final_factor


## you can add other functions as you wish.
# Helper to get domain size
def get_domain_size(data, var):
    return len(set(data[var]))
