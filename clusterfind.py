#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as clus
import argparse
from collections import Counter
from pymaps import make_wafermap


# In[1]:


def clusterfind(mytable, targcol, labelcol, instancecol, numclusters=20, ncountstr='N Wafers:', imputation=None):
    makevector = mytable.pivot(index=instancecol, columns=labelcol, values=targcol)
    for col in makevector.columns:
        mymedian = np.median(makevector[col].dropna().tolist())
        makevector[col] = makevector[col].fillna(mymedian)
    # neighbors imputation
    if imputation and imputation.lower() == 'neighbors':
        for col in makevector.columns:
            values={}
            for index, row in makevector.iterrows():
                values[row.name] = row[col]
            mymedian = np.median(makevector[col].dropna())
            # two identical but sequential passes
            for x in range(2):
                for mykey in values:
                    if pd.notnull(values[mykey]):
                        continue
                    mydiex, mydiey = mykey.split('^')
                    mydiex = int(mydiex)
                    mydiey = int(mydiey)
                    myeight = [
                        str(mydiex-1) + '^' + str(mydiey-1),
                        str(mydiex-1) + '^' + str(mydiey),
                        str(mydiex-1) + '^' + str(mydiey+1),
                        str(mydiex) + '^' + str(mydiey-1),
                        str(mydiex) + '^' + str(mydiey+1),
                        str(mydiex+1) + '^' + str(mydiey-1),
                        str(mydiex+1) + '^' + str(mydiey),
                        str(mydiex+1) + '^' + str(mydiey+1)
                    ]
                    numseen = 0
                    makemean = 0
                    for valtry in myeight:
                        if valtry not in values:
                            continue
                        if pd.isnull(values[valtry]):
                            continue
                        numseen += 1
                        makemean += values[valtry]
                    if numseen > 3:
                        values[mykey] = makemean / numseen
            for mykey in values:
                if pd.isnull(values[mykey]):
                    values[mykey] = mymedian
            makevector[col] = makevector.apply(lambda row: values[row.name], axis=1)
    elif imputation and imputation.lower() == 'zone':
        makevector['DieX'] =makevector.apply(lambda row: int(row.name.split("^")[0]), axis=1)
        makevector['DieY'] =makevector.apply(lambda row: int(row.name.split("^")[1]), axis=1)
        # crude zone
        pass0sizex = 294 / (np.max(makevector['DieX']) - np.min(makevector['DieX']) + 1)
        pass0sizey = 294 / (np.max(makevector['DieY']) - np.min(makevector['DieY']) + 1)
        centerx = np.average([np.max(makevector['DieX']), np.min(makevector['DieX'])])
        centery = np.average([np.max(makevector['DieY']), np.min(makevector['DieY'])])
        makevector['radius'] = makevector.apply(lambda row: np.sqrt(((np.abs(row['DieX'] - centerx) + 0.5) * pass0sizex) ** 2 +
                                                                   ((np.abs(row['DieY'] - centery) + 0.5) * pass0sizey) ** 2), axis=1)
        def zonefun(row, centerx, centery):
            radius = row['radius']
            diex = row['DieX']
            diey = row['DieY']
            if radius < 70:
                returnval = 'A'
            elif radius < 95:
                returnval = 'B'
            elif radius < 115:
                returnval = 'C'
            elif radius < 130:
                returnval = 'D'
            else:
                returnval = 'E'
            if diex > centerx:
                if diey > centery:
                    returnval += '_Q1'
                else:
                    returnval += '_Q4'
            else:
                if diey > centery:
                    returnval += '_Q2'
                else:
                    returnval += '_Q3'
            return returnval
        makevector['Zone'] = makevector.apply(lambda row: zonefun(row, centerx, centery), axis=1)
        
        del makevector['DieX']
        del makevector['DieY']
        del makevector['radius']
        for col in makevector.columns:
            zonesum = makevector[[col, 'Zone']].groupby('Zone').median()
            totmedian = np.median(makevector[col].dropna())
            zonedict = {}
            # if a target zone has an imputation, use it.  Otherwise use mean.
            for index, row in zonesum.iterrows():
                if pd.notnull(row[col]):
                    zonedict[row.name] = row[col]
                else:
                    zonedict[row.name] = totmedian
            makevector[col] = makevector.apply(lambda row: row[col] if pd.notnull(row[col]) else zonedict[row['Zone']], axis=1)
        del makevector['Zone']

    # flatten for scipy.cluster
    targvector = []
    for col in makevector.columns:
        targvector.append(makevector[col].tolist())
    clustersfound = clus.linkage(targvector, 'ward')
    topclusters = clus.fcluster(clustersfound, numclusters, criterion='maxclust')
    topcluscount = Counter(topclusters)
    reversi = {}
    for key in sorted(topcluscount, reverse=True):
        if topcluscount[key] not in reversi:
            reversi[topcluscount[key]] = []
        reversi[topcluscount[key]].append(key)
    topcluskey = []
    topcluslis = []
    for key in sorted(reversi, reverse=True):
        for subkey in reversi[key]:
            topcluskey.append(subkey)
            topcluslis.append(key)
    remap = {}
    for counter, index in enumerate(topcluskey):
        remap[index] = counter + 1
    inclus = {}
    for col, precluster in zip(makevector.columns, topclusters):
        thiscluster = remap[precluster]
        if ncountstr:
            inclus[col] = str(thiscluster) + ' ' + ncountstr + ' ' + str(topcluscount[precluster])
        else:
            inclus[col] = str(thiscluster)
    return inclus


# In[ ]:




