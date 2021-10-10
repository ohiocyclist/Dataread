#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from itertools import cycle
import pandas as pd
import numpy as np


def waterfallindexfun(lookup, waterfallorder, waterfallindex):
    if lookup == ".":
        return 9999
    if lookup not in waterfallindex:
        waterfallorder += [lookup]
        waterfallindex[lookup] = len(waterfallorder) + 1
    return waterfallindex[lookup]


def yieldmosaic_function(onelot, waterfallorder, binlettersname, spname='Split', wfrname='WaferId', savelocn=False):
    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
            ]
    waterfallindex = dict()
    for index in range(len(waterfallorder)):
        waferfallorder[index] = re.sub(r'\s+', '', waterfallorder[index])
        waterfallindex[waterfallorder[index]] = index
        
    newcolumn = onelot.apply(lambda row: waterfallindexfun(row[binlettersname], waterfallorder, waterfallindex), axis=1)
    try:
        onelot.insert(value=newcolumn, column='ordering', loc=len(onelot.columns))
    except ValueError as myerror:
        print(myerror)
        
    def spfun(spval, wfrval):
        if spval == 'Unknown':
            return wfrval
        return spval
    
    newc = onelot.apply(lambda row: spfun(str(row[spname]), str(row[wfrname])), axis=1)
    try:
        del onelot[spname]
    except Exception as E:
        print(E)
    try:
        onelot.insert(column=spname, value=newc, loc=0)
    except Exception as E:
        print(E)

    # fix waferId_waferId for lots with no splits
    spcol = spname
    countbybincols = [spname, 'DieX', binlettersname, 'ordering']
    groupbycols = [spname, binlettersname, 'ordering']
    indexlevel = [0, 1, 2]
    rarveto = True
    for index, row in onelot.iterrows():
        if str(row[spname]) != str(row[wfrname]) and len(str(row[spname])) > 0:
            rarveto = False
    if wfrname in onelot.columns and not rarveto:
        spcol = spname + '_' + wfrname
        newc = onelot.apply(lambda row: str(row[spname]) + '_' + str(row[wfrname]), axis=1)
        try:
            onelot.insert(column=spcol, value=newc, loc=len(onelot.columns))
        except Exception as E:
            print('could not add split_wafer, probably already exists,', E)
        countbybincols.append(spcol)
        groupbycols.append(spcol)
        indexlevel.append(3)
    # this counts up the DieX we defined above
    countbybin = onelot[countbybincols].groupby(groupbycols).count().reset_index(indexlevel)
    countbybin = countbybin.sort_values(['ordering'], ascending=True)
    splitlist = countbybin[spcol].drop_duplicates().sort_values(ascending=False)
    if len(splitlist) < 1:  # no splits, use wafers
        try:
            del onelot[spname]
        except Exception as E:
            print(E)
        newc = onelot.apply(lambda row: str(row[wfrname]), axis=1)
        onelot.insert(column=spname, value=newc, loc=len(onelot.columns))
        # counts up Die X
        countbybin = onelot[[spname, 'DieX', binlettername, 'ordering']].groupby([spname, binlettersname, 'ordering']).count().reset_index([0, 1, 2])
        countbybin = countbybin.sort_values(['ordering'], ascending=True)
        splitlist = countbybin[spcol].drop_duplicates().sort_values(ascending=False)
        
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(111)
    plt.gcf().subplots_adjust(left=0.25)
    
    voffset = 0
    myfaceset = {'.': 'blue'}
    lastsplit = False
    
    xystart = -10
    maxlen = 0
    for mysplit in splitlist:
        if len(mysplit) > maxlen:
            maxlen = len(mysplit)
    xystart = int(-4 - maxlen * 0.7)
    
    canzoom = True
    for mysplit in splitlist:
        anydot = False
        splitbybin = countbybin[countbybin[spcol] == mysplit]
        binwidths = splitbybin['DieX'].values
        bins = splitbybin[binlettersname].values
        totbins = np.sum(binwidths)
        for binval, binwidth in zip(bins, binwidths):
            if binval == ".":
                anydot = True
                if binwidth / totbins <= 0.5:
                    canzoom = False
                break
            else:
                continue
        if not anydot:
            canzoom = False
        if not canzoom:
            break
            
    if canzoom:
        xystart = xystart / 2
        
    if canzoom:
        bincut = 0.00075
    else:
        bincut = 0.0015
    totbinwidths = np.sum(countbybin['DieX'])
    binwidthsdf = countbybin[['DieX', binlettersname]].groupby([binlettersname]).sum().reset_index()
    for index, row in binwidthsdf.iterrows():
        if row['DieX'] / totbinwidths < bincut:
            myfaceset[row[binlettersname]] = 'darkviolet'
            
    if canzoom:
        bwshow = 0.8
    else:
        bwshow = 1.6
        
    facegen = colorgen()
    
    sawknown = False
    
    for mysplit in splitlist:
        if 'Unknown' in mysplit:
            if sawknown:
                continue
        else:
            sawknown = True
        if spcol == spname + '_' + wfrname:
            temparray = mysplit.split('_')
            thissplit = '_'.join(temparray[:-1])
            if lastsplit and thissplit != lastsplit:
                voffset += 6
            lastsplit = thissplit
        splitbybin = countbybin[countbybin[spcol] == mysplit]
        binwidths = splitbybin['DieX'].values
        bins = splitbybin[binlettersname].values
        totbins = np.sum(binwidths)
        startlocn = 0
        ax.annotate(mysplit, xy=(xystart, voffset + 2), xytext=(xystart, voffset + 2), annotation_clip=False)
        for binval, binwidth in zip(bins, binwidths):
            if canzoom and binval == '.':
                binwidth = (binwidth / totbins) * 100 - 50
            else:
                binwidth = (binwidth / totbins) * 100
            verts = [(startlocn, voffset),
                    (startlocn + binwidth, voffset),
                    (startlocn + binwidth, voffset + 10),
                    (startlocn, voffset + 10),
                    (startlocn, voffset)
                    ]
            path = Path(verts, codes)
            if binval not in myfaceset:
                myfaceset[binval] = next(facegen)
            facecolor = myfaceset[binval]
            patch = patches.PathPatch(path, facecolor=facecolor, lw=2)
            ax.add_patch(patch)
            if binwidth > bwshow:
                if binwidth < bwshow * 1.5:
                    ax.annotate(binval, xy=(startlocn + binwidth * 1 / 16, voffset + 2), xytext=(startlocn + binwidth * 1 / 16, voffset + 2))
                else:
                    ax.annotate(binval, xy=(startlocn + binwidth * 1 / 4, voffset + 2), xytext=(startlocn + binwidth * 1 / 4, voffset + 2))
                startlocn = startlocn + binwidth
            voffset += 12
        
        ax.set_xlim(0, startlocn)
        ax.set_ylim(-1, voffset)
        ax.set_yticks([])
        ax.set_xlabel('Percent Fallout')
        plt.title('Yield Mosaic')
        
        if savelocn and savelocn != chr(127):
            plt.savefig(savelocn)
            plt.close()
        elif savelocn != chr(127):
            myfolder = lotid[:7]
            if lotid[8:10] != '00':
                myfolder += '_' + lotid[8:11]
            myfolder += '_' + lotid[-7:]
            os.makedirs(r'c:\temp\\' + myfolder, exist_ok=True)
            plt.savefig(r'c:\temp\\' + myfolder + '\\aaYieldMosaic.PNG', bbox_inches='tight')
            plt.close()
            


# In[ ]:




