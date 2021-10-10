#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mplc
import pandas as pd
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib import cm
from matplotlib import style
import sys

import safesave

def onewafermap(fewmaps, thissplit, splitname, targcol, savefolder, maxz, minw, medblu, maxblu,
               minred, medred, maxw, flipscale=False, mycmap='seismic', appendtab=False, canlog=True,
               myax=None, xcolor='#000000'):
    rowmask = []
    for index, row in fewmaps.iterrows():
        rowmask.append(str(row[splitname])[:len(str(thissplit))] == str(thissplit))
    onemap = fewmaps[rowmask]
    if type(appendtab) != type(False):
        onemap = onemap.append(appendtab)
    if 'DieX' not in onemap.columns and 'Row' in onemap.columns:
        onemap.rename(columns={'Row': 'DieX'}, inplace=True)
    if 'DieY' not in onemap.columns and 'Col' in onemap.columns:
        onemap.rename(columns={'Col': 'DieY'}, inplace=True)
    x = onemap['DieX'].values
    y = onemap['DieY'].values
    w = onemap[targcol].values
    style.use('ggplot')
    if myax is None:
        f = plt.figure(figsize=(6, 4.5))
        ax = f.add_subplot(111)
    else:
        ax = myax
    
    codessq = [Path.MOVETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.CLOSEPOLY]
    codesx = [Path.MOVETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.CLOSEPOLY
             ]
    # with much help from David Daycock and Kevin Tetz
    norepeats = {}
    for thisx, thisy, thisw in zip(x, y, w):
        if str(thisx) + '_' + str(thisy) in norepeats:
            continue
        norepeats[str(thisx) + '_' + str(thisy)] = True
        if pd.isnull(thisw):
            startlocn = thisx - 0.5
            voffset = thisy - 0.5
            binwidth = 1
            verts = [(startlocn, voffset),
                     (startlocn + binwidth, voffset + 1),
                     (startlocn, voffset + 1),
                     (startlocn + binwidth * 0.5, voffset + 0.5),
                     (startlocn, voffset),
                     (startlocn + binwidth, voffset),
                     (startlocn + binwidth * 0.55, voffset + 0.45),
                     (startlocn + binwidth, voffset),
                     (startlocn + binwidth, voffset + 1),
                     (startlocn, voffset + 1),
                     (startlocn, voffset),
                    ]
            path = Path(verts, codesx)
            patch = patches.PathPatch(path, edgecolor=xcolor, facecolor="#888888", lw=2)
            ax.add_patch(patch)
        dolog = False
        if canlog and medblu > 0 and maxw > medblu * 60:
            # must have no zeroes and non negative!  Truncate tail to 0.1
            tailtruncate = 0.1
            wmask = numpy.ma.masked_equal(w, 0.0, copy=False)
            truncmin = np.nanmin(wmask)
            if truncmin > 0:
                while tailtruncate > truncmin:
                    tailtruncate /= 10
            for i in range(len(w)):
                if w[i] <= 0:
                    w[i] = tailtruncate
            if minw  <= 0:
                minw = tailtruncate
            dolog = True
            
        mypatches = []
        mypatchcolors = []
        for thisx, thisy, thisw in zip(x, y, w):
            if str(thisx) + '_' + str(thisy) in norepeats:
                continue
            norepeats[str(thisx) + '_' + str(thisy)] = True
            if pd.isnull(thisw):
                continue
            if maxw > minw:  # if maxw is minw this div's 0
                if dolog:
                    thiscv = (np.log(thisw) - np.log(minw)) / (np.log(maxw) - np.log(minw))
                else:
                    thiscv = (thisw - minw) / (maxw - minw)
            else:
                thiscv = 0.5
            if thiscv < 0:
                thiscv = 0
            # moving blue for max bugfix
            if thiscv > 0.99:
                thiscv = 0.99
            startlocn = thisx - 0.5
            voffset = thisy - 0.5
            binwidth = 1
            verts = [(startlocn, voffset),
                    (startlocn + binwidth, voffset),
                    (startlocn + binwidth, voffset + 1),
                    (startlocn, voffset + 1),
                    (startlocn, voffset)
                    ]
            path = Path(verts, codessq)
            if flipscale:
                patchcmd = 'patches.PathPatch(path, edgecolor="#000000", facecolor=cm.' + mycmap + '_r(thiscv), lw=2)'
                patch = eval(patchcmd)
            else:
                if type(mycmap) == str:
                    patchcmd = 'patches.PathPatch(path, edgecolor="#000000", facecolor=cm.' + mycmap + '(thiscv), lw=2)'
                else:
                    patchcmd = 'patches.PathPatch(path, edgecolor="#000000", facecolor=mycmap(thiscv), lw=2)'
                patch = eval(patchcmd)
            ax.add_patch(patch)
            mypatches.append(patch)
            mypatchcolors.append(cm.seismic(thiscv))

    if dolog:
        if flipscale:
            cbartarg = PatchCollection(mypatches, cmap=plt.get_cmap(mycmap + '_r'), norm=mplc.LogNorm())
        else:
            cbartarg = PatchCollection(mypatches, cmap=plt.get_cmap(mycmap), norm=mplc.LogNorm())
    else:
        if flipscale:
            cbartarg = PatchCollection(mypatches, cmap=plt.get_cmap(mycmap + '_r'))
        else:
            cbartarg = PatchCollection(mypatches, cmap=plt.get_cmap(mycmap))
    cbartarg.set(array=mypatchcolors)
    cbartarg.set_clim(minw, maxw)
    ax.set_title(targcol + ' map ' + splitname + ' ' + str(thissplit), fontsize=8)
    ax.set_xlabel('Row', fontsize=8)
    ax.set_ylabel('Col', fontsize=8)
    ax.autoscale(tight=True)
    plt.colorbar(cbartarg, ax=ax)
    
    savename = targcol + '_map_split_' + str(thissplit)
    savename = safesave.safesavemaker(savename)
    if savefolder:
        try:
            plt.savefig(savefolder + '\\' + savename + '.png', bbox_inches='tight')
        except Exception as myerr:
            print("pymaps saving error:", myerr)
        plt.close()
        
def make_wafermap(mapsrc, targcol, savefolder, splitcol, canlog=True, cmap='seismic', mymin=None, mymax=None, myax=None, xcolor='#000000'):
    if 'DieX' not in mapsrc.columns and 'Row' in mapsrc.columns:
        mapsrc.rename(columns={'Row': 'DieX'}, inplace=True)
    if 'DieY' not in mapsrc.columns and 'Col' in mapsrc.columns:
        mapsrc.rename(columns={'Col': 'DieY'}, inplace=True)
    fewmaps = mapsrc[[splitcol, 'DieX', 'DieY', targcol]].groupby([splitcol, 'DieX', 'DieY']).mean().sort_values(by=targcol).reset_index([0, 1, 2])
    appendtab = mapsrc[['DieX', 'DieY']].drop_duplicates()
    indexin = fewmaps[[targcol]].drop_duplicates().reset_index([0])
    fewmaps = pd.merge(fewmaps, indexin, on=[targcol], suffixes=('', '__'))
    splitlist = mapsrc[splitcol].drop_duplicates()
    z = fewmaps['index'].values
    w = fewmaps[targcol].values
    maxz = np.nan
    for thisz, thisw in zip(z, w):
        if pd.notnull(thisw):
            maxw = thisw
            maxz = thisz
    if mymax is not None:
        maxw = mymax
    # don't run for all null thisw
    if pd.notnull(maxz):
        minw = -1
        maxblu = -1
        minred = -1
        medblu = -1
        medred = -1
        for thisz, thisw in zip(z, w):
            if minw == -1:
                minw = thisw
            if medblu == -1 and thisz > maxz * 0.159 / 2:
                medblu = thisw
            if maxblu == -1 and thisz > maxz * 0.159:
                maxblu = thisw
            if minred == -1 and thisz > maxz * 0.841:
                minred = thisw
            if medred == -1 and thisz > maxz * (0.841 + (1 - 0.841) / 2):
                medred = thisw
        if mymin is not None:
            minw = mymin
        # crude tail detection and elimination
        # take in 1% slack if we detect a tail
        onepercent = int(maxz / 100)
        if onepercent < 5:
            onepercent = 5
        if onepercent > 50:
            onepercent = 50
        # this was five.  Bias towards chopping more tails.
        if maxw > medred + 2 * (medred - minred) and medred - minred > 0:
            clickpercent = onepercent
            realclick = 1
            actualclick = clickpercent
            while (pd.isnull(w[-clickpercent]) or realclick < actualclick) and clickpercent < len(w) / 2:
                if pd.notnull(w[-clickpercent]):
                    realclick += 1
                clickpercent += 1
            if pd.notnull(w[-clickpercent]) and mymax is None:
                maxw = w[-clickpercent]
        if minw < medblu - 2 * (maxblu - medblu) and maxblu - medblu > 0:
            clickpercent = onepercent
            while pd.isnull(w[clickpercent]) and clickpercent > 1:
                clickpercent = clickpercent - 1
            if pd.notnull(w[clickpercent]) and mymin is None:
                minw = w[clickpercent]
        for thissplit in splitlist:
            if pd.isnull(thissplit):
                continue
            onewafermap(fewmaps, thissplit, splitcol, targcol, savefolder, maxz, minw, medblu, maxblu, minred, medred, maxw,
                        appendtab=appendtab, canlog=True, mycmap=cmap, myax=myax, xcolor=xcolor)
            


# In[ ]:




