#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# import sys


def bycolorgraph(tograph, xcol, ycol, spcol='Split', dotrendline=False,
                 dologscale=False, subplot=111, fig=False, sharex=False,
                 sharey=False, linearfit=False, myax=None):
    splitlist = sorted(tograph[spcol].drop_duplicates().tolist())
    if not fig:
        fig = plt.figure(figsize=(16, 12))
    if myax is None:
        if not sharex:
            if not sharey:
                ax = plt.subplot(subplot)
            else:
                ax = plt.subplot(subplot, sharey=sharey)
        else:
            if not sharey:
                ax = plt.subplot(subplot, sharex=sharex)
            else:
                ax = plt.subplot(subplot, sharex=sharex, sharey=sharey)
    else:
        ax = myax
        
    mycolors = colorgen()
    
    for color, _ in zip(mycolors, range(5)):
        pass
    
    for color, mysplit in zip(mycolors, splitlist):
        mysplittab = tograph[tograph.apply(lambda row: row[spcol] == mysplit, axis=1)]
        if np.issubdtype(mysplittab[xcol].dtype, np.datetime64):
            plt.plot_date(mysplittab[xcol], mysplittab[ycol], c=color, label=mysplit)
        else:
            mysplittab.plot.scatter(xcol, ycol, c=color, label=mysplit, ax=ax)
        if dotrendline:
            mytrendlines = mysplittab.groupby(xcol).mean().reset_index()
            ax.plot(mytrendlines[xcol], mytrendlines[ycol], '-', c=color, label='')
        if linearfit:
            mysplittabd = mysplittab[[xcol, ycol]].dropna()
            slope, intercept, r_value, _, _ = stats.linregress(mysplittabd[xcol], mysplittabd[ycol])
            mymin = np.min(mysplittabd[xcol])
            mymax = np.max(mysplittabd[xcol])
            if mymin < 0:
                mymin *= 1.02
            else:
                mymin *= 0.98
            if mymax < 0:
                mymax *= 0.98
            else:
                mymax *= 1.02
            myfitlines = [
                [mymin, mymax],
                [mymin * slope + intercept, mymax * slope + intercept]
            ]
            plt.plot(myfitlines[0], myfitlines[1], '-', c=color)
    ax.set_title(ycol + ' by ' + xcol)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1), borderaxespad=0., fontsize=10)
    if dologscale:
        ax.set_yscale('log')
    return fig, ax





