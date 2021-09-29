"""
Probplot.py

Create probplot / boxplot combos for quick and accurate visual
understanding of (quasi-)normal data
Stack multiple probplots on one graph
Create a summary stats table if desired
Show the results on the screen or save them as a PNG

"""
from __future__ import print_function
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
import importlib

# check to see if we have seaborn before optionally importing it
seaborn_spec = importlib.util.find_spec("seaborn")
if seaborn_spec is not None:
    import seaborn as sns


##############################################################################
def numclean(infloat):
    """
    returns something decimals appropriate for infloat

    Parameters
    ----------
    infloat : float
        floating point number to round to appropriate decimals

    Returns
    -------
    string rounding of floating decimals

    """
    if infloat == 0:
        return "0"
    elif pd.isnull(infloat):
        return ""
    elif abs(infloat) >= 1e6:
        return"%1.3e" % infloat
    elif float(int(infloat)) == float(infloat):
        # in other words, no decimals
        return "%6.0f" % infloat
    elif abs(infloat) >= 100:
        # decimals not important
        return "%6.0f" % infloat
    elif abs(infloat) >= 10:
        return "%2.2f" % infloat
    elif abs(infloat) >= 1:
        return "%1.3f" % infloat
    elif abs(infloat) >= 0.1:
        return "%1.4f" % infloat
    elif abs(infloat) >= 0.01:
        return "%1.5f" % infloat
    return "%1.3e" % infloat


##############################################################################
def probplot(dat, desiredtarget, mypred, connect_points=False, mytitle=False,
             axesswap=False, withboxplots=False, setranges=False,
             usepicker=False, logscale=None, newsize=False):
    """

    Create a multi-probplot graph, optionally with boxplots and summary stats

    Parameters
    ----------
    dat : Pandas Dataframe
        Dataframe with our data to plot.  Should have desiredtarget and mypred
        columns
    desiredtarget : String
        Column name in dat that contains the numeric data we are plotting
    mypred : String
        Column name in dat that contains the grouping column (string)
    connect_points : Boolean, optional
        Whether to join points on the probplot with connecting lines.
        The default is False.
    mytitle : String or Boolean, optional
        The title string to display over the graph.  False for auto-title.
        The default is False.
    axesswap : Boolean, optional
        Whether to swap the X and Y axes The default is False.
    withboxplots : Boolean, optional
        Whether to add boxplots to the graph. Also adds a summary table.
        The default is False.
    setranges : 2x2 tuple of float or False, optional
        If a 2x2 tuple of float is passed in, the first dimension is the lower
        and upper x bound and the second the y bound.  False for auto bounds.
        The default is False.
    usepicker : Boolean, optional
        If true, use matplotlib plot, otherwise use scatter.
        The default is False.
    logscale : Boolean, optional
        If None, set logarithmic scale automatically.  Otherwise False for
        linear, True for log. The default is None.
    newsize : Integer or False, optional
        If an integer is passed, set the graphic size, or False for automatic.
        The default is False.

    Returns
    -------
    thisplt : Matplotlib figure
        Handle to the plot created
    success : Boolean
        Whether the function ran correctly
    toprange : Integer
        The automatically determined top end of the graph
    botrange : Integer
        The automatically determined bottom end of the graph
    ppax : Matplotlib ax
        A handle to the axis of just the probplot, not the box plot

    """
    # cannot do boxplots without seaborn
    if seaborn_spec is None:
        withboxplots = False
    newstyleplots = True
    # cannot do newstyleplots with old matplotlib
    if str(mpl.__version__)[:1] == "1":
        newstyleplots = False
    # get the dataframe out of the way early
    pretotlist = dat[desiredtarget].astype(float).values.tolist()
    if mypred not in dat.columns:
        print('probplot error,', mypred, 'not in dat.columns')
        return(None, False, -1, -1, -1)
    totkeys = dat[mypred].values.tolist()

    testvals = []
    keysoftestvals = []
    # this is a Numba-ism, predeclaring memory like C
    # at one point I attempted to speed this up with Numba.  It didn't work.
    totlist = np.full(shape=len(pretotlist), fill_value=np.nan,
                      dtype=np.float32)
    tlend = 0
    # break out the data into the groups
    for thisval, keyelement in zip(pretotlist, totkeys):
        # force a single column, not multiple columns with the same name
        if type(thisval) is list:
            thisval = thisval[0]
        # if the conditional throws an error, evaluate to False
        try:
            myconditional = pd.notnull(thisval) and pd.notnull(keyelement)
        except ValueError as myerror:
            print("Probplot Trendplot myconditional error", myerror)
            print("passed thisval and keyelement", thisval, keyelement)
            myconditional = False
        if myconditional:
            mypredfound = 0
            shortlist = 0
            # linear search, no dictionary
            # (from the aborted Numba optimization)
            while shortlist < len(keysoftestvals) and mypredfound == 0:
                if keysoftestvals[shortlist] == keyelement:
                    mypredfound = 1
                else:
                    shortlist = shortlist + 1
            if mypredfound == 0:
                keysoftestvals.append(keyelement)
                myadd = np.full(shape=1, fill_value=thisval, dtype=np.float32)
                testvals.append(myadd)
            else:
                testvals[shortlist] = np.append(testvals[shortlist], thisval)
            totlist[tlend] = thisval
            tlend = tlend + 1

    # truncate allocation to what we need
    totlist = totlist[:tlend]

    mydpi = 100

    # set the figure size
    # works on newer matplotlib
    try:
        mpl.style.use('seaborn')
    except Exception as myerr:
        print("could not set style seaborn (is it installed?)", myerr)

    # matplotlib alt versions
    if newsize:
        thisplt = plt.figure(figsize=(newsize, newsize / 2))
    else:
        if withboxplots:
            thisplt = plt.figure(figsize=(1536/mydpi, 768/mydpi), dpi=mydpi)
        else:
            thisplt = plt.figure(figsize=(1024/mydpi, 768/mydpi), dpi=mydpi)

    # if we were unable to find any groups, we're sunk.
    if len(testvals) < 1:
        # return null
        print("probplot len testvals < 1")
        return (thisplt, False, 0, 0, 0)

    probvals = []
    markervals = []
    markerstep = 0
    bintestvals = []
    titles = []
    totlen = 0
    # sort each of the groups for easy quantile determination
    for index in range(len(testvals)):
        testvals[index].sort()
        bintestvals.append(str(keysoftestvals[index]))
        probvals.append([])
        markervals.append(markerstep)
        markerstep = markerstep + 1
        totlen = totlen + len(testvals[index])

    # set a reasonable graph range not driven by the flier points
    # how many points to cut is determined by the data size
    # but some of this does assume some type of normality
    if totlen < 30:
        qbotrange = np.min(totlist)
        qtoprange = np.max(totlist)
    elif totlen < 300:
        qbotrange = np.percentile(totlist, q=1)
        qtoprange = np.percentile(totlist, q=99)
    else:
        qbotrange = np.percentile(totlist, q=0.1)
        qtoprange = np.percentile(totlist, q=99.9)

    # use the total min/max if we're not too flier-like
    ubotrange = np.min(totlist)
    quanllrange = np.percentile(totlist, q=10)
    utoprange = np.max(totlist)

    if utoprange < qtoprange * 2 or utoprange < 0:
        toprange = utoprange
    else:
        toprange = qtoprange

    # this is better for log scale
    # to use the minimum point > 0
    if ubotrange == 0:
        for val in sorted(totlist):
            if val > 0:
                ubotrange = val
                break

    if ubotrange > qbotrange * 0.5 and ubotrange >= 0:
        botrange = ubotrange
    else:
        botrange = qbotrange

    # now that we have avoided the fliers, step the range out slightly
    if toprange > 0:
        toprange = toprange * 1.005
    else:
        toprange = toprange * 0.995

    if botrange < 0:
        botrange = botrange * 1.005
    else:
        botrange = botrange * 0.995

    # determine if the data is logscale
    # cheat a little bit.  If the minimum is zero just shuffle the zero
    # points off the edge of the graph.
    if logscale is None:
        logscale = False
        if botrange > 0 and toprange > botrange * 200:
            logscale = True
            if ubotrange > 0:
                botrange = ubotrange
            elif botrange == 0 and toprange > quanllrange * 200 \
                    and quanllrange > 0:
                logscale = True
                # botrange should not be zero for logscale
                for val in sorted(totlist):
                    if val > 0:
                        botrange = val
                        break

    # Make the tic marks look nice by overriding Matplotlib's defaults
    # we need to find a top and a bottom and an interval to do this.
    # scale the data to a 1-10 range, set intervals, and rescale to original
    interval = abs((toprange - botrange) / 5)
    bringup = 0
    # we can have unform data after all
    if abs(interval) > 0:
        while interval < 1:
            bringup = bringup + 1
            interval = interval * 10
        while interval > 10:
            bringup = bringup - 1
            interval = interval / 10
    if interval > 5:
        interval = 5
    elif interval > 2:
        interval = 2
    else:
        interval = 1

    while bringup > 0:
        interval = interval / 10
        bringup = bringup - 1
    while bringup < 0:
        interval = interval * 10
        bringup = bringup + 1

    # set the bottom scale to be some floor'd multiple of the interval
    # unless we're in logscale, that multiple can be zero.
    if botrange >= 0 and not logscale:
        botrange = int(botrange / interval) * interval
    elif not logscale:
        botrange = int((botrange - interval) / interval) * interval
    toprange = int((toprange + interval) / interval) * interval

    # calculate prob values
    minprob = 0
    for index in range(len(keysoftestvals)):
        # if we call stats.probplot on < 3 values we get spurious warnings
        if len(testvals[index]) > 2:
            res = stats.probplot(testvals[index])
            probvals[index] = res[0][0]
        elif len(testvals[index]) == 2:
            probvals[index] = [-0.54495214, 0.54495214]
        else:
            probvals[index] = [0]
        thisprob = np.min(probvals[index])
        if thisprob < minprob:
            minprob = thisprob

    # create the summary table, if requested
    if withboxplots:
        # make the probplot and the boxplot take up 3 high each to fill space
        gridspec.GridSpec(4, 2)
        seab_ax = plt.subplot2grid((4, 2), (0, 0), rowspan=3)
        ax = plt.subplot2grid((4, 2), (0, 1), rowspan=3)
        ppax = ax
        titleax = plt.subplot2grid((4, 2), (3, 0), colspan=2)

        # these are all the types of summary we include
        data = []
        short_table = dat[[mypred, desiredtarget]]
        data.append(short_table.groupby(mypred).count()
                    .sort_index()[desiredtarget].tolist())
        data.append(short_table.groupby(mypred).std()
                    .sort_index()[desiredtarget].tolist())
        data.append(short_table.groupby(mypred).min()
                    .sort_index()[desiredtarget].tolist())
        data.append(short_table.groupby(mypred).quantile(0.1)
                    .sort_index()[desiredtarget].tolist())
        data.append(short_table.groupby(mypred).quantile(0.25)
                    .sort_index()[desiredtarget].tolist())
        data.append(short_table.groupby(mypred).median()
                    .sort_index()[desiredtarget].tolist())
        data.append(short_table.groupby(mypred).mean()
                    .sort_index()[desiredtarget].tolist())
        data.append(short_table.groupby(mypred).quantile(0.75)
                    .sort_index()[desiredtarget].tolist())
        data.append(short_table.groupby(mypred).quantile(0.9)
                    .sort_index()[desiredtarget].tolist())
        data.append(short_table.groupby(mypred).max()
                    .sort_index()[desiredtarget].tolist())
        data = np.transpose(data)
        columns = ('N', 'Stddev', 'Min', 'Quan10', 'Quan25',
                   'Median', 'Mean', 'Quan75', 'Quan90', 'Max')
        rows = dat[mypred].sort_values().drop_duplicates().dropna().tolist()
        cell_text = []
        for row in range(len(rows)):
            cell_text.append([numclean(x) for x in data[row]])

        # don't show an axis on our table
        titleax.axis('tight')
        titleax.axis('off')

        # add the table
        titleax.table(cellText=cell_text, rowLabels=rows, colLabels=columns,
                      loc='bottom')

        # courtesy David Daycock
        mydat = dat[[mypred, desiredtarget]] \
            .dropna(axis=0).sort_values(by=mypred)
        current_palette = sns.color_palette()
        sns.boxplot(data=mydat, x=mypred, y=desiredtarget,
                    ax=seab_ax, showfliers=False, palette=current_palette)
        sns.stripplot(x=mypred, y=desiredtarget, data=mydat,
                      jitter=True, palette=current_palette,
                      edgecolor='black', linewidth=0.3, ax=seab_ax)

        sumticklen = 0
        for ticklabel in mydat[mypred].drop_duplicates().tolist():
            sumticklen += len(str(ticklabel))

        # this may or may not properly rotate tick labels, feel free
        # to adjust to taste
        if sumticklen > 24:
            seab_ax.set_xticklabels(sorted(mydat[mypred]
                                    .drop_duplicates().tolist()), rotation=90)
        if not mytitle:
            seab_ax.set_title(mypred + ' vs. ' + desiredtarget, fontsize=12)
        else:
            seab_ax.set_title(mytitle, fontsize=12)
    else:
        ax = plt.subplot(111)
        ppax = ax
        if not mytitle:
            ax.set_title(mypred + ' vs. ' + desiredtarget, fontsize=12)
        else:
            ax.set_title(mytitle, fontsize=12)
    # also update in the trendplot section
    # we want to have more possible color/marker combinations than is practical
    # newstyle however just uses Matplotlib color schemes
    markerlist = ['ro', 'go', 'bo', 'co', 'mo', 'yo' 'rv', 'bv', 'gv', 'cv',
                  'mv', 'yv', 'r^', 'g^', 'b^', 'c^', 'm^', 'y^', 'rs', 'gs',
                  'bs', 'cs', 'ms', 'ys', 'rD', 'gD', 'bD', 'cD', 'mD', 'yD',
                  'kD']
    countup = 0
    colorcount = 0
    # plot all the data
    for key in sorted(bintestvals):
        newstyle = 'C' + str(colorcount)
        colorcount += 1
        if colorcount >= 6:
            colorcount = 0
        mypredfound = 0
        # find the group of interest (for this for loop round)
        shortlist = 0
        while shortlist < len(bintestvals) and mypredfound == 0:
            if bintestvals[shortlist] == key:
                mypredfound = 1
            else:
                shortlist = shortlist + 1
        if mypredfound == 1:
            titles.append(key)
            markerstep = markervals[countup]
            countup = countup + 1
            while markerstep >= len(markerlist):
                markerstep = markerstep - len(markerlist)
            if axesswap:
                ax.scatter(testvals[shortlist], probvals[shortlist],
                           marker=markerlist[markerstep][-1:], s=24,
                           color=newstyle)
            else:
                if newstyleplots:
                    if usepicker:
                        ax.plot(probvals[shortlist],
                                testvals[shortlist],
                                marker=markerlist[markerstep][-1:],
                                color=newstyle, linewidth=0)
                    else:
                        ax.scatter(probvals[shortlist],
                                   testvals[shortlist],
                                   marker=markerlist[markerstep][-1:],
                                   s=24, color=newstyle)
                else:  # old style matplotlib < 2
                    ax.plot(probvals[shortlist],
                            testvals[shortlist],
                            markerlist[markerstep])
            markerstep = markerstep + 1
            if markerstep > len(markerlist):
                markerstep = 0
        else:
            print("PROBPLOT WARNING: Could Not Find Key", key)

    # plot the boxplot
    if withboxplots:
        # space depending on the number of graphs
        ylegendloc = 0.5
        if len(titles) < 15:
            ylegendloc = 0.8
        maxlegendlen = 0
        for thisval in titles:
            if len(thisval) > maxlegendlen:
                maxlegendlen = len(thisval)
        xlegendoffset = 0.01 * maxlegendlen
        ax.legend(titles, bbox_to_anchor=(1.2 + xlegendoffset, ylegendloc),
                  borderaxespad=0., fontsize=10)
    else:
        # where the legend goes depends on whether there's one plot or two
        plt.legend(titles, bbox_to_anchor=(1, 1), loc=2,
                   borderaxespad=0., fontsize=10)
    # draw the lines, per group
    if connect_points:
        colorcount = 0
        predash = False
        dodash = False
        predoubledash = False
        doubledash = False
        countup = 0
        for key in sorted(bintestvals):
            newstyle = 'C' + str(colorcount)
            colorcount += 1
            if predash:
                dodash = True
            if predoubledash:
                doubledash = True
            if dodash and colorcount == 6:
                predoubledash = True
            if colorcount >= 6:
                colorcount = 0
                # defer dashes to next loop
                predash = True
            if doubledash:
                newerstyle = newstyle + '-.'
            elif dodash:
                newerstyle = newstyle + '--'
            else:
                newerstyle = newstyle
            mypredfound = 0
            shortlist = 0
            while shortlist < len(bintestvals) and mypredfound == 0:
                if bintestvals[shortlist] == key:
                    mypredfound = 1
                else:
                    shortlist = shortlist + 1
            if mypredfound == 1:
                markerstep = markervals[countup]
                countup = countup + 1
                if logscale:
                    # this is to prevent an issue where lines
                    # lead off to negative infinity on logscale graphs
                    newvals = []
                    for thisval in testvals[shortlist]:
                        if float(thisval) > 0.0:
                            newvals.append(thisval)
                        else:
                            newvals.append(botrange * 3 / 4)
                    testvals[shortlist] = newvals
                if axesswap:
                    ax.plot(testvals[shortlist],
                            probvals[shortlist], newerstyle, ms=24)
                else:
                    if newstyleplots:
                        ax.plot(probvals[shortlist],
                                testvals[shortlist], newerstyle, ms=24)
                    else:  # pre matplotlib 2.0 compatibility
                        ax.plot(probvals[shortlist], testvals[shortlist],
                                c=markerlist[markerstep][:1])
                markerstep = markerstep + 1
                if markerstep >= len(markerlist):
                    markerstep = 0
    try:
        botrange = float(botrange)
        toprange = float(toprange)
    except Exception as myerr:
        print("could not convert ranges to float", myerr)

    if axesswap:
        ax.set_ylabel('Standard Deviation', fontsize=10)
        ax.set_xlabel(desiredtarget, fontsize=10)
        plt.xticks(np.arange(botrange, toprange, interval))
        minprob = int((minprob - 0.5) * 2) / 2
        plt.yticks(np.arange(minprob, 0 - minprob + 0.5, 0.5))
        plt.grid(b=True, which='both', color='0.65', linestyle='-')
        if logscale:
            ax.set_xscale('log')
        else:
            plt.xlim([botrange, toprange])
    else:
        ax.set_xlabel('Standard Deviation', fontsize=10)
        if not withboxplots:
            ax.set_ylabel(desiredtarget, fontsize=10)
            plt.yticks(np.arange(botrange, toprange, interval))
        else:
            seab_ax.set_ylim(botrange, toprange)
            ax.set_ylim(botrange, toprange)
        minprob = int((minprob - 0.5) * 2) / 2
        plt.xticks(np.arange(minprob, 0 - minprob + 0.5, 0.5))
        plt.grid(b=True, which='both', color='0.65', linestyle='-')
        if logscale:
            ax.set_yscale('log')
            ax.set_ylim(botrange, toprange)
            if withboxplots:
                seab_ax.set_yscale('log')
                seab_ax.set_ylim(botrange, toprange)
        else:
            try:
                plt.ylim([botrange, toprange])
            except Exception as myerr:
                print('probplot trendplot warning could not set range',
                      botrange, toprange, myerr)
    if setranges:
        if np.shape(setranges) == (2, 2):
            ax.set_ylim(setranges[1][0], setranges[1][1])
            ax.set_xlim(setranges[0][0], setranges[0][1])
            if withboxplots:
                seab_ax.set_ylim(setranges[1][0], setranges[1][1])
        else:
            print("Probplot ignoring setranges, wrong shape.")

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    return (thisplt, True, toprange, botrange, ppax)


if __name__ == '__main__':
    # this is just test data to verify functionality
    df = pd.read_csv(r'.\test.csv')
    # there are too many clusters in the test data, thin them out
    row_mask = []
    for row in df.itertuples():
        if row.Cluster[0] == '1':
            row_mask.append(True)
        else:
            row_mask.append(False)
    df = df[row_mask]
    probplot(df, 'Random', 'Cluster', withboxplots=True, connect_points=True)
