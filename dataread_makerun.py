##############################################################################
"""

 Dataread.py

 Create a Tk data summarization script

 By Ben Buck, Intel Corporation

"""
##############################################################################

import os
import sys
import re
# import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import itertools
import time

from scipy import stats
from matplotlib import gridspec
from statsmodels.formula.api import ols
from sklearn import neighbors

import tkinter as Tk
from tkinter import filedialog
# from PIL import Image
# from typing import Any

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, \
        NavigationToolbar2Tk
except ImportError:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, \
        NavigationToolbar2TkAgg
    NavigationToolbar2Tk = NavigationToolbar2TkAgg
from recently_updated_makegraphs_script import oneprobplot

"""
from recently_updated_yieldmosaic import yieldmosaic_function
from waterfallchart import waterfallchart
from pymaps import make_wafermap
from pydiemaps import make_wafermap as make_diemap
from clusterfind import clusterfind
"""


##############################################################################
def convert_to_epoch(trendcsv, dtcol='StartDateTime',
                     epochname='Unix_Epoch_Time',
                     pattern='%m/%d/%Y %I:%M:%S %p', newloc=3):
    if dtcol not in trendcsv.columns:
        print(">>> convert to epoch warninig: could not find your",
              "date column.  Will attempt to discover one.")
        for thiscol in trendcsv.columns:
            if 'date' in thiscol.lower():
                dtcol = thiscol
                break
    if dtcol not in trendcsv.columns:
        print(">>> convert to epoch error: could not find",
              "any sort of date column")
        return trendcsv
    trendcsv.dropna(subset=[dtcol], inplace=True)
    trendcsv.insert(value=range(len(trendcsv)), column='newdex',
                    loc=len(trendcsv.columns))
    trendcsv.set_index(['newdex'], inplace=True, drop=True, append=False)
    del trendcsv.index.name

    datecol = trendcsv.loc[:, dtcol]
    patterntries = ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                    "%Y/%m/%d %I:%M:%S %p"]
    for patterntry in patterntries:
        for tconvert in datecol:
            try:
                epoch = int(time.mktime(time.strptime(str(tconvert), pattern)))
            except ValueError:
                pattern = patterntry
            break
    epochcollist = []
    for tconvert in datecol:
        epoch = int(time.mktime(time.strptime(str(tconvert), pattern)))
        epochcollist.append(epoch)
    epochcol = pd.Series(epochcollist)
    try:
        trendcsv.insert(value=epochcol, column=epochname, loc=newloc)
    except ValueError as E:
        print(E)
    return trendcsv


##############################################################################
def colorgen():
    """
    on an infinite loop, yield as large a list as possible of distinguishable
    colors so that if we have a ton of bins, we have a ton of colors to
    represent them.  This is finite, so it can loop back if we still have bins
    and run out of colors I'm not using this elsewhere, but a good color cycle
    might have further applications completely apart from this script...
    but note that blue is missing (because it's reserved for bin dot.)
    """

    facecolors = ['#4C8C2C', '#8C4C2C', '#2C8C4C', '#2C4C8C', '#8C2C4C',
                  '#4C2C8C', 'orange', 'red', 'green', 'yellow', 'brown',
                  '#80FF10', 'cyan', 'magenta', 'olive', 'lightsalmon',
                  'teal', 'darkgreen', 'lightgreen', 'orangered', "#99CCFF",
                  "#CC99FF", "#FF99CC", "#FFCC99", "#CCFF99", "#99FFCC",
                  '#4CFFCC', '#FF4CCC', '#CC6CFF', '#FFCC4C', '#4CCCFF',
                  '#7af9ab', '#8C99FF', "#2CEF6C", '#FF8C99', '#2CCCFF',
                  '#CC2CFF', '#2CFFCC', '#CCFF2C', '#FFEE1C', '#FF00DD',
                  '#FFDCDC', '#2CFF0C', "#6C2CFF", "#6CFF2C", "#FF6C2C",
                  "#FF2C6C", "#2C6CFF", '#8CFF99',  # xkcd colors
                  '#014600', '#028f1e', '#0485d1', '#089404', '#0f9b8e',
                  '#1e9167', '#276ab3', '#343837', '#3c73a8', '#464196',
                  '#4f9153', '#58bc08', '#5fa052', '#653700', '#6a6e09',
                  '#706c11', '#75b84f']

    for facecolor in itertools.cycle(facecolors):
        yield facecolor


##############################################################################
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
        mysplittab = tograph[tograph.apply(
            lambda row: row[spcol] == mysplit, axis=1)]
        if np.issubdtype(mysplittab[xcol].dtype, np.datetime64):
            plt.plot_date(mysplittab[xcol], mysplittab[ycol],
                          c=color, label=mysplit)
        else:
            mysplittab.plot.scatter(xcol, ycol, c=color, label=mysplit, ax=ax)
        if dotrendline:
            mytrendlines = mysplittab.groupby(xcol).mean().reset_index()
            ax.plot(mytrendlines[xcol], mytrendlines[ycol], '-',
                    c=color, label='')
        if linearfit:
            mysplittabd = mysplittab[[xcol, ycol]].dropna()
            slope, intercept, r_value, _, _ = stats.linregress(
                mysplittabd[xcol], mysplittabd[ycol])
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
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1),
              borderaxespad=0., fontsize=10)
    if dologscale:
        ax.set_yscale('log')
    return fig, ax


##############################################################################
class table_registry(object):
    """
    Keep a list of all open tables and their names
    """

    def __init__(self):
        self.table_list = []
        self.default_names = 0
        self.table_handles = {}

    def get_new_table_name(self):
        """
        Get the next available unique table name
        """
        newname = "dataread_table_number_" + str(self.default_names)
        self.default_names += 1
        self.table_list.append(newname)
        return newname

    def rename_table(self, old_name, new_name, table_reference=None):
        """

        Set a new table name.  Raise KeyError if the new name is
        already in the list.
        Raise ValueError if the table does not exist to rename.

        """
        foundit = False
        for table in self.table_list:
            if table == new_name:
                # check if the old name is referenced.  If not, add it
                if table_reference is not None and \
                  old_name not in self.table_handles:
                    self.table_handles[old_name] = table_reference
                raise KeyError("Dataread: there is already" +
                               " a table with that name.")
        for counter, table in enumerate(self.table_list):
            if table == old_name:
                foundit = True
                self.table_list[counter] = new_name
                break
        if not foundit:
            raise ValueError("Dataread: attempt to rename " +
                             "a table that does not exist.")
        self.table_handles[old_name] = None
        try:
            del self.table_handles[old_name]
        except KeyError:
            pass
        if table_reference is not None:
            self.table_handles[new_name] = table_reference

    def joinable_tables(self):
        """

        Return a list of other tables for joining

        """
        outlist = []
        for mykey in sorted(self.table_handles):
            if self.table_handles[mykey] is not None:
                outlist.append(mykey)
        return outlist


##############################################################################
class dashclass(Tk.Frame):
    """

    Creates a dashboard window
    Also creates a child dashboard window from the first one
    Acts as a scrolling reader for a .csv file (imported via Pandas)

    Inputs:
    filein         -- Pandas Dataframe with overall lot summary
    basedir        -- scratch location for saved graphs
    basefile       -- name of the file in filein
    parentchild    -- initial invocation should be "parent"
    pythonwin      -- handle to the Pythonwin class
    table_registry -- handle to the table_registry class
    table_name     -- the current name of the table this class is representing
    master         -- Tk master for window drawing
                      (for "parent" assumed to be root)
    """

    def __init__(self, filein, basedir, basefile, parentchild, pythonwin,
                 table_registry, table_name, master=None):
        Tk.Frame.__init__(self, master)
        self.grid()
        self.master = master
        self.filein = filein
        self.graphtab = filein
        self.basedir = basedir
        self.basefile = basefile
        self.pythonwin = pythonwin
        self.parentchild = parentchild
        self.table_name = table_name
        self.table_registry = table_registry

        self.runningmain = 1
        self.leftheader = 1
        if parentchild == "parent":
            self.leftheader = 1
        if self.leftheader > len(self.filein.columns):
            self.leftheader = 1
        self.screensize = 24
        self.imin = 0
        self.jmin = 0
        self.jmax = 8
        self.jarr = []
        cellwidth = 25

        # menu system
        self.menuBar = Tk.Menu(self.master)
        filemenu = Tk.Menu(self.menuBar, tearoff=0)
        filemenu.add_command(label='Open', command=self.fileopen)
        filemenu.add_command(label='Save As', command=self.filesave)
        filemenu.add_command(label='Quit', command=self.canceled)
        self.menuBar.add_cascade(label='File', menu=filemenu)
        pandamenu = Tk.Menu(self.menuBar, tearoff=0)
        pandamenu.add_command(label='Summarize', command=self.summarize)
        pandamenu.add_command(label='Row Subset', command=self.rowsubset)
        pandamenu.add_command(label='Col Subset', command=self.colsubset)
        pandamenu.add_command(label='Join', command=self.tablejoin)
        pandamenu.add_command(label='Stack', command=self.meltmenu)
        pandamenu.add_command(label='Pivot', command=self.pivotmenu)
        pandamenu.add_command(label='Concatenate', command=self.concatmenu)
        pandamenu.add_command(label='Unique Values', command=self.uniquemenu)
        self.menuBar.add_cascade(label='New Table Ops', menu=pandamenu)
        ppandamenu = Tk.Menu(self.menuBar, tearoff=0)
        ppandamenu.add_command(label='Sort', command=self.sortmenu)
        ppandamenu.add_command(label='Add Column', command=self.addcol)
        ppandamenu.add_command(label='Drop Missing', command=self.dropmenu)
        ppandamenu.add_command(label='Rename Column', command=self.renamecol)
        ppandamenu.add_command(label='Mass Col Rename',
                               command=self.massrename)
        ppandamenu.add_command(label='Rename This Table',
                               command=self.newname)
        ppandamenu.add_command(label='Create Time Column',
                               command=self.timecol)
        self.menuBar.add_cascade(label='Current Table Ops', menu=ppandamenu)
        mathmenu = Tk.Menu(self.menuBar, tearoff=0)
        mathmenu.add_command(label='F-Oneway', command=self.anovamenu)
        mathmenu.add_command(label='Mass R^2', command=self.rsqmenu)
        mathmenu.add_command(label='Detect Outliers', command=self.lofmenu)
        mathmenu.add_command(label='Detect 50/50', command=self.fiftyfiftymenu)
        mathmenu.add_command(label='Make Cluster Maps', command=self.clusmenu)
        self.menuBar.add_cascade(label='Analysis', menu=mathmenu)
        plotmenu = Tk.Menu(self.menuBar, tearoff=0)
        plotmenu.add_command(label='Probplot', command=self.probselect)
        plotmenu.add_command(label='WaferMap', command=self.wafermapselect)
        plotmenu.add_command(label='FailbinMap', command=self.fbmapselect)
        plotmenu.add_command(label='Scatterplot', command=self.scatterplot)
        plotmenu.add_command(label='Fit Linear Model', command=self.mlm)
        plotmenu.add_command(label='Waterfall Chart', command=self.watermenu)
        plotmenu.add_command(label='Yield Mosaic', command=self.ymmenu)
        plotmenu.add_command(label='Facet Grid - Cols',
                             command=self.facetcolsmenu)
        plotmenu.add_command(label='Facet Grid - RowCol',
                             command=self.facetgridmenu)
        self.menuBar.add_cascade(label='Plot', menu=plotmenu)
        pythmenu = Tk.Menu(self.menuBar, tearoff=0)
        pythmenu.add_command(label='Pandas Command Updating This Table',
                             command=self.directpandasmenu)
        pythmenu.add_command(label='Pandas Command Creating New Table',
                             command=self.newpandasmenu)
        pythmenu.add_command(label='Python Command Creating a Plot',
                             command=self.arbplotmenu)
        self.menuBar.add_cascade(label='Direct Python', menu=pythmenu)
        scrollmenu = Tk.Menu(self.menuBar, tearoff=0)
        scrollmenu.add_command(label='Scroll Top', command=self.scrolltop)
        scrollmenu.add_command(label='Scroll Bottom', command=self.scrollbot)
        scrollmenu.add_command(label='Scroll Left Side',
                               command=self.scrollleftmost)
        scrollmenu.add_command(label='Scroll Right Side',
                               command=self.scrollrightmost)
        scrollmenu.add_command(label='Scroll to Column',
                               command=self.scrollcolmenu)
        scrollmenu.add_command(label='Scroll To Value',
                               command=self.scrollvalmenu)
        self.menuBar.add_cascade(label='Scroll', menu=scrollmenu)
        self.master.config(menu=self.menuBar)

        screen_width = master.winfo_screenwidth()
        if screen_width > 1900:
            self.jmax = 9
        elif screen_width < 1050:
            self.jmax = 7
            cellwidth = 15
        elif screen_width < 1300:
            cellwidth = 18

        self.cellwidth = cellwidth
        self.cellwidthmpad = int(cellwidth * 4/5)

        if self.jmax > len(self.filein.columns):
            self.jmax = len(self.filein.columns)
        if self.screensize > len(self.filein):
            self.screensize = len(self.filein)

        self.imax = self.screensize

        self.screentabs = self.jmax

        # set up mouse-over name expansion capability
        self.helptext = Tk.StringVar()
        self.helptext.set("")
        self.helplabel = Tk.Label(master=master, textvariable=self.helptext)
        self.helplabel.grid(columnspan=3, rowspan=3, column=7,
                            row=self.screensize + 7)

        # rows of headers
        self.mysub = [[], [], []]
        self.topline = []
        for jndex in range(self.jmin, self.jmax):
            while len(self.mysub[0]) < jndex + 1:
                self.mysub[0].append("")
                self.mysub[1].append("")
                self.mysub[2].append("")
            self.mysub[0][jndex] = Tk.StringVar()
            self.mysub[0][jndex].set(
                    self.filein.columns[jndex][:self.cellwidthmpad])
            self.topline.append(Tk.Entry(master=master,
                                         textvariable=self.mysub[0][jndex],
                                         width=cellwidth,
                                         readonlybackground="",
                                         state="readonly"))
            self.topline[len(self.topline) - 1] \
                .grid(row=2, column=(jndex - self.jmin + 3))
            self.mysub[1][jndex] = Tk.StringVar()
            self.mysub[1][jndex].set(
                self.filein.columns[jndex]
                [self.cellwidthmpad:self.cellwidthmpad * 2])
            self.topline.append(Tk.Entry(master=master,
                                         textvariable=self.mysub[1][jndex],
                                         width=cellwidth,
                                         readonlybackground="",
                                         state="readonly"))
            self.topline[len(self.topline) - 1] \
                .grid(row=3, column=(jndex - self.jmin + 3))
            self.mysub[2][jndex] = Tk.StringVar()
            self.mysub[2][jndex].set(
                    self.filein.columns[jndex][self.cellwidthmpad * 2:])
            self.topline.append(Tk.Entry(master=master,
                                         textvariable=self.mysub[2][jndex],
                                         width=cellwidth,
                                         readonlybackground="",
                                         state="readonly"))
            self.topline[len(self.topline) - 1] \
                .grid(row=4, column=(jndex - self.jmin + 3))

        # rows of entries
        kndex = 0
        self.entryhash = {}
        self.colorcycle = {}
        self.colorlist = colorgen()
        self.screendraw = []
        self.roback = []
        self.thisentry = []
        for index in range(self.imin, self.imax):
            lldex = 0
            self.screendraw.append([])
            self.roback.append([])
            self.thisentry.append([])
            for jndex in range(self.jmin, self.jmax):
                self.screendraw[kndex].append(Tk.StringVar())
                screendrawref = self.filein.iloc[index, jndex]
                if pd.isnull(screendrawref):
                    screendrawref = ""
                self.screendraw[kndex][lldex].set(screendrawref)
                self.roback[kndex].append("")

                # create the entry
                self.thisentry[kndex] \
                    .append(Tk.Entry(master=master,
                            textvariable=self.screendraw[kndex][lldex],
                            width=cellwidth,
                            readonlybackground=self.roback[kndex][lldex],
                            state="readonly"))
                self.thisentry[kndex][lldex] \
                    .grid(row=(index - self.imin + 5),
                          column=(jndex - self.jmin + 3))
                # add bindings for mouseover text
                self.thisentry[kndex][lldex] \
                    .bind('<Enter>',
                          lambda x, msg=self.filein.iloc[index, jndex]:
                          self.genericmsg(x, msg))
                self.thisentry[kndex][lldex] \
                    .bind('<FocusIn>',
                          lambda x, msg=self.filein.iloc[index, jndex]:
                          self.genericmsg(x, msg))
                self.thisentry[kndex][lldex] \
                    .bind('<Leave>',
                          lambda x, msg="": self.genericmsg(x, msg))
                lldex += 1
            kndex += 1

        self.swrlist = []
        self.listboxen = []
        self.plusbuttons = []
        self.swrvar = []

        # scroll and quit buttons
        self.scrollupbutton = Tk.Button(master=master, text="Up",
                                        command=self.scrollup)
        self.scrollupbutton.grid(column=4, row=self.screensize + 7)
        self.scrolldownbutton = Tk.Button(master=master, text="Down",
                                          command=self.scrolldown)
        self.scrolldownbutton.grid(column=4, row=self.screensize + 8)
        self.scrollleftbutton = Tk.Button(master=master, text="Left",
                                          command=self.scrollleft)
        self.scrollleftbutton.grid(column=3, row=self.screensize + 7)
        self.scrollrightbutton = Tk.Button(master=master, text="Right",
                                           command=self.scrollright)
        self.scrollrightbutton.grid(column=5, row=self.screensize + 7)

        if self.parentchild == "parent":
            self.quitbutton = Tk.Button(master=master, text="Quit",
                                        command=self.canceled)
        else:
            self.quitbutton = Tk.Button(master=master, text="Close",
                                        command=lambda masterref=master:
                                        self.close_win(masterref))
        if self.screentabs > 4:
            self.quitbutton.grid(column=self.screentabs + 2,
                                 row=self.screensize + 7)
        else:
            self.quitbutton.grid(column=6, row=self.screensize + 7)

        # Row x Col indicator
        self.rowcol = Tk.StringVar()
        self.rowcol.set("Shape: " + str(self.filein.shape))
        self.rowcolind = Tk.Label(master=master, textvariable=self.rowcol)
        self.rowcolind.grid(column=3, row=self.screensize + 8)

        # This default is B27A January 2018
        self.waterfallorder = ["H", "i", "I", "KA", "Ka", "KE", "Ke", "TS",
                               "TR", "bb", "Ts", "TT", "Tu", "Tt", "TF", "TB",
                               "RD", "RC", "RF", "RR", "RP", "RM", "RL", "RS",
                               "RT", "ST", "SP", "SI", "SJ", "op", "IS", "Rp",
                               "Rq", "Rk", "RX", "Rx", "RV", "RG", "Da", "mS",
                               "ms", "nD", "Md", "MD", "Mb", "mD", "md", "XD",
                               "ME", "nA", "nC", "nc", "nR", "XN", "MN", "JD",
                               "dU", "dD", "di", "JW", "dd", "RW", "Rw", "Rr",
                               "qm", "qj", "qn", "qk", "qo", "qa", "qb", "qi",
                               "qx", "qq", "qy", "qh", "qz", "qc", "qs", "Hd",
                               "Hi", "N", "ks", "kd", "BA", "Ba", "BC", "Bc",
                               "Jx", "dx", "Jd", "dW", "dV", "dt", "Jw", "dR",
                               "kt", "ke", "BT", "BU", "B", "b", "kT", "kE",
                               "Eu", "XE", "EA", "ED", "FD", "EE", "FE", "Ed",
                               "Fd", "Ee", "Fe", "be", "BE", "ZB", "fc", "XF",
                               "Mf", "nE", "ne", "l", "Ox", "OZ", "OX", "O",
                               "OA", "OB", "OC", "OI", "RU", "Ru", "RQ", "qM",
                               "qJ", "qN", "qK", "qO", "qA", "qB", "qI", "qX",
                               "qQ", "qY", "qF", "qH", "qZ", "qC", "qS", "qP",
                               "qR", "qE", "KO", "Ko", "KI", "Ki", "mX", "mx",
                               "nF", "nf", "mY", "my", "XY", "MF", "na", "Ot",
                               "OF", "OU"]

    #########################################################################
    def listbox_and_scrollbar(self, parentform, numericcols, column=0, row=1,
                              maxlen=10, multiselect=False):
        """
        scrollbar decoding found in "Programming Python, Fourth Edition,
        by Mark Lutz (O'Reilly).  Copyright 2011 Mark Lutz, 978-0-576-15810-1"
        pg. 522-523
        """
        leftscatterform = Tk.Frame(parentform)
        leftscatterform.grid(column=column, row=row)
        sbar = Tk.Scrollbar(leftscatterform, orient=Tk.VERTICAL)
        myselectmode = Tk.BROWSE
        if multiselect:
            myselectmode = Tk.EXTENDED
        if maxlen > 40:
            maxlen = 40
            xlistbox = Tk.Listbox(leftscatterform, selectmode=myselectmode,
                                  width=int(maxlen * 1.2),
                                  exportselection=False,
                                  relief="sunken")
            hsbar = Tk.Scrollbar(leftscatterform, orient=Tk.HORIZONTAL)
            hsbar.config(command=xlistbox.xview)
            xlistbox.config(xscrollcommand=hsbar.set)
            hsbar.grid(column=0, row=1, sticky=Tk.W+Tk.E)
        else:
            xlistbox = Tk.Listbox(leftscatterform, selectmode=myselectmode,
                                  width=int(maxlen * 1.2),
                                  exportselection=False,
                                  relief="sunken")
        sbar.config(command=xlistbox.yview)
        xlistbox.config(yscrollcommand=sbar.set)
        for col in numericcols:
            xlistbox.insert("end", col)
        sbar.grid(column=1, row=0, sticky=Tk.N+Tk.S)
        xlistbox.grid(column=0, row=0)
        return xlistbox

    def summarize(self):
        """
        create a new table that is a (pandas) summary of the current table
        (this part is the form creator)
        """
        summarizewindow = Tk.Toplevel(self)
        summarizewindow.wm_title("Select the summary type " +
                                 "to apply to the table")
        sumrform = Tk.Frame(summarizewindow)
        sumrform.pack()
        label1 = Tk.Label(sumrform, text="Select the column(s) to group by  ")
        label1.grid(row=0, column=0)
        label1 = Tk.Label(sumrform, text="  Select the summary statistic")
        label1.grid(row=0, column=1)
        maxlen = 10
        for col in self.filein.columns:
            if len(col) > maxlen:
                maxlen = len(col)
        entergpcols = self.listbox_and_scrollbar(sumrform, self.filein.columns,
                                                 column=0, row=1,
                                                 maxlen=maxlen,
                                                 multiselect=True)

        summarytypes = [
            "Median",
            "Mean",
            "Min",
            "Max",
            "Sum",
            "StdDev",
            "Count",
            "Quantile"
        ]
        sumstat = self.listbox_and_scrollbar(sumrform, summarytypes,
                                             column=1, row=1)
        quanval = Tk.StringVar()
        sumbutton = Tk.Button(summarizewindow, text="Summarize")
        sumbutton["command"] = lambda summarizewindow=summarizewindow, \
            entergpcols=entergpcols, sumstat=sumstat, \
            quanval=quanval: \
            self.summarizego(summarizewindow, entergpcols, sumstat, quanval)
        sumbutton.pack(side="bottom")
        quanform = Tk.Frame(summarizewindow)
        quanlabel = Tk.Label(master=quanform, text="Quantile Value (0-100):  ")
        quanlabel.grid(column=0, row=0)
        quanvalen = Tk.Entry(master=quanform, textvariable=quanval, width=10)
        quanvalen.grid(column=1, row=0)
        quanform.pack(side="bottom")

    def summarizego(self, summarizewindow, entergpcols, sumstat, quanval):
        """
        create a summary table based on the input from the form
        """
        pregpcols = entergpcols.curselection()
        gpcols = []
        for pregpcol in pregpcols:
            gpcols.append(entergpcols.get(pregpcol))
        presumstat = sumstat.curselection()
        worked = False
        try:
            sumstat = sumstat.get(presumstat[0])
            worked = True
        except Exception:
            self.tkprint("You did not select a summary stat.",
                         sys.exc_info()[0], sys.exc_info()[1])
        if worked:
            gostat = sumstat
            if sumstat == "StdDev":
                gostat = "std"
            worked = False
            self.pythonwin.new_code("gpcols = [\n    \"" +
                                    "\",\n    \"".join(gpcols) + "\"\n]")
            if sumstat == "Quantile":
                quanval = quanval.get()
                try:
                    self.subtable = self.filein.groupby(gpcols). \
                        quantile(float(quanval) / 100).reset_index()
                    self.pythonwin.new_code(
                            "subtable = datatables[\"" + self.table_name +
                            "\"].groupby(gpcols).quantile(float("
                            + str(quanval) + ") / 100).reset_index()")
                    self.pythonwin.new_code(
                            "colnames = []\nfor col in " +
                            "subtable.columns:\n    if col not in gpcols:\n" +
                            "        colnames.append(\"Quantile_" +
                            str(quanval) + "_of_\" + col)")
                    self.pythonwin.new_code(
                            "    else:\n        colnames." +
                            "append(col)\nsubtable.columns = colnames")
                    worked = True
                    colnames = []
                    for col in self.subtable.columns:
                        if col not in gpcols:
                            colnames.append("Quantile_" + str(quanval) +
                                            "_of_" + col)
                        else:
                            colnames.append(col)
                    self.subtable.columns = colnames
                except Exception as myerr:
                    self.tkprint("Could not take summary,", myerr,
                                 sys.exc_info()[0], sys.exc_info()[1])
            else:
                try:
                    self.subtable = self.filein.groupby(gpcols).agg(
                            gostat.lower()).reset_index()
                    self.pythonwin.new_code(
                            "subtable = datatables[\"" + self.table_name +
                            "\"].groupby(gpcols).agg(\"" +
                            gostat.lower() + "\").reset_index()")
                    self.pythonwin.new_code(
                            "colnames = []\nfor col in " +
                            "subtable.columns:\n    if col not in gpcols:")
                    self.pythonwin.new_code(
                            " " * 8 + "colnames.append(\"" +
                            sumstat + "_of_\" + col)")
                    self.pythonwin.new_code(
                            "    else:\n" + " " * 8 + "colnames"
                            ".append(col)\nsubtable.columns = colnames")
                    worked = True
                    colnames = []
                    for col in self.subtable.columns:
                        if col not in gpcols:
                            colnames.append(sumstat + "_of_" + col)
                        else:
                            colnames.append(col)
                    self.subtable.columns = colnames
                except Exception as myerr:
                    self.tkprint("Could not take summary,", myerr,
                                 sys.exc_info()[0], sys.exc_info()[1])
            if worked:
                summarizewindow.destroy()
                window = Tk.Toplevel()
                pnewname = self.table_registry.get_new_table_name()
                newname = "Summary Table " + sumstat + " of " + self.table_name
                try:
                    self.table_registry.rename_table(pnewname,
                                                     newname, self.subtable)
                except KeyError:
                    newname = pnewname
                window.title(newname)
                self.pythonwin.new_code("datatables[\"" + newname +
                                        "\"] = subtable")
                dashclass(self.subtable, self.basedir, basefile, "child",
                          self.pythonwin, master=window,
                          table_registry=self.table_registry,
                          table_name=newname)

    def probselect(self):
        """
        create a query window of two column list boxes to probplot
        """
        listboxprobwindow = Tk.Toplevel(self)
        listboxprobwindow.wm_title("Select the X and Y Columns to Probplot")
        numericcols = []
        maxlen = 10
        nummaxlen = 10
        for col in self.filein.columns:
            if self.filein[col].dtype == np.float64 or \
              self.filein[col].dtype == np.int64:
                numericcols.append(col)
                if len(col) > nummaxlen:
                    nummaxlen = len(col)
            if len(col) > maxlen:
                maxlen = len(col)
        label1 = Tk.Label(listboxprobwindow,
                          text="Select the X and Y Columns to Probplot")
        label1.pack(side="top")
        form1 = Tk.Frame(listboxprobwindow)
        form1.pack()
        label2 = Tk.Label(form1, text="X Column:")
        label2.grid(row=0, column=0)

        xlistboxprob = self.listbox_and_scrollbar(form1, self.filein.columns,
                                                  column=0, row=1,
                                                  maxlen=maxlen)
        ylistboxprob = self.listbox_and_scrollbar(form1, numericcols,
                                                  column=1, row=1,
                                                  maxlen=nummaxlen,
                                                  multiselect=True)

        label3 = Tk.Label(form1, text="Y Column:")
        label3.grid(row=0, column=1)
        button1 = Tk.Button(form1, text="Use a List in a Table Instead")
        button1["command"] = lambda ylistboxprob=ylistboxprob: \
            self.colloader(ylistboxprob)
        button1.grid(row=2, column=1)
        scatterbutton = Tk.Button(listboxprobwindow, text="Probplot")
        scatterbutton["command"] = lambda listboxprobwindow=listboxprobwindow,\
            xlistboxprob=xlistboxprob, ylistboxprob=ylistboxprob: \
            self.plotchild(listboxprobwindow, xlistboxprob, ylistboxprob)
        scatterbutton.pack(side="bottom")

    def colloader(self, ylistboxprob):
        pretablelist = self.table_registry.joinable_tables()
        tablelist = []
        maxlen = 20
        for thistable in pretablelist:
            if self.table_name != thistable:
                tablelist.append(thistable)
                if len(thistable) > maxlen:
                    maxlen = len(thistable)
        if len(tablelist) < 1:
            self.tkprint("There are no other tables to" +
                         " draw column names out of.")
        else:
            colloaderwin = Tk.Toplevel()
            label1 = Tk.Label(colloaderwin,
                              text="Select a Data Table Containing a list of" +
                              " column names that you wish to graph in this " +
                              "table.")
            label1.grid(column=0, row=0)
            coltablesel = self.listbox_and_scrollbar(colloaderwin, tablelist,
                                                     column=0, row=1,
                                                     maxlen=maxlen)
            button1 = Tk.Button(colloaderwin, text="Continue")
            button1["command"] = lambda colloaderwin=colloaderwin, \
                ylistboxprob=ylistboxprob, coltablesel=coltablesel: \
                self.colloadertwo(colloaderwin, ylistboxprob, coltablesel)
            button1.grid(column=0, row=2)

    def colloadertwo(self, colloaderwin, ylistboxprob, coltablesel):
        try:
            mycoltable = coltablesel.get(coltablesel.curselection()[0])
            colloaderwintwo = Tk.Toplevel()
            label1 = Tk.Label(colloaderwintwo,
                              text="Select Column containing the list of " +
                              "column names to graph")
            label1.grid(column=0, row=0)
            othertable = self.table_registry.table_handles[mycoltable]
            maxlen = np.max([len(col) for col in othertable.columns])
            colloadersel = self.listbox_and_scrollbar(colloaderwintwo,
                                                      othertable.columns,
                                                      column=0, row=1,
                                                      maxlen=maxlen)
            button1 = Tk.Button(colloaderwintwo, text="Select Column")
            button1["command"] = lambda colloaderwintwo=colloaderwintwo, \
                ylistboxprob=ylistboxprob, colloadersel=colloadersel, \
                othertable=othertable: self.colloadergo(colloaderwintwo,
                                                        ylistboxprob,
                                                        colloadersel,
                                                        othertable)
            button1.grid(column=0, row=2)
            colloaderwin.destroy()
        except Exception as myerr:
            self.tkprint("Selecting a Data Table Containing " +
                         "a list of column names Exception:", myerr,
                         sys.exc_info()[0], sys.exc_info()[1])

    def colloadergo(self, colloaderwintwo, ylistboxprob,
                    colloadersel, othertable):
        try:
            colofcols = colloadersel.get(colloadersel.curselection()[0])
            indexhash = {}
            alllist = ylistboxprob.get(0, Tk.END)
            for counter, listval in enumerate(alllist):
                indexhash[listval] = counter
            mycols = othertable[colofcols].dropna().drop_duplicates().tolist()
            for mycol in mycols:
                if mycol in indexhash:
                    selectionindex = indexhash[mycol]
                    ylistboxprob.selection_set(selectionindex)
            ylistboxprob.focus()
            ylistboxprob.event_generate("<<ListboxSelect>>")
            colloaderwintwo.destroy()
        except Exception as myerr:
            self.tkprint("Selecting a Data Table Containing a list of" +
                         " column names Exception:", myerr,
                         sys.exc_info()[0], sys.exc_info()[1])

    def genericmsg(self, _, msg):
        """
        Update a generic message into the help text
        """
        if pd.isnull(msg):
            msg = ""
        msg = re.sub("\\&", "\n", str(msg))
        self.helptext.set(msg)

    def close_win(self, thiswin):
        """ close this window button function """
        thiswin.destroy()

    def canceled(self):
        """ quit entire script button function """
        exit()

    def scrollup(self):
        """ scroll data upwards """
        self.imin -= 8
        if self.imin < 0:
            self.imin = 0
        self.imax = self.imin + self.screensize
        self.doupdate()

    def scrolltop(self):
        """ scroll to the very top of the dataset """
        self.imin = 0
        self.imax = self.imin + self.screensize
        self.doupdate()

    def scrollbot(self):
        """ scroll to the bottom screen """
        self.imin = len(self.filein) - self.screensize
        self.imax = self.imin + self.screensize
        self.doupdate()

    def scrollleftmost(self):
        """ scroll to the left extent """
        self.jmin = 0
        self.jmax = self.jmin + self.screentabs
        self.doupdate()

    def scrollrightmost(self):
        """ scroll to the right extent """
        self.jmin = len(self.filein.columns) - self.screentabs
        if self.jmin < 0:
            self.jmin = 0
        self.jmax = self.jmin + self.screentabs
        self.doupdate()

    def scrolldown(self):
        """ scroll data downwards """
        self.imin += 8
        if self.imin > len(self.filein) - self.screensize:
            self.imin = len(self.filein) - self.screensize
        self.imax = self.imin + self.screensize
        self.doupdate()

    def scrollleft(self):
        """ scroll data left """
        self.jmin -= 8
        if self.jmin < 0:
            self.jmin = 0
        self.jmax = self.jmin + self.screentabs
        self.doupdate()

    def scrollright(self):
        """ scroll data right """
        self.jmin += 8
        if self.jmin > len(self.filein.columns) - self.screentabs:
            self.jmin = len(self.filein.columns) - self.screentabs
        if self.jmin < 0:
            self.jmin = 0
        self.jmax = self.jmin + self.screentabs
        self.doupdate()

    def doupdate(self):
        """ After a scroll, display the new data """
        kndex = 0
        lldex = 0

        # left headers
        for jndex in range(self.leftheader):
            self.mysub[0][kndex].set(
                    self.filein.columns[jndex][:self.cellwidthmpad])
            self.mysub[1][kndex].set(
                    self.filein.columns[jndex]
                    [self.cellwidthmpad:self.cellwidthmpad * 2])
            self.mysub[2][kndex].set(
                    self.filein.columns[jndex][self.cellwidthmpad * 2:])
            kndex += 1

        # we don't reset kndex here, it's shifted from the left header.

        # rest of the table headers
        for jndex in range(self.jmin + self.leftheader, self.jmax):
            self.mysub[0][kndex].set(
                    self.filein.columns[jndex][:self.cellwidthmpad])
            self.mysub[1][kndex].set(
                    self.filein.columns[jndex]
                    [self.cellwidthmpad:self.cellwidthmpad * 2])
            self.mysub[2][kndex].set(
                    self.filein.columns[jndex][self.cellwidthmpad * 2:])
            kndex += 1

        # and the rest of the table
        kndex = 0
        for index in range(self.imin, self.imax):
            lldex = 0
            # left side
            for jndex in range(self.leftheader):
                self.screendraw[kndex][lldex].set(
                        self.filein.iloc[index, jndex])
                self.roback[kndex][lldex] = ""
                self.thisentry[kndex][lldex].configure(
                        readonlybackground=self.roback[kndex][lldex])
                self.thisentry[kndex][lldex].update()
                lldex += 1

            # rest of table
            # deliberately not resetting lldex
            for jndex in range(self.jmin + self.leftheader, self.jmax):
                self.screendraw[kndex][lldex].set(
                        self.filein.iloc[index, jndex])
                # limits, color alike
                screendrawref = self.filein.iloc[index, jndex]
                if pd.isnull(screendrawref):
                    screendrawref = ""
                self.screendraw[kndex][lldex].set(screendrawref)
                self.roback[kndex][lldex] = ""

                self.thisentry[kndex][lldex].configure(
                        readonlybackground=self.roback[kndex][lldex])
                self.thisentry[kndex][lldex].update()
                self.thisentry[kndex][lldex].bind(
                        '<Enter>', lambda x,
                        msg=self.filein.iloc[index, jndex]:
                        self.genericmsg(x, msg))
                self.thisentry[kndex][lldex].bind(
                        '<FocusIn>', lambda x,
                        msg=self.filein.iloc[index, jndex]:
                        self.genericmsg(x, msg))
                self.thisentry[kndex][lldex].bind(
                        '<Leave>', lambda x,
                        msg="": self.genericmsg(x, msg))
                lldex += 1
            kndex += 1
        self.rowcol.set("Shape: " + str(self.filein.shape))

    def plotchild(self, listboxprobwindow, xlistboxprob, ylistboxprob):
        """ make a plot / react to the 'Plot' button """
        try:
            xval = xlistboxprob.get(xlistboxprob.curselection()[0])
            yvals = [ylistboxprob.get(ylbp) for ylbp in
                     ylistboxprob.curselection()]
            listboxprobwindow.destroy()
            graphtab = self.filein
            for yval in yvals:
                oneprobplot(graphtab, yval, self.basedir,
                            makethismap=False, keepplot=True, splitcol=xval)
                tablelen = len(graphtab[xval].unique())
                botval = 0.05 + 0.018 * tablelen
                # max table size in any case
                if botval > 0.45:
                    botval = 0.45
                try:
                    plt.subplots_adjust(bottom=botval)
                except ValueError:  # bottom cannot be greater than top
                    pass  # extreme tablelen
                fig = plt.gcf()
                probfigout = Tk.Toplevel()
                probfigout.wm_title("Probplot " + xval + " vs. " + yval)
                formrone = Tk.Frame(probfigout)
                canvas = FigureCanvasTkAgg(fig, master=formrone)
                canvas.draw()
                canvas._tkcanvas.pack(side="top", fill="both", expand=1)
                toolbar_frame_one = Tk.Frame(formrone)
                toolbarone = NavigationToolbar2Tk(canvas, toolbar_frame_one)
                toolbarone.update()
                toolbarone.grid(row=0, column=0, columnspan=2)
                toolbar_frame_one.pack()
                formrone.pack()
                plt.close(fig)  # otherwise, the next plt.show will show this
                self.pythonwin.new_code(
                        "oneprobplot(datatables[\"" + self.table_name +
                        "\"], \"" + str(yval) + "\", r\"" + self.basedir +
                        "\", makethismap=False, keepplot=True," +
                        " splitcol=\"" + str(xval) + "\")")
        except Exception as myerr:
            self.tkprint("Probplot Exception:", myerr,
                         sys.exc_info()[0], sys.exc_info()[1])

    def wafermapselect(self):
        """
        create a query window of two column list boxes for Wafer Mapping
        """
        self.listboxwmwindow = Tk.Toplevel(self)
        self.listboxwmwindow.wm_title("Select the Map and Split Columns")
        numericcols = []
        maxlen = 1
        for col in self.filein.columns:
            if self.filein[col].dtype == np.float64 or \
              self.filein[col].dtype == np.int64:
                numericcols.append(col)
                if len(col) > maxlen:
                    maxlen = len(col)
        label1 = Tk.Label(self.listboxwmwindow,
                          text="Select the Map and Split Columns\n" +
                          "X/Y columns should be DieX/DieY or Row/Col")
        label1.pack(side="top")
        self.formwm = Tk.Frame(self.listboxwmwindow)
        self.formwm.pack()
        # label2 = Tk.Label(self.form, text="X and Y Columns:")
        # label2.grid(row=0, column=0)

        # self.xlistboxwm = self.listbox_and_scrollbar(self.formwm,
        #    self.filein.columns, column=0, row=1, maxlen=maxlen,
        #    multiselect=True)
        self.ylistboxwm = self.listbox_and_scrollbar(self.formwm,
                                                     numericcols,
                                                     column=1, row=1,
                                                     maxlen=maxlen)
        self.zlistboxwm = self.listbox_and_scrollbar(self.formwm,
                                                     self.filein.columns,
                                                     column=0, row=1,
                                                     maxlen=maxlen)

        label3 = Tk.Label(self.formwm, text="Map Column:")
        label3.grid(row=0, column=1)
        label3 = Tk.Label(self.formwm, text="Split Column:")
        label3.grid(row=0, column=0)
        scatterbutton = Tk.Button(self.listboxwmwindow, text="Wafermap")
        scatterbutton["command"] = self.wafermapchild
        scatterbutton.pack(side="bottom")

    def wafermapchild(self):
        """
        create a wafermap
        """
        # wmxy = self.xlistboxwm.curselection()
        # wmx = self.xlistboxwm.get(wmxy[0])
        # wmy = self.xlistboxwm.get(wmxy[1])
        prewmmapcol = self.ylistboxwm.curselection()
        wmmapcol = self.ylistboxwm.get(prewmmapcol[0])
        prewmspcol = self.zlistboxwm.curselection()
        wmspcol = self.zlistboxwm.get(prewmspcol[0])
        self.listboxwmwindow.destroy()
        make_wafermap(self.filein, wmmapcol, False, wmspcol)
        self.pythonwin.new_code(
                "make_wafermap(datatables[\"" + self.table_name +
                "\"], \"" + str(wmmapcol) + "\", False, \"" +
                str(wmspcol) + "\")")
        for counter in plt.get_fignums():
            fig = plt.figure(counter)
            wmapfigout = Tk.Toplevel()
            wmapfigout.wm_title("WaferMap " + str(wmmapcol) + " by " +
                                str(wmspcol) + " Output Number " +
                                str(counter))
            formrone = Tk.Frame(wmapfigout)
            canvas = FigureCanvasTkAgg(fig, master=formrone)
            canvas.draw()
            canvas._tkcanvas.pack(side="top", fill="both", expand=1)
            toolbar_frame_one = Tk.Frame(formrone)
            toolbarone = NavigationToolbar2Tk(canvas, toolbar_frame_one)
            toolbarone.update()
            toolbarone.grid(row=0, column=0, columnspan=2)
            toolbar_frame_one.pack()
            formrone.pack()
            plt.close(fig)  # otherwise, the next plt.show will show this

    def fbmapselect(self):
        """
        create a Failbin map per WaferId, selecting columns
        """
        fbmapwin = Tk.Toplevel()
        label0 = Tk.Label(fbmapwin, text="DieX/DieY columns are assumed.")
        label0.grid(column=0, row=0)
        label1 = Tk.Label(fbmapwin, text="Select FailBin column  ")
        label1.grid(column=0, row=1)
        label2 = Tk.Label(fbmapwin, text="  Select WaferId column")
        label2.grid(column=1, row=1)
        maxlen = np.max([len(col) for col in self.filein.columns])
        fbmapsel = self.listbox_and_scrollbar(fbmapwin, self.filein.columns,
                                              column=0, row=2, maxlen=maxlen)
        wfmapsel = self.listbox_and_scrollbar(fbmapwin, self.filein.columns,
                                              column=1, row=2, maxlen=maxlen)
        button1 = Tk.Button(fbmapwin, text="Make FB Maps")
        button1["command"] = lambda fbmapwin=fbmapwin, fbmapsel=fbmapsel, \
            wfmapsel=wfmapsel: self.fbmapgo(fbmapwin, fbmapsel, wfmapsel)
        button1.grid(column=0, row=3)

    def fbmapgo(self, fbmapwin, fbmapsel, wfmapsel):
        """
        create a Failbin map per WaferId
        """
        fbcol = fbmapsel.get(fbmapsel.curselection()[0])
        wfridcol = wfmapsel.get(wfmapsel.curselection()[0])
        fbmapwin.destroy()
        make_diemap(self.filein, fbcol, False, wfridcol)
        self.pythonwin.new_code(
                "make_diemap(datatables[\"" + self.table_name +
                "\"], \"" + str(fbcol) + "\", False, \"" +
                str(wfridcol) + "\")")
        for counter in plt.get_fignums():
            fig = plt.figure(counter)
            wmapfigout = Tk.Toplevel()
            wmapfigout.wm_title("FbinMap " + str(fbcol) + " by " +
                                str(wfridcol) + " Output Number " +
                                str(counter))
            formrone = Tk.Frame(wmapfigout)
            canvas = FigureCanvasTkAgg(fig, master=formrone)
            canvas.draw()
            canvas._tkcanvas.pack(side="top", fill="both", expand=1)
            toolbar_frame_one = Tk.Frame(formrone)
            toolbarone = NavigationToolbar2Tk(canvas, toolbar_frame_one)
            toolbarone.update()
            toolbarone.grid(row=0, column=0, columnspan=2)
            toolbar_frame_one.pack()
            formrone.pack()
            plt.close(fig)  # otherwise, the next plt.show will show this

    def rowsubset(self):
        """
        create a new table with only selected rows
        """
        rowsubsetwindow = Tk.Toplevel(self)
        rowsubsetwindow.wm_title("Select the type of row subset to take")
        self.rsubrform = Tk.Frame(rowsubsetwindow)
        self.rsubrform.pack()
        label1 = Tk.Label(self.rsubrform,
                          text="Select the column to subset rows by  ")
        label1.grid(row=0, column=0)
        label2 = Tk.Label(self.rsubrform, text="  Select the subset type")
        label2.grid(row=0, column=1)
        label3 = Tk.Label(self.rsubrform, text="  Values to subset to")
        label3.grid(row=0, column=2)
        maxlen = 10
        for col in self.filein.columns:
            if len(col) > maxlen:
                maxlen = len(col)
        rowsubsetcol = self.listbox_and_scrollbar(self.rsubrform,
                                                  self.filein.columns,
                                                  column=0, row=1,
                                                  maxlen=maxlen)

        subsettypes = [
            "equals",
            "is in (comma-delimit)",
            "contains",
            "<",
            "<=",
            ">",
            ">="
        ]
        subsetstyle = self.listbox_and_scrollbar(self.rsubrform,
                                                 subsettypes, column=1,
                                                 row=1, maxlen=20)
        subsetvalue = Tk.StringVar()
        subvalen = Tk.Entry(master=self.rsubrform, textvariable=subsetvalue,
                            width=40)
        subvalen.grid(column=2, row=1)

        pythrow = Tk.IntVar(value=0)

        sumbutton = Tk.Button(rowsubsetwindow, text="Take a Row Subset")
        sumbutton["command"] = lambda rowsubsetwindow=rowsubsetwindow, \
            rowsubsetcol=rowsubsetcol, subsetstyle=subsetstyle, \
            subsetvalue=subsetvalue, pythrow=pythrow: self.rowsubsetgo(
                    rowsubsetwindow, rowsubsetcol, subsetstyle, subsetvalue,
                    pythrow)
        sumbutton.pack(side="bottom")
        pythrowbox = Tk.Checkbutton(rowsubsetwindow,
                                    text="At your own risk," +
                                    " enter a Python command on 'mytable' " +
                                    "instead of entering values to subset " +
                                    "to\n (e.g. \"(mytable.Split == '01C') " +
                                    "& (mytable.WaferId == '0123-04')\")",
                                    variable=pythrow)
        pythrowbox.pack(side="bottom")

    def rowsubsetgo(self, rowsubsetwindow, rowsubsetcol,
                    subsetstyle, subsetvalue, pythrow):
        """

        With valid input, finish making the row subset

        """
        ssval = subsetvalue.get()
        try:
            if pythrow.get() == 1:
                mytable = self.filein
                # clear spyder warning
                mytable
                self.subtable = eval("self.filein[" + ssval + "]")
            else:
                prestyle = subsetstyle.curselection()
                sstyle = subsetstyle.get(prestyle[0])
                prerscol = rowsubsetcol.curselection()
                rscol = rowsubsetcol.get(prerscol[0])
                if sstyle == "equals":
                    self.subtable = self.filein[
                            self.filein.apply(lambda row:
                                              str(row[rscol]) ==
                                              str(ssval), axis=1)]
                    self.pythonwin.new_code(
                            "subtable = self.filein[" + "datatables[\"" +
                            self.table_name + "\"].apply(" + "lambda row: " +
                            "str(row[\"" + str(rscol) + "\"]) == \"" +
                            str(ssval) + "\", axis=1)]")
                elif 'is in' in sstyle:
                    isinopts = subsetvalue.get().split(",")
                    self.subtable = self.filein[
                            self.filein[rscol].isin(isinopts)]
                    self.pythonwin.new_code(
                        "subtable = self.filein[datatables[\"" +
                        self.table_name + "\"][\"" + str(rscol) +
                        "\"].isin([\"" + "\",\"".join(isinopts) + "\"])]")
                elif sstyle == "contains":
                    self.subtable = self.filein[
                        self.filein.apply(lambda row: str(ssval) in
                                          str(row[rscol]), axis=1)]
                    self.pythonwin.new_code(
                        "subtable = datatables[\"" + self.table_name +
                        "\"][datatables[\"" + self.table_name +
                        "\"].apply(lambda row: \"" + str(ssval) +
                        "\" in str(row[\"" + str(rscol) + "\"]), axis=1)]")
                else:
                    mytable = self.filein
                    # clear spyder warning
                    mytable
                    self.subtable = eval("self.filein[self.filein.apply(" +
                                         "lambda row: row['" + rscol + "'] " +
                                         sstyle + " float(" + str(ssval) +
                                         "), axis=1)]")
                    self.pythonwin.new_code(
                        "subtable = datatables[\"" + self.table_name +
                        "\"][datatables[\"" + self.table_name +
                        "\"].apply(lambda row: row['" + rscol +
                        "'] " + sstyle + " float(" + str(ssval) +
                        "), axis=1)]")
            rowsubsetwindow.destroy()
            pnewname = self.table_registry.get_new_table_name()
            newname = "Subset Table " + self.table_name
            try:
                self.table_registry.rename_table(pnewname, newname,
                                                 self.subtable)
            except KeyError:
                newname = pnewname
            window = Tk.Toplevel()
            window.wm_title(newname)
            dashclass(self.subtable, self.basedir, basefile, "child",
                      self.pythonwin, master=window,
                      table_registry=self.table_registry, table_name=newname)
            self.pythonwin.new_code("datatables[\"" + newname +
                                    "\"] = subtable")
        except Exception as myerr:
            self.tkprint("I could not take a row subset due to an error.",
                         myerr, sys.exc_info()[0], sys.exc_info()[1])

    def newname(self):
        tablenamewindow = Tk.Toplevel(self)
        label1 = Tk.Label(tablenamewindow, text="New Table Name:")
        label1.pack()
        newtablename = Tk.StringVar()
        entry1 = Tk.Entry(tablenamewindow, textvariable=newtablename, width=60)
        entry1.pack()
        button1 = Tk.Button(tablenamewindow, text="Set Name")
        button1["command"] = lambda tablenamewindow=tablenamewindow, \
            newtablename=newtablename: self.newnamego(tablenamewindow,
                                                      newtablename)
        button1.pack()

    def newnamego(self, tablenamewindow, newtablename):
        if len(newtablename.get()) > 0:
            try:
                oldname = self.table_name
                self.table_registry.rename_table(self.table_name,
                                                 newtablename.get(),
                                                 self.filein)
                self.table_name = newtablename.get()
                self.master.wm_title("Table: " + self.table_name)
                tablenamewindow.destroy()
                self.pythonwin.new_code(
                    "datatables[\"" + self.table_name + "\"] = datatables[\""
                    + oldname + "\"]")
            except KeyError:
                self.tkprint("There is already a table with that name." +
                             "  Choose another please.")

    def fileopen(self):
        filetypes = (("CSV files", "*.csv"), ("All files", "*.*"))
        basefile = filedialog.askopenfilename(initialdir='.',
                                              filetypes=filetypes)
        forcedtype = {
            "Lot": str,
            "Lot Id": str,
            "LotId": str,
            "LOTID": str
        }
        self.subtable = pd.read_csv(basefile, dtype=forcedtype,
                                    low_memory=False)
        newtablename = self.table_registry.get_new_table_name()
        path, myname = os.path.split(basefile)
        try:
            self.table_registry.rename_table(newtablename, myname,
                                             table_reference=self.subtable)
        except KeyError:
            myname = newtablename
        tkroot = Tk.Toplevel()
        tkroot.wm_title("Table: " + myname)
        self.pythonwin.new_code(
            "datatables[\"" + myname + "\"] = pd.read_csv(\"" + basefile +
            "\", low_memory=False)")
        dashclass(self.subtable, self.basedir, basefile, "child",
                  self.pythonwin, master=tkroot,
                  table_registry=self.table_registry, table_name=myname)

    def tablejoin(self):
        pretablelist = self.table_registry.joinable_tables()
        tablelist = []
        maxlen = 20
        for thistable in pretablelist:
            if self.table_name != thistable:
                tablelist.append(thistable)
                if len(thistable) > maxlen:
                    maxlen = len(thistable)
        tablejoinmenuone = Tk.Toplevel(self)
        if len(tablelist) < 1:
            label1 = Tk.Label(tablejoinmenuone,
                              text="There are no other tables.  Please open" +
                              " or create something to join with.")
            label1.pack()
            button1 = Tk.Button(tablejoinmenuone, text="Close")
            button1["command"] = lambda: tablejoinmenuone.destroy()
            button1.pack()
        else:
            label1 = Tk.Label(tablejoinmenuone,
                              text="Select the Other Table for this Join.")
            label1.grid(column=0, row=0)
            tablesel = self.listbox_and_scrollbar(tablejoinmenuone,
                                                  tablelist, column=0, row=1,
                                                  maxlen=maxlen)
            button1 = Tk.Button(tablejoinmenuone, text="Continue")
            button1["command"] = lambda tablejoinmenuone=tablejoinmenuone, \
                tablesel=tablesel: self.tablejointwo(tablejoinmenuone,
                                                     tablesel)
            button1.grid(column=0, row=2)

    def tablejointwo(self, tablejoinmenuone, tablesel):
        preothertable = tablesel.curselection()
        tableselval = tablesel.get(preothertable[0])
        self.othertable = self.table_registry.table_handles[tableselval]
        tablejoinmenuone.destroy()
        mutualcols = []
        for col in self.filein.columns:
            if col in self.othertable.columns:
                mutualcols.append(col)
        if len(mutualcols) < 1:
            nomutuals = Tk.Toplevel()
            label1 = Tk.Label(nomutuals,
                              text="There are no mutual columns to join on." +
                              "  Choose two tables with some " +
                              "identical columns.")
            label1.pack()
            button1 = Tk.Button(nomutuals, text="Got It.")
            button1["command"] = lambda: nomutuals.destroy()
            button1.pack()
        else:
            choosemerge = Tk.Toplevel()
            label1 = Tk.Label(choosemerge,
                              text="Choose which columns present in both" +
                              " data tables to use to merge")
            label1.grid(column=0, row=0)
            maxlen = 20
            for col in mutualcols:
                if len(col) > maxlen:
                    maxlen = len(col)
            joincols = self.listbox_and_scrollbar(choosemerge, mutualcols,
                                                  column=0, row=1,
                                                  maxlen=maxlen,
                                                  multiselect=True)
            button1 = Tk.Button(choosemerge, text="Merge")
            button1["command"] = lambda choosemerge=choosemerge, \
                joincols=joincols, tableselval=tableselval: \
                self.tablejoingo(choosemerge, joincols, tableselval)
            button1.grid(column=0, row=2)

    def tablejoingo(self, choosemerge, prejoincols, tableselval):
        try:
            joincols = [prejoincols.get(pjc) for pjc in
                        prejoincols.curselection()]
            self.othertable = self.table_registry.table_handles[tableselval]
            self.subtable = self.filein.join(
                self.othertable.set_index(joincols), on=joincols,
                lsuffix="_lefttable", rsuffix="_righttable")
            choosemerge.destroy()
            pnewname = self.table_registry.get_new_table_name()
            newname = "Merge Table " + pnewname
            try:
                self.table_registry.rename_table(pnewname, newname,
                                                 self.subtable)
            except KeyError:
                newname = pnewname
            window = Tk.Toplevel()
            window.wm_title(newname)
            dashclass(
                self.subtable, self.basedir, basefile, "child",
                self.pythonwin, master=window,
                table_registry=self.table_registry, table_name=newname)
            self.pythonwin.new_code(
                    "joincols = [\"" + "\",\"".join(joincols) + "\"]")
            self.pythonwin.new_code(
                    "datatables[\"" + newname + "\"] = datatables[\"" +
                    self.table_name + "\"].join(datatables[\"" + tableselval +
                    "\"].set_index(joincols),")
            self.pythonwin.new_code(
                    "        on=joincols,\n        lsuffix=\"_lefttable\"," +
                    "\n        rsuffix=\"_righttable\"\n)")
        except Exception as myerr:
            self.tkprint("I could not join tables due to an error.",
                         myerr, sys.exc_info()[0], sys.exc_info()[1])

    def addcol(self):
        addcolwin = Tk.Toplevel()
        instructions = "Create a new column.\nTo reference existing " + \
                       "columns, use the row[\"Column\"] idiom.\n" + \
                       "You can use Numpy commands as np.<command>, " + \
                       "such as np.log10(row[\"Column\"]).\n" + \
                       "If you are familiar with Pandas " + \
                       "DataFrame.apply(lambda row: <function>, axis=1)\n" + \
                       "Note that you are supplying <function> " + \
                       "in this construct."
        label1 = Tk.Label(addcolwin, text=instructions)
        label1.pack()
        frame1 = Tk.Frame(addcolwin)
        label2 = Tk.Label(frame1, text="New Column Name")
        label2.grid(column=0, row=0)
        newcolname = Tk.StringVar()
        entry1 = Tk.Entry(frame1, textvariable=newcolname, width=30)
        entry1.grid(column=1, row=0)
        frame1.pack()
        frame2 = Tk.Frame(addcolwin)
        label3 = Tk.Label(frame2, text="New Column Function")
        label3.grid(column=0, row=0)
        newcolfunction = Tk.StringVar()
        entry2 = Tk.Entry(frame2, textvariable=newcolfunction, width=100)
        entry2.grid(column=1, row=0)
        frame2.pack()
        button1 = Tk.Button(addcolwin, text="Create Column")
        button1["command"] = lambda addcolwin=addcolwin, \
            newcolname=newcolname, newcolfunction=newcolfunction: \
            self.addcolgo(addcolwin, newcolname, newcolfunction)
        button1.pack()

    def addcolgo(self, addcolwin, newcolname, newcolfunction):
        if newcolname.get() in self.filein.columns:
            self.tkprint(newcolname.get(),
                         "is already a column in your table.  " +
                         "Please choose a new column name.")
        else:
            try:
                self.filein[newcolname.get()] = eval(
                        "self.filein.apply(lambda row: " +
                        newcolfunction.get() + ", axis=1)")
                self.pythonwin.new_code(
                        "datatables[\"" + self.table_name + "\"][\"" +
                        newcolname.get() + "\"] = self.filein.apply(" +
                        "lambda row: " + newcolfunction.get() + ", axis=1)")
                addcolwin.destroy()
                self.doupdate()
            except Exception as myerr:
                self.tkprint("Error adding new column:",
                             myerr, sys.exc_info()[0], sys.exc_info()[1])

    def filesave(self):
        filetypes = (("CSV files", "*.csv"), ("All files", "*.*"))
        basefile = filedialog.asksaveasfile(mode='w', filetypes=filetypes)
        if basefile is None:    # asksaveasfile return `None`
            return   # if dialog closed with "cancel".
        basefile = basefile.name
        if basefile[-4:].lower() != ".csv":
            basefile += ".csv"
        self.filein.to_csv(basefile, index=False)
        self.pythonwin.new_code("datatables[\"" + self.table_name +
                                "\"].to_csv(\"" + basefile +
                                "\", index=False)")

    def scatterplot(self):
        spwin = Tk.Toplevel()
        label1 = Tk.Label(spwin,
                          text="Select Column(s) to make Scatterplot(s) for")
        label1.pack()
        frame1 = Tk.Frame(spwin)
        maxlen = 10
        numericcols = []
        numericdtcols = []
        for col in self.filein.columns:
            if self.filein[col].dtype == np.float64 or \
              self.filein[col].dtype == np.int64:
                numericcols.append(col)

            if len(col) > maxlen:
                maxlen = len(col)

            if (self.filein[col].dtype == np.float64 or
                    self.filein[col].dtype == np.int64 or
                    np.issubdtype(self.filein[col].dtype, np.datetime64)):
                numericdtcols.append(col)

        label2 = Tk.Label(frame1, text="Split Column")
        label2.grid(column=0, row=0)
        scatterspcol = self.listbox_and_scrollbar(frame1, self.filein.columns,
                                                  column=0, row=1,
                                                  maxlen=maxlen)
        label3 = Tk.Label(frame1, text="X Column")
        label3.grid(column=1, row=0)
        scatterxcols = self.listbox_and_scrollbar(frame1, numericdtcols,
                                                  column=1, row=1,
                                                  maxlen=maxlen,
                                                  multiselect=True)
        label4 = Tk.Label(frame1, text="Y Column")
        label4.grid(column=2, row=0)
        scatterycols = self.listbox_and_scrollbar(frame1, numericcols,
                                                  column=2, row=1,
                                                  maxlen=maxlen,
                                                  multiselect=True)
        button2 = Tk.Button(frame1,
                            text="For X Column, Use a List in a Table Instead")
        button2["command"] = lambda scatterxcols=scatterxcols: \
            self.colloader(scatterxcols)
        button2.grid(column=1, row=2)
        button3 = Tk.Button(frame1,
                            text="For Y Column, Use a List in a Table Instead")
        button3["command"] = lambda scatterxcols=scatterycols: \
            self.colloader(scatterycols)
        button3.grid(column=2, row=2)
        frame1.pack()
        linfitcheckbox = Tk.IntVar(value=0)
        self.linfitchk = Tk.Checkbutton(spwin,
                                        text="Create Linear Fit per Split",
                                        variable=linfitcheckbox)
        self.linfitchk.pack()
        button1 = Tk.Button(spwin, text="Make Scatterplot(s)")
        button1["command"] = lambda spwin=spwin, scatterspcol=scatterspcol, \
            scatterxcols=scatterxcols, scatterycols=scatterycols, \
            linfitcheckbox=linfitcheckbox: self.scatterplotgo(spwin,
                                                              scatterspcol,
                                                              scatterxcols,
                                                              scatterycols,
                                                              linfitcheckbox)
        button1.pack()

    def scatterplotgo(self, spwin, scatterspcol, scatterxcols,
                      scatterycols, linfitcheckbox):
        prexcols = scatterxcols.curselection()
        preycols = scatterycols.curselection()
        spcol = scatterspcol.get(scatterspcol.curselection()[0])
        linearfit = False
        if linfitcheckbox.get() == 1:
            linearfit = True
        slopes = {}
        intercepts = {}
        rsqs = {}
        for prexcol in prexcols:
            for preycol in preycols:
                xcol = scatterxcols.get(prexcol)
                ycol = scatterycols.get(preycol)
                if not linearfit:
                    fig = plt.figure(figsize=(10, 8))
                    gridspec.GridSpec(1, 1)
                    ax = plt.subplot2grid((1, 1), (0, 0))
                    bycolorgraph(self.filein, xcol, ycol, spcol,
                                 linearfit=linearfit, fig=fig, myax=ax)
                    self.pythonwin.new_code("bycolorgraph(datatables[\"" +
                                            self.table_name + "\"], \"" +
                                            str(xcol) + "\", \"" + str(ycol) +
                                            "\", \"" + str(spcol) +
                                            "\", linearfit=False)")
                else:
                    fig = plt.figure(figsize=(10, 8))
                    gridspec.GridSpec(6, 1)
                    ax = plt.subplot2grid((6, 1), (0, 0), rowspan=4)
                    plt.plot(0, 0, c='w', marker='.', label=spcol)
                    bycolorgraph(self.filein, xcol, ycol, spcol,
                                 linearfit=linearfit, fig=fig, myax=ax)
                    self.pythonwin.new_code("bycolorgraph(datatables[\"" +
                                            self.table_name + "\"], \"" +
                                            str(xcol) + "\", \"" + str(ycol) +
                                            "\", \"" + str(spcol) +
                                            "\", linearfit=True)")
                    self.pythonwin.new_code(
                            "#Code for summary box not " +
                            "provided; see also slope, intercept, rsqs, *_ " +
                            "= stats.linregress()")
                    # make a table of fit parameters
                    for mysp in self.filein[spcol].unique():
                        subtab = self.filein[self.filein[spcol].apply(
                            lambda val: val == mysp)][[xcol, ycol, spcol]] \
                            .dropna()
                        slopes[mysp], intercepts[mysp], rsqs[mysp], *_ = \
                            stats.linregress(subtab[xcol], subtab[ycol])
                        rsqs[mysp] = "%0.4f" % rsqs[mysp] ** 2
                        if (np.abs(slopes[mysp]) < 1000 and
                                np.abs(slopes[mysp]) > 0.999) or \
                                slopes[mysp] == 0:
                            slopes[mysp] = "%0.3f" % slopes[mysp]
                        else:
                            slopes[mysp] = "%0.3e" % slopes[mysp]
                        if (np.abs(intercepts[mysp]) < 1000 and
                                np.abs(intercepts[mysp]) > 0.999) or \
                                intercepts[mysp] == 0:
                            intercepts[mysp] = "%0.3f" % intercepts[mysp]
                        else:
                            intercepts[mysp] = "%0.3e" % intercepts[mysp]
                    outtable_columns = ('Slope', 'Intercept', 'Rsq')
                    outtable_rows = \
                        sorted(self.filein[spcol].unique().tolist())
                    outtable = []
                    for row in outtable_rows:
                        outtable.append((slopes[row],
                                         intercepts[row], rsqs[row]))
                    titleax = plt.subplot2grid((6, 1), (4, 0))
                    titleax = plt.gca()
                    titleax.axis('tight')
                    titleax.axis('off')
                    titleax.table(cellText=outtable,
                                  rowLabels=outtable_rows,
                                  colLabels=outtable_columns,
                                  loc='bottom')
                    dummyax = plt.subplot2grid((6, 1), (5, 0))
                    dummyax.axis('off')
                    botval = 0.05 + 0.018 * len(outtable_rows)
                    plt.subplots_adjust(bottom=botval)
                fig = plt.gcf()
                scatterfigout = Tk.Toplevel()
                scatterfigout.wm_title("Scatterplot " + xcol + " vs. " + ycol)
                formrone = Tk.Frame(scatterfigout)
                canvas = FigureCanvasTkAgg(fig, master=formrone)
                canvas.draw()
                canvas._tkcanvas.pack(side="top", fill="both", expand=1)
                toolbar_frame_one = Tk.Frame(formrone)
                toolbarone = NavigationToolbar2Tk(canvas, toolbar_frame_one)
                toolbarone.update()
                toolbarone.grid(row=0, column=0, columnspan=2)
                toolbar_frame_one.pack()
                formrone.pack()
                plt.close(fig)  # otherwise, the next plt.show will show this
        spwin.destroy()

    def meltmenu(self):
        meltmenuwin = Tk.Toplevel()
        label1 = Tk.Label(meltmenuwin,
                          text="Stack columns in this table and keep others")
        label1.pack()
        maxlen = 10
        for col in self.filein.columns:
            if len(col) > maxlen:
                maxlen = len(col)
        frame1 = Tk.Frame(meltmenuwin)
        label2 = Tk.Label(frame1, text="Columns to keep:")
        label2.grid(column=0, row=0)
        meltkeepcols = self.listbox_and_scrollbar(frame1, self.filein.columns,
                                                  column=0, row=1,
                                                  maxlen=maxlen,
                                                  multiselect=True)
        label3 = Tk.Label(frame1, text="Columns to stack:")
        label3.grid(column=1, row=0)
        meltstackcols = self.listbox_and_scrollbar(frame1, self.filein.columns,
                                                   column=1, row=1,
                                                   maxlen=maxlen,
                                                   multiselect=True)
        frame1.pack()
        frame2 = Tk.Frame(meltmenuwin)
        label4 = Tk.Label(frame2, text="Output Data Column Name:")
        label4.grid(column=0, row=0)
        valuename = Tk.StringVar()
        entry1 = Tk.Entry(frame2, textvariable=valuename)
        entry1.grid(column=1, row=0)
        frame2.pack()
        button1 = Tk.Button(meltmenuwin, text="Stack")
        button1["command"] = lambda meltmenuwin=meltmenuwin, \
            meltkeepcols=meltkeepcols, meltstackcols=meltstackcols, \
            valuename=valuename: self.meltgo(meltmenuwin, meltkeepcols,
                                             meltstackcols, valuename)
        button1.pack()

    def meltgo(self, meltmenuwin, meltkeepcols, meltstackcols, valuename):
        prekeepcols = meltkeepcols.curselection()
        keepcols = [meltkeepcols.get(pkc) for pkc in prekeepcols]
        prestackcols = meltstackcols.curselection()
        stackcols = [meltstackcols.get(psc) for psc in prestackcols]
        worked = False
        try:
            if len(valuename.get()) < 1:
                self.subtable = pd.melt(
                        self.filein, id_vars=keepcols, value_vars=stackcols)
                self.pythonwin.new_code(
                        "subtable = pd.melt(datatables[\"" +
                        self.table_name + "\"],")
                self.pythonwin.new_code(
                        "        id_vars=[\"" +
                        "\",\"".join(keepcols) + "\"],")
                self.pythonwin.new_code(
                        "        value_vars=[\"" +
                        "\",\"".join(stackcols) + "\"]\n)")
            else:
                self.subtable = pd.melt(self.filein,
                                        id_vars=keepcols,
                                        value_vars=stackcols,
                                        value_name=valuename.get())
                self.pythonwin.new_code("subtable = pd.melt(datatables[\"" +
                                        self.table_name + "\"],")
                self.pythonwin.new_code("        id_vars=[\"" +
                                        "\",\"".join(keepcols) + "\"],")
                self.pythonwin.new_code("        value_vars=[\"" +
                                        "\",\"".join(stackcols) + "\"],")
                self.pythonwin.new_code("        value_name=\"" +
                                        valuename.get() + "\"\n)")
            worked = True
        except Exception as myerr:
            self.tkprint("Could Not Stack,", myerr,
                         sys.exc_info()[0], sys.exc_info()[1])
        if worked:
            meltmenuwin.destroy()
            window = Tk.Toplevel()
            pnewname = self.table_registry.get_new_table_name()
            newname = "Stack Table of " + self.table_name
            try:
                self.table_registry.rename_table(pnewname,
                                                 newname, self.subtable)
            except KeyError:
                newname = pnewname
            window.title(newname)
            self.pythonwin.new_code(
                    "datatables[\"" + newname + "\"] = subtable")
            dashclass(self.subtable, self.basedir, basefile, "child",
                      self.pythonwin, master=window,
                      table_registry=self.table_registry, table_name=newname)

    def colsubset(self):
        colsubsetwin = Tk.Toplevel()
        label1 = Tk.Label(colsubsetwin, text="Columns to keep:")
        label1.grid(column=0, row=0)
        maxlen = 10
        for col in self.filein.columns:
            if len(col) > maxlen:
                maxlen = len(col)
        colstokeepchoice = self.listbox_and_scrollbar(colsubsetwin,
                                                      self.filein.columns,
                                                      column=0, row=1,
                                                      maxlen=maxlen,
                                                      multiselect=True)
        button1 = Tk.Button(colsubsetwin, text="Subset")
        button1["command"] = lambda colsubsetwin=colsubsetwin, \
            colstokeepchoice=colstokeepchoice: \
            self.colsubsetgo(colsubsetwin, colstokeepchoice)
        button1.grid(column=0, row=2)

    def colsubsetgo(self, colsubsetwin, colstokeepchoice):
        precolkeep = colstokeepchoice.curselection()
        colkeep = [colstokeepchoice.get(pck) for pck in precolkeep]
        self.subtable = self.filein[colkeep]
        self.pythonwin.new_code("colkeep = [\n    \"" +
                                "\",\n    \"".join(colkeep) + "\"\n]")
        colsubsetwin.destroy()
        pnewname = self.table_registry.get_new_table_name()
        newname = "Subset Table " + self.table_name
        try:
            self.table_registry.rename_table(pnewname, newname, self.subtable)
        except KeyError:
            newname = pnewname
        window = Tk.Toplevel()
        window.wm_title(newname)
        self.pythonwin.new_code(
            "datatables[\"" + newname + "\"] = datatables[\"" +
            self.table_name + "\"][colkeep]")
        dashclass(self.subtable, self.basedir, basefile, "child",
                  self.pythonwin, master=window,
                  table_registry=self.table_registry, table_name=newname)

    def mlm(self):
        mlmwin = Tk.Toplevel()
        label1 = Tk.Label(mlmwin, text="Fit Linear Model")
        label1.pack()
        form1 = Tk.Frame(mlmwin)
        maxlen = 10
        nummaxlen = 10
        numericcols = []
        for col in self.filein.columns:
            if self.filein[col].dtype == np.float64 or \
              self.filein[col].dtype == np.int64:
                numericcols.append(col)
                if len(col) > nummaxlen:
                    nummaxlen = len(col)
            if len(col) > maxlen:
                maxlen = len(col)
        label2 = Tk.Label(form1, text="Input Columns to Linear Model:  ")
        label2.grid(column=0, row=0)
        mlmdrivercols = self.listbox_and_scrollbar(form1, self.filein.columns,
                                                   column=0, row=1,
                                                   maxlen=maxlen,
                                                   multiselect=True)
        label3 = Tk.Label(form1, text="  Target Column:")
        label3.grid(column=1, row=0)
        mlmtargetcol = self.listbox_and_scrollbar(form1, numericcols,
                                                  column=1, row=1,
                                                  maxlen=nummaxlen,
                                                  multiselect=False)
        form1.pack()
        button1 = Tk.Button(mlmwin, text="Make Model")
        button1["command"] = lambda mlmwin=mlmwin, \
            mlmdrivercols=mlmdrivercols, mlmtargetcol=mlmtargetcol: \
            self.mlmgo(mlmwin, mlmdrivercols, mlmtargetcol)
        button1.pack()

    def mlmgo(self, mlmwin, mlmdrivercols, mlmtargetcol):
        premlmdrivers = mlmdrivercols.curselection()
        mlmdrive = [mlmdrivercols.get(pmd) for pmd in premlmdrivers]
        premlmtarg = mlmtargetcol.curselection()
        mlmtarg = [mlmtargetcol.get(pmt) for pmt in premlmtarg]
        mlmwin.destroy()
        colrenames = {}
        mlmdriveren = []
        for numcol in mlmdrive:
            newcolname = numcol
            newcolname = re.sub(" ", "_", newcolname)
            newcolname = re.sub("\(", "_", newcolname)
            newcolname = re.sub("\)", "_", newcolname)
            newcolname = re.sub(":", "_", newcolname)
            newcolname = re.sub("\\\\", "_", newcolname)
            newcolname = re.sub("/", "_", newcolname)
            if re.search("\\d", newcolname[:1]):
                newcolname = "_" + newcolname
            if newcolname != numcol:
                colrenames[numcol] = newcolname
            mlmdriveren.append(newcolname)
        olsrenamed = self.filein.rename(columns=colrenames)
        commandstring = mlmtarg[0] + " ~ "
        for numcol in mlmdriveren:
            commandstring = commandstring + numcol + " + "
        commandstring = commandstring[:-3]
        worked = False
        try:
            res = ols(commandstring, data=olsrenamed).fit()
            self.pythonwin.new_code("commandstring=\"" + commandstring + "\"")
            self.pythonwin.new_code("res = ols(commandstring, " +
                                    "data=datatables[\"" + self.table_name +
                                    "\"]).fit()")
            worked = True
        except SyntaxError as myerr:
            self.tkprint("Linear Model res = ols SyntaxError",
                         commandstring, myerr)
        except ValueError as myerr:
            self.tkprint("Linear Model res = ols ValueError", myerr)
        except Exception as myerr:
            self.tkprint("Linear Model res = ols other error", myerr,
                         commandstring, sys.exc_info()[0], sys.exc_info()[1])
        if worked:
            try:
                ypred = res.predict(olsrenamed)
                self.pythonwin.new_code("ypred = res.predict(datatables[\"" +
                                        self.table_name + "\"])")
                self.pythonwin.new_code("datatables[\"" + self.table_name +
                                        "\"].insert(value=ypred, column=" +
                                        "'Predicted " + mlmtarg[0] + "', " +
                                        "loc=len(datatables[\"" +
                                        self.table_name + "\"].columns))")
                worked = False
                try:
                    self.filein.insert(value=ypred, column='Predicted ' +
                                       mlmtarg[0],
                                       loc=len(self.filein.columns))
                    myres = self.filein[mlmtarg[0]] - \
                        self.filein['Predicted ' + mlmtarg[0]]
                    self.filein.insert(value=myres,
                                       column='Residual ' + mlmtarg[0],
                                       loc=len(self.filein.columns))
                    worked = True
                except ValueError as myerr:
                    self.tkprint("Linear Model insert Column Value Error",
                                 myerr)

                if worked:
                    fig = plt.figure(figsize=(10, 8))
                    graph1 = sns.regplot(
                        x=self.filein['Predicted ' + mlmtarg[0]],
                        y=self.filein[mlmtarg[0]], marker="+", robust=True)
                    # try to ax up color by split
                    spcol = "Split"
                    colorlist = colorgen()
                    # queue off first 5
                    for _, thiscolor in zip(range(5), colorlist):
                        pass
                    if spcol in self.filein.columns:
                        for thiscolor, mysp in \
                          zip(colorlist, sorted(self.filein[spcol].unique())):
                            spsub = self.filein[self.filein[spcol]
                                                .apply(lambda x: x == mysp)]
                            plt.scatter(x=spsub['Predicted ' + mlmtarg[0]],
                                        y=spsub[mlmtarg[0]], marker='+',
                                        color=thiscolor, label=mysp)
                        graph1.legend(loc='upper right',
                                      bbox_to_anchor=(1.2, 1),
                                      borderaxespad=0., fontsize=10)
                    self.pythonwin.new_code(
                        "graph1 = sns.regplot(x=datatables[\"" +
                        self.table_name + "\"]['Predicted " +
                        mlmtarg[0] + "'],")
                    self.pythonwin.new_code(
                        "        y=datatables[\"" + self.table_name +
                        "\"]['" + mlmtarg[0] + "'],")
                    self.pythonwin.new_code("        marker='+',")
                    self.pythonwin.new_code("        robust=True\n)")
                    mlmfigout = Tk.Toplevel()
                    mlmfigout.wm_title("Fit Predicted vs. Actuals")
                    fig.gca().set_title("Fit Predicted vs. Actuals")
                    formrone = Tk.Frame(mlmfigout)
                    canvas = FigureCanvasTkAgg(fig, master=formrone)
                    plt.subplots_adjust(right=0.8)
                    canvas.draw()
                    canvas._tkcanvas.pack(side="top", fill="both", expand=1)
                    toolbar_frame_one = Tk.Frame(formrone)
                    toolbarone = NavigationToolbar2Tk(canvas,
                                                      toolbar_frame_one)
                    toolbarone.update()
                    toolbarone.grid(row=0, column=0, columnspan=2)
                    toolbar_frame_one.pack()
                    formrone.pack()
                    mlmfigout4 = Tk.Toplevel()
                    # otherwise, the next plt.show will show this
                    plt.close(fig)
                    fig3 = plt.figure(figsize=(8, 8))
                    sns.regplot(x=self.filein[mlmtarg[0]],
                                y=self.filein['Residual ' + mlmtarg[0]],
                                marker="+", robust=True)
                    # try to ax up color by split
                    colorlist = colorgen()
                    # queue off first 5
                    for _, thiscolor in zip(range(5), colorlist):
                        pass
                    if spcol in self.filein.columns:
                        for thiscolor, mysp in \
                          zip(colorlist, sorted(self.filein[spcol].unique())):
                            spsub = self.filein[self.filein[spcol]
                                                .apply(lambda x: x == mysp)]
                            plt.scatter(x=spsub[mlmtarg[0]],
                                        y=spsub['Residual ' + mlmtarg[0]],
                                        marker='+', color=thiscolor,
                                        label=mysp)
                        fig3.gca().legend(loc='upper right',
                                          bbox_to_anchor=(1.2, 1),
                                          borderaxespad=0., fontsize=10)
                    mlmfigout4.wm_title("Fit Actuals vs. Residuals")
                    fig3.gca().set_title("Fit Actuals vs. Residuals")
                    formfour = Tk.Frame(mlmfigout4)
                    canvastwo = FigureCanvasTkAgg(fig3, master=formfour)
                    plt.subplots_adjust(right=0.8)
                    canvastwo.show()
                    canvastwo._tkcanvas.pack(side="top", fill="both", expand=1)
                    toolbar_frame_four = Tk.Frame(formfour)
                    toolbarfour = NavigationToolbar2Tk(canvastwo,
                                                       toolbar_frame_four)
                    toolbarfour.update()
                    toolbarfour.grid(row=0, column=0, columnspan=2)
                    toolbar_frame_four.pack()
                    formfour.pack()
                    mlmfigout3 = Tk.Toplevel()
                    mlmfigout3.wm_title("Fit Model Summary")
                    labelx = Tk.Label(mlmfigout3, text=res.summary(),
                                      font=('Consolas', 15))
                    self.pythonwin.new_code("print(res.summary())")
                    labelx.pack()
                    # create the window even if it will fail.
                    # Have something to call mlmmulticlose with.
                    mlmfigout2 = Tk.Toplevel()
                    mlmfigout2.wm_title(
                        "Fit Model Component and Component-Plus-Residual Grid")
                    button1 = Tk.Button(mlmfigout3,
                                        text="Close Fit Model Windows")
                    button1["command"] = lambda mlmfigout=mlmfigout, \
                        mlmfigout2=mlmfigout2, mlmfigout3=mlmfigout3, \
                        mlmfigout4=mlmfigout4: self.mlmmulticlose(mlmfigout,
                                                                  mlmfigout2,
                                                                  mlmfigout3,
                                                                  mlmfigout4)
                    button1.pack()
                    # otherwise, the next plt.show will show this
                    plt.close(fig3)
                    fig2 = plt.figure(figsize=(10, 8))
                    try:
                        gout = sm.graphics.plot_ccpr_grid(res, fig=fig2)
                        gout  # clear spyder warning
                        self.pythonwin.new_code(
                                "gout = sm.graphics.plot_ccpr_grid(res)")
                        formrtwo = Tk.Frame(mlmfigout2)
                        canvasth = FigureCanvasTkAgg(fig2, master=formrtwo)
                        canvasth.show()
                        canvasth._tkcanvas.pack(side="top", fill="both",
                                                expand=1)
                        toolbar_frame_two = Tk.Frame(formrtwo)
                        toolbartwo = NavigationToolbar2Tk(canvasth,
                                                          toolbar_frame_two)
                        toolbartwo.update()
                        toolbartwo.grid(row=0, column=0, columnspan=2)
                        toolbar_frame_two.pack()
                        formrtwo.pack()
                        # otherwise, the next plt.show will show this
                        plt.close(fig2)
                    except Exception as myerr:
                        self.tkprint("Could not make a CCPR grid,", myerr,
                                     sys.exc_info()[0], sys.exc_info()[1])
            except Exception as myerr:
                self.tkprint("Fit Linear Model other error,", myerr,
                             sys.exc_info()[0], sys.exc_info()[1])

    def mlmmulticlose(self, mlmfigout, mlmfigout2, mlmfigout3, mlmfigout4):
        mlmfigout.destroy()
        try:
            mlmfigout2.destroy()
        except Exception as myerr:
            pass
        mlmfigout3.destroy()
        mlmfigout4.destroy()

    def tkprint(self, *message):
        tkprintwin = Tk.Toplevel()
        label1 = Tk.Label(tkprintwin, text=" ".join(str(x) for x in message))
        label1.pack()
        button1 = Tk.Button(tkprintwin, text="Got It.")
        button1["command"] = lambda: self.tkprintclose(tkprintwin)
        button1.pack()

    def tkprintclose(self, parentwindow):
        parentwindow.destroy()

    def timecol(self):
        # find candidate timecols
        timecols = []
        maxlen = 10
        sawcols = {}
        for index, row in self.filein.iterrows():
            unseen = 0
            for col in self.filein.columns:
                if col in sawcols:
                    continue
                unseen += 1
                if pd.notnull(row[col]):
                    sawcols[col] = 1
                    compiled = re.compile("\\d{2}(-|\\/)\\d{2}(-|\\/)\\d{2}")
                    if re.search(compiled, str(row[col])):
                        timecols.append(col)
                        if len(col) > maxlen:
                            maxlen = len(col)
            # break out of the iterrows() loop
            if unseen == 0:
                break

        if len(timecols) < 1:
            self.tkprint("I could not find any candidate time columns.")
        else:
            timecolwin = Tk.Toplevel()
            label1 = Tk.Label(
                timecolwin,
                text="Select Column to serve as a basis for a Time Column")
            label1.grid(column=0, row=0)
            timetargetcol = self.listbox_and_scrollbar(timecolwin, timecols,
                                                       column=0, row=1,
                                                       maxlen=maxlen,
                                                       multiselect=False)
            button1 = Tk.Button(timecolwin, text="Create Time Column")
            button1["command"] = lambda timecolwin=timecolwin, \
                timetargetcol=timetargetcol: self.timecolgo(timecolwin,
                                                            timetargetcol)
            button1.grid(column=0, row=2)

    def timecolgo(self, timecolwin, timetargetcol):
        timecol = timetargetcol.get(timetargetcol.curselection()[0])
        convert_to_epoch(self.filein, dtcol=timecol,
                         newloc=len(self.filein.columns))
        plotsdt = pd.to_datetime(self.filein['Unix_Epoch_Time'], unit="s")
        self.pythonwin.new_code(
            "convert_to_epoch(datatables[\"" + self.table_name +
            "\"], dtcol='" + timecol + "', newloc=len(datatables[\"" +
            self.table_name + "\"].columns))")
        self.pythonwin.new_code(
            "plotsdt = pd.to_datetime(datatables[\"" + self.table_name +
            "\"]['Unix_Epoch_Time'], unit='s')")
        self.pythonwin.new_code(
            "datatables[\"" + self.table_name +
            "\"].insert(value=plotsdt,column='" + timecol +
            "_datetime',loc=len(datatables[\"" + self.table_name +
            "\"].columns))")
        self.pythonwin.new_code("del datatables[\"" + self.table_name +
                                "\"]['Unix_Epoch_Time']")
        try:
            self.filein.insert(value=plotsdt, column=timecol + '_datetime',
                               loc=len(self.filein.columns))
            del self.filein['Unix_Epoch_Time']
        except ValueError as myerr:
            self.tkprint("time column creation Value error", myerr)
        timecolwin.destroy()

    def sortmenu(self):
        sortmenuwin = Tk.Toplevel()
        label1 = Tk.Label(sortmenuwin, text="Select a column to sort by")
        label1.grid(column=0, row=0)
        maxlen = 10
        for col in self.filein.columns:
            if len(col) > maxlen:
                maxlen = len(col)
        sorttargetcol = self.listbox_and_scrollbar(sortmenuwin,
                                                   self.filein.columns,
                                                   column=0, row=1,
                                                   maxlen=maxlen,
                                                   multiselect=False)
        descendingvar = Tk.IntVar(value=0)
        descbox = Tk.Checkbutton(sortmenuwin, text="Sort Descending",
                                 variable=descendingvar)
        descbox.grid(column=0, row=2)
        button2 = Tk.Button(sortmenuwin, text="Add Sortable Column")
        sortlist = Tk.StringVar()
        sortlist.set("")
        button2["command"] = lambda sortlist=sortlist, \
            sorttargetcol=sorttargetcol, descendingvar=descendingvar: \
            self.sortmenuupd(sortlist, sorttargetcol, descendingvar)
        button2.grid(column=0, row=3)
        label3 = Tk.Label(sortmenuwin, text="Sort By:")
        label3.grid(column=0, row=4)
        label2 = Tk.Label(sortmenuwin, textvariable=sortlist)
        label2.grid(column=0, row=5)
        button1 = Tk.Button(sortmenuwin, text="Sort")
        button1["command"] = lambda sortmenuwin=sortmenuwin, \
            sortlist=sortlist: self.sortmenugo(sortmenuwin, sortlist)
        button1.grid(column=0, row=6)

    def sortmenuupd(self, sortlist, sorttargetcol, descendingvar):
        presort = sorttargetcol.get(sorttargetcol.curselection()[0])
        asc = ", Ascending"
        if descendingvar.get() == 1:
            asc = ", Descending"
        wehave = sortlist.get()
        sortlist.set(wehave + presort + asc + '\n')

    def sortmenugo(self, sortmenuwin, sortlist):
        presort = sortlist.get()
        if len(presort) < 1:
            self.tkprint("Cannot sort with no sort columns added")
        else:
            pscols = presort.split("\n")
            # the last item in the array is a blank.
            pscols = pscols[:-1]
            sortcols = []
            myascends = []
            for pscol in pscols:
                myarr = pscol.split(",")
                sortcols.append(",".join(myarr[:-1]))
                if myarr[-1:] == [" Ascending"]:
                    myascends.append(True)
                else:
                    myascends.append(False)
            try:
                self.filein = self.filein.sort_values(by=sortcols,
                                                      ascending=myascends)
                self.pythonwin.new_code(
                    "sortcols=[\"" + "\",\"".join(sortcols) + "\"]")
                self.pythonwin.new_code(
                    "myascends=[" + ",".join(str(x) for x in myascends) + "]")
                self.pythonwin.new_code(
                    "datatables[\"" + self.table_name + "\"] = datatables[\"" +
                    self.table_name +
                    "\"].sort_values(by=sortcols, ascending=myascends)")
                sortmenuwin.destroy()
                self.doupdate()
            except Exception as myerr:
                self.tkprint("Sort Encountered an error", myerr,
                             sys.exc_info()[0], sys.exc_info()[1])

    def ymmenu(self):
        ymmenuwin = Tk.Toplevel()
        label1 = Tk.Label(ymmenuwin,
                          text="Select Columns Needed for Yield Mosaic; " +
                          "DieX/DieY is assumed")
        label1.pack()
        form1 = Tk.Frame(ymmenuwin)
        label2 = Tk.Label(form1, text="Bin Letters Column  ")
        label2.grid(column=0, row=0)
        label3 = Tk.Label(form1, text="  Split Column  ")
        label3.grid(column=1, row=0)
        label4 = Tk.Label(form1, text="  WaferId Column")
        label4.grid(column=2, row=0)
        maxlen = np.max([len(col) for col in self.filein.columns])
        blcol = self.listbox_and_scrollbar(form1, self.filein.columns,
                                           column=0, row=1, maxlen=maxlen,
                                           multiselect=False)
        spcol = self.listbox_and_scrollbar(form1, self.filein.columns,
                                           column=1, row=1, maxlen=maxlen,
                                           multiselect=False)
        wfrcol = self.listbox_and_scrollbar(form1, self.filein.columns,
                                            column=2, row=1, maxlen=maxlen,
                                            multiselect=False)
        form1.pack()
        button1 = Tk.Button(ymmenuwin, text="Load Bin Order Reference File")
        button1["command"] = self.binorderref
        button1.pack()
        button2 = Tk.Button(ymmenuwin, text="Create Yield Mosaic")
        button2["command"] = lambda ymmenuwin=ymmenuwin, blcol=blcol, \
            spcol=spcol, wfrcol=wfrcol: self.ymcreate(ymmenuwin,
                                                      blcol, spcol, wfrcol)
        button2.pack()

    def binorderref(self):
        filetypes = (("CSV files", "*.csv"),
                     ("Text files", "*.txt"), ("All files", "*.*"))
        wfofile = filedialog.askopenfilename(initialdir='.',
                                             filetypes=filetypes)
        with open(wfofile, 'r') as fhand:
            firstline = True
            for line in fhand:
                if firstline:
                    firstline = False
                    line = re.sub("\"", "", line)
                    line = re.sub(r"\[", "", line)
                    line = re.sub(r"\]", "", line)
                    self.waterfallorder = re.split(",", line)

    def ymcreate(self, ymmenuwin, blcol, spcol, wfrcol):
        try:
            bitlcol = blcol.get(blcol.curselection()[0])
            splcol = spcol.get(spcol.curselection()[0])
            wfridcol = wfrcol.get(wfrcol.curselection()[0])
            ymmenuwin.destroy()
            subtab = self.filein[[bitlcol, splcol, wfridcol,
                                  'DieX', 'DieY']].copy()
            yieldmosaic_function(subtab, self.waterfallorder,
                                 bitlcol, spname=splcol,
                                 wfrname=wfridcol, savelocn=chr(127))
            self.pythonwin.new_code(
                "subtab = datatables[\"" + self.table_name + "\"][['" +
                bitlcol + "', '" + splcol + "', '" + wfridcol +
                "', 'DieX', 'DieY']].copy()")
            self.pythonwin.new_code(
                "#!!WARNING You may prefer to define this yourself.!!#")
            wfprintable = self.wfprintmaker()
            self.pythonwin.new_code("waterfallorder = [" + wfprintable + "]")
            self.pythonwin.new_code(
                "yieldmosaic_function(subtab, waterfallorder, '" + bitlcol +
                "', spname='" + splcol + "', wfrname='" + wfridcol +
                "', savelocn=chr(127))")
            fig = plt.gcf()
            wmapfigout = Tk.Toplevel()
            wmapfigout.wm_title("YieldMosaic Map")
            formrone = Tk.Frame(wmapfigout)
            canvas = FigureCanvasTkAgg(fig, master=formrone)
            canvas.draw()
            canvas._tkcanvas.pack(side="top", fill="both", expand=1)
            toolbar_frame_one = Tk.Frame(formrone)
            toolbarone = NavigationToolbar2Tk(canvas, toolbar_frame_one)
            toolbarone.update()
            toolbarone.grid(row=0, column=0, columnspan=2)
            toolbar_frame_one.pack()
            formrone.pack()
            plt.close(fig)  # otherwise, the next plt.show will show this
        except Exception as myerr:
            self.tkprint("Yield Mosaic Creation Encountered an Error,",
                         myerr, sys.exc_info()[0], sys.exc_info()[1])

    def renamecol(self):
        renamecolwin = Tk.Toplevel()
        label1 = Tk.Label(renamecolwin, text="Select Column:")
        label1.grid(column=0, row=0)
        maxlen = np.max([len(col) for col in self.filein.columns])
        rncol = self.listbox_and_scrollbar(renamecolwin,
                                           self.filein.columns,
                                           column=0, row=1, maxlen=maxlen,
                                           multiselect=False)
        label2 = Tk.Label(renamecolwin, text="New Name:")
        label2.grid(column=0, row=2)
        rnname = Tk.StringVar()
        entry1 = Tk.Entry(renamecolwin, textvariable=rnname)
        entry1.grid(column=0, row=3)
        button1 = Tk.Button(renamecolwin, text="Rename")
        button1["command"] = lambda renamecolwin=renamecolwin, rncol=rncol, \
            rnname=rnname: self.renamecolgo(renamecolwin, rncol, rnname)
        button1.grid(column=0, row=4)

    def renamecolgo(self, renamecolwin, rncol, rnname):
        rncolval = rncol.get(rncol.curselection()[0])
        rnnameval = rnname.get()
        if rnnameval not in self.filein.columns:
            renamecolwin.destroy()
            self.filein = self.filein.rename(columns={rncolval: rnnameval})
            self.pythonwin.new_code(
                "datatables[\"" + self.table_name + "\"] = datatables[\"" +
                self.table_name + "\"].rename(columns={'" + rncolval +
                "' : '" + rnnameval + "'})")
            self.doupdate()
        else:
            self.tkprint("There is already a column with that name.  " +
                         "Choose another please.")

    def watermenu(self):
        watermenuwin = Tk.Toplevel()
        label1 = Tk.Label(watermenuwin,
                          text="Select Columns Needed for Waterfall Chart; " +
                          "LotId is assumed.")
        label1.pack()
        form1 = Tk.Frame(watermenuwin)
        label2 = Tk.Label(form1, text="Bin Letters Column  ")
        label2.grid(column=0, row=0)
        label3 = Tk.Label(form1, text="  Split Column  ")
        label3.grid(column=1, row=0)
        maxlen = np.max([len(col) for col in self.filein.columns])
        blcol = self.listbox_and_scrollbar(form1, self.filein.columns,
                                           column=0, row=1,
                                           maxlen=maxlen, multiselect=False)
        spcol = self.listbox_and_scrollbar(form1, self.filein.columns,
                                           column=1, row=1,
                                           maxlen=maxlen, multiselect=False)
        form1.pack()
        button1 = Tk.Button(watermenuwin, text="Load Bin Order Reference File")
        button1["command"] = self.binorderref
        button1.pack()
        button2 = Tk.Button(watermenuwin, text="Create WaterFall Chart")
        button2["command"] = lambda watermenuwin=watermenuwin, blcol=blcol, \
            spcol=spcol: self.watercreate(watermenuwin, blcol, spcol)
        button2.pack()

    def wfprintmaker(self):
        retval = "\""
        for counter in range(0, len(self.waterfallorder), 15):
            retval += "\",\"".join(self.waterfallorder[counter:counter + 15]) \
                   + "\",\n        \""
        retval = retval[:-9]
        return retval

    def watercreate(self, watermenuwin, blcol, spcol):
        try:
            bitlcol = blcol.get(blcol.curselection()[0])
            splcol = spcol.get(spcol.curselection()[0])
            watermenuwin.destroy()
            lotidname = "LotId"
            if lotidname not in self.filein.columns:
                lotidname = "Lot Id"
            if lotidname not in self.filein.columns:
                lotidname = "StartLotKey"
            subtab = self.filein[[bitlcol, splcol, lotidname]].copy()
            if lotidname != "LotId":
                subtab = subtab.rename(columns={lotidname: "LotId"})
            waterfallchart(subtab, self.waterfallorder, bitlcol, False,
                           split=splcol)
            self.pythonwin.new_code(
                "subtab = datatables[\"" + self.table_name + "\"][['" +
                bitlcol + "', '" + splcol + "', 'LotId']].copy()")
            self.pythonwin.new_code(
                "#!!WARNING You may prefer to define this yourself.!!#")
            wfprintable = self.wfprintmaker()
            self.pythonwin.new_code("waterfallorder = [" + wfprintable + "]")
            self.pythonwin.new_code(
                "waterfallchart(subtab, waterfallorder, '" + bitlcol +
                "', False, split='" + splcol + "')")
            fig = plt.gcf()
            waterfigout = Tk.Toplevel()
            waterfigout.wm_title("Waterfall Plot of " + bitlcol +
                                 " by " + splcol)
            formrone = Tk.Frame(waterfigout)
            canvas = FigureCanvasTkAgg(fig, master=formrone)
            canvas.draw()
            canvas._tkcanvas.pack(side="top", fill="both", expand=1)
            toolbar_frame_one = Tk.Frame(formrone)
            toolbarone = NavigationToolbar2Tk(canvas, toolbar_frame_one)
            toolbarone.update()
            toolbarone.grid(row=0, column=0, columnspan=2)
            toolbar_frame_one.pack()
            formrone.pack()
            plt.close(fig)  # otherwise, the next plt.show will show this
        except Exception as myerr:
            self.tkprint("Waterfall Graph Creation Encountered an Error,",
                         myerr, sys.exc_info()[0], sys.exc_info()[1])

    def pivotmenu(self):
        pivotmenuwin = Tk.Toplevel()
        label1 = Tk.Label(pivotmenuwin,
                          text="Select Single Columns for Each Pivot Role")
        label1.pack()
        form1 = Tk.Frame(pivotmenuwin)
        label2 = Tk.Label(form1, text="Column to make row index  ")
        label2.grid(column=0, row=0)
        label3 = Tk.Label(form1, text="  Column to split into columns  ")
        label3.grid(column=1, row=0)
        label4 = Tk.Label(form1, text="  Column containing values to keep")
        label4.grid(column=2, row=0)
        maxlen = np.max([len(col) for col in self.filein.columns])
        indcol = self.listbox_and_scrollbar(form1, self.filein.columns,
                                            column=0, row=1,
                                            maxlen=maxlen, multiselect=False)
        colcol = self.listbox_and_scrollbar(form1, self.filein.columns,
                                            column=1, row=1,
                                            maxlen=maxlen, multiselect=False)
        valcol = self.listbox_and_scrollbar(form1, self.filein.columns,
                                            column=2, row=1,
                                            maxlen=maxlen, multiselect=False)
        form1.pack()
        button1 = Tk.Button(pivotmenuwin, text="Pivot")
        button1["command"] = lambda pivotmenuwin=pivotmenuwin, indcol=indcol, \
            colcol=colcol, valcol=valcol: self.pivotmenugo(pivotmenuwin,
                                                           indcol, colcol,
                                                           valcol)
        button1.pack()

    def pivotmenugo(self, pivotmenuwin, indcol, colcol, valcol):
        try:
            indexcol = indcol.get(indcol.curselection()[0])
            columncol = colcol.get(colcol.curselection()[0])
            valuecol = valcol.get(valcol.curselection()[0])
            pivotmenuwin.destroy()
            self.subtable = self.filein.pivot(index=indexcol,
                                              columns=columncol,
                                              values=valuecol).reset_index()
            window = Tk.Toplevel()
            pnewname = self.table_registry.get_new_table_name()
            newname = "Pivot Table of " + self.table_name
            try:
                self.table_registry.rename_table(pnewname,
                                                 newname, self.subtable)
            except KeyError:
                newname = pnewname
            window.title(newname)
            self.pythonwin.new_code(
                "datatables[\"" + newname + "\"] = datatables[\"" +
                self.table_name + "\"].pivot(index='" + indexcol +
                "', columns='" + columncol + "', values='" + valuecol +
                "').reset_index()")
            dashclass(self.subtable, self.basedir, basefile, "child",
                      self.pythonwin, master=window,
                      table_registry=self.table_registry, table_name=newname)
        except Exception as myerr:
            self.tkprint("Pivot Encountered an Error,",
                         myerr, sys.exc_info()[0], sys.exc_info()[1])

    def directpandasmenu(self):
        directpandaswin = Tk.Toplevel()
        label1 = Tk.Label(directpandaswin,
                          text="Enter a command to execute on " +
                          "this Pandas Dataframe")
        label1.pack()
        label3 = Tk.Label(directpandaswin,
                          text="Commands should be trusted, executing " +
                          "directly, routing errors to the GUI, routing " +
                          "results to the GUI")
        label3.pack()
        label4 = Tk.Label(directpandaswin,
                          text="Examples: mycol.unique(), " +
                          "dropna(inplace=True), shape")
        label4.pack()
        form1 = Tk.Frame(directpandaswin)
        label2 = Tk.Label(form1, text="mydataframe.")
        label2.grid(column=0, row=0)
        directpandascommand = Tk.StringVar()
        entry1 = Tk.Entry(form1, textvariable=directpandascommand, width=60)
        entry1.grid(column=1, row=0)
        form1.pack()
        button1 = Tk.Button(directpandaswin, text="EXECUTE")
        button1["command"] = lambda directpandaswin=directpandaswin, \
            directpandascommand=directpandascommand: self.directpandasgo(
                    directpandaswin, directpandascommand)
        button1.pack()

    def directpandasgo(self, directpandaswin, directpandascommand):
        mycommand = directpandascommand.get()
        try:
            mydataframe = self.filein
            mydataframe  # clear spyder warning
            retval = eval("self.filein." + mycommand)
            self.pythonwin.new_code(
                "mydataframe = datatables[\"" + self.table_name + "\"]")
            if retval is not None:
                self.tkprint(retval)
                self.pythonwin.new_code("print(mydataframe." + mycommand + ")")
            else:
                self.pythonwin.new_code("mydataframe." + mycommand)
            directpandaswin.destroy()
            self.doupdate()
        except Exception as myerr:
            self.tkprint("Direct Pandas Command Exception:",
                         myerr, sys.exc_info()[0], sys.exc_info()[1])

    def newpandasmenu(self):
        newpandaswin = Tk.Toplevel()
        label1 = Tk.Label(newpandaswin,
                          text="Enter a command to execute " +
                          "on this Pandas Dataframe")
        label1.pack()
        label3 = Tk.Label(newpandaswin,
                          text="Commands should be trusted, executing " +
                          "directly, routing errors to the GUI, routing " +
                          "results to a new dataframe")
        label3.pack()
        label4 = Tk.Label(newpandaswin,
                          text="Examples: .sort_values(by='mycol'), " +
                          ".dropna(), .groupby('mycol').mean()." +
                          "reset_index(), [[col for col in mydataframe." +
                          "columns if 'mycol' in col]]")
        label4.pack()
        form1 = Tk.Frame(newpandaswin)
        label2 = Tk.Label(form1, text="mydataframe")
        label2.grid(column=0, row=0)
        newpandascommand = Tk.StringVar()
        entry1 = Tk.Entry(form1, textvariable=newpandascommand, width=60)
        entry1.grid(column=1, row=0)
        form1.pack()
        button1 = Tk.Button(newpandaswin, text="EXECUTE")
        button1["command"] = lambda newpandaswin=newpandaswin, \
            newpandascommand=newpandascommand: self.newpandasgo(
                    newpandaswin, newpandascommand)
        button1.pack()

    def newpandasgo(self, newpandaswin, newpandascommand):
        mycommand = newpandascommand.get()
        if mycommand[:1] != "." and mycommand[:1] != "[":
            self.tkprint("Direct Pandas Command for New Table Issue:" +
                         "  command should start with '.' or '['")
        else:
            try:
                mydataframe = self.filein
                mydataframe  # clear spyder warning
                self.subtable = eval("self.filein" + mycommand)
                self.pythonwin.new_code(
                    "mydataframe = datatables[\"" + self.table_name + "\"]")
                newpandaswin.destroy()
                window = Tk.Toplevel()
                pnewname = self.table_registry.get_new_table_name()
                newname = "Pandas Code Direct Creation of " + self.table_name
                try:
                    self.table_registry.rename_table(pnewname,
                                                     newname, self.subtable)
                except KeyError:
                    newname = pnewname
                window.title(newname)
                self.pythonwin.new_code(
                    "datatables[\"" + newname + "\"] = mydataframe" +
                    mycommand)
                dashclass(self.subtable, self.basedir, basefile, "child",
                          self.pythonwin, master=window,
                          table_registry=self.table_registry,
                          table_name=newname)
            except Exception as myerr:
                self.tkprint("Direct Pandas Command for New Table Exception:",
                             myerr, sys.exc_info()[0], sys.exc_info()[1])

    def arbplotmenu(self):
        arbplotwin = Tk.Toplevel()
        label1 = Tk.Label(arbplotwin,
                          text="Enter a command to execute on " +
                          "this Pandas Dataframe")
        label1.pack()
        label3 = Tk.Label(arbplotwin,
                          text="Commands should be trusted, executing " +
                          "directly, routing errors to the GUI, routing " +
                          "results to a MatPlotLib Output")
        label3.pack()
        label4 = Tk.Label(arbplotwin,
                          text="Examples: plt.plot('mycol', 'mycol2', " +
                          "data=mydataframe), sns.FacetGrid(mydataframe, " +
                          "col='mycol', col_wrap=5) and mygraph.map(plt" +
                          ".scatter, 'mycol2', 'mycol3')")
        label4.pack()
        form1 = Tk.Frame(arbplotwin)
        label2 = Tk.Label(form1, text="mygraph=")
        label2.grid(column=0, row=0)
        arbplotcommand = Tk.StringVar()
        arbplotfollow = Tk.StringVar()
        entry1 = Tk.Entry(form1, textvariable=arbplotcommand, width=80)
        entry1.grid(column=1, row=0)
        label5 = Tk.Label(form1, text="<optional>")
        label5.grid(column=0, row=1)
        arbplotfollow = Tk.StringVar()
        entry2 = Tk.Entry(form1, textvariable=arbplotfollow, width=80)
        entry2.grid(column=1, row=1)
        form1.pack()
        button1 = Tk.Button(arbplotwin, text="EXECUTE")
        button1["command"] = lambda arbplotwin=arbplotwin, \
            arbplotcommand=arbplotcommand: self.arbplotgo(arbplotwin,
                                                          arbplotcommand,
                                                          arbplotfollow)
        button1.pack()

    def arbplotgo(self, arbplotwin, arbplotcommand, arbplotfollow):
        try:
            mydataframe = self.filein
            # Spyder warning override: mygraph may be used by the user's
            # arbitrary Python code part 2.  Thus that this code
            # does not reference it is not an actual error.
            mygraph = eval(arbplotcommand.get())
            mygraph
            mydataframe
            if len(arbplotfollow.get()) > 0:
                eval(arbplotfollow.get())
            self.pythonwin.new_code(
                    "mydataframe = datatables[\"" + self.table_name + "\"]")
            self.pythonwin.new_code("mygraph = " + arbplotcommand.get())
            if len(arbplotfollow.get()) > 0:
                self.pythonwin.new_code(arbplotfollow.get())
            arbplotwin.destroy()
            for counter in plt.get_fignums():
                fig = plt.figure(counter)
                wmapfigout = Tk.Toplevel()
                wmapfigout.wm_title(
                        "Arbitrary Plot Output Number " + str(counter))
                formrone = Tk.Frame(wmapfigout)
                canvas = FigureCanvasTkAgg(fig, master=formrone)
                canvas.draw()
                canvas._tkcanvas.pack(side="top", fill="both", expand=1)
                toolbar_frame_one = Tk.Frame(formrone)
                toolbarone = NavigationToolbar2Tk(canvas, toolbar_frame_one)
                toolbarone.update()
                toolbarone.grid(row=0, column=0, columnspan=2)
                toolbar_frame_one.pack()
                formrone.pack()
                plt.close(fig)  # otherwise, the next plt.show will show this
        except Exception as myerr:
            self.tkprint("Arbitrary Plotting code Exception:", myerr,
                         sys.exc_info()[0], sys.exc_info()[1])

    def dropmenu(self):
        dropwin = Tk.Toplevel()
        label1 = Tk.Label(dropwin,
                          text="Remove Rows or Columns with Missing Data")
        label1.pack()
        label2 = Tk.Label(dropwin, text="Dropping:")
        label2.pack()
        dropax = Tk.IntVar()
        dropax.set(1)
        rb2 = Tk.Radiobutton(dropwin, text="Columns",
                             variable=dropax, value=1)
        rb2.pack()
        rb1 = Tk.Radiobutton(dropwin, text="Rows   ",
                             variable=dropax, value=0)
        rb1.pack()
        drophow = Tk.IntVar()
        drophow.set(1)
        label3 = Tk.Label(dropwin, text="Drop If:")
        label3.pack()
        rb3 = Tk.Radiobutton(dropwin,
                             text="All are Missing",
                             variable=drophow, value=1)
        rb3.pack()
        rb4 = Tk.Radiobutton(dropwin,
                             text="Any are Missing",
                             variable=drophow, value=2)
        rb4.pack()
        button1 = Tk.Button(dropwin, text="Drop Missing")
        button1["command"] = lambda dropwin=dropwin, dropax=dropax, \
            drophow=drophow: self.dropgo(dropwin, dropax, drophow)
        button1.pack()

    def dropgo(self, dropwin, dropax, drophow):
        myaxis = dropax.get()
        if drophow.get() == 1:
            myhow = 'all'
        else:
            myhow = 'any'
        try:
            self.filein.dropna(axis=myaxis, how=myhow, inplace=True)
            if self.jmin > len(self.filein.columns) - self.screentabs:
                self.jmin = len(self.filein.columns) - self.screentabs
            if self.jmin < 0:
                self.jmin = 0
            self.jmax = self.jmin + self.screentabs
            if self.jmax > len(self.filein.columns):
                self.jmax = len(self.filein.columns)
            if self.screensize > len(self.filein):
                self.screensize = len(self.filein)
            if self.screentabs > len(self.filein):
                self.screentabs = len(self.filein)
            self.doupdate()
            self.pythonwin.new_code(
                "datatables[\"" + self.table_name + "\"].dropna(axis=" +
                str(myaxis) + ", how='" + myhow + "', inplace=True)")
            dropwin.destroy()
        except Exception as myerr:
            self.tkprint("Drop Missing Exception:", myerr,
                         sys.exc_info()[0], sys.exc_info()[1])

    def massrename(self):
        masswin = Tk.Toplevel()
        label1 = Tk.Label(masswin,
                          text="Rename All Columns Containing Strings")
        label1.pack()
        cb1 = Tk.IntVar()
        cbox1 = Tk.Checkbutton(masswin,
                               text='"FINAL FUNCT PROD::" -> ""', variable=cb1)
        cbox1.pack()
        cb2 = Tk.IntVar()
        cbox2 = Tk.Checkbutton(masswin,
                               text='"FINAL FUNCT CORR::" -> ""', variable=cb2)
        cbox2.pack()
        cb3 = Tk.IntVar()
        cbox3 = Tk.Checkbutton(masswin,
                               text='"LotData::"          -> ""', variable=cb3)
        cbox3.pack()
        cb4 = Tk.IntVar()
        cbox4 = Tk.Checkbutton(masswin,
                               text='"WaferData::"        -> ""', variable=cb4)
        cbox4.pack()
        cb5 = Tk.IntVar()
        cbox5 = Tk.Checkbutton(masswin,
                               text='"DieData::"          -> ""', variable=cb5)
        cbox5.pack()
        cb6 = Tk.IntVar()
        cbox6 = Tk.Checkbutton(masswin,
                               text='"::"                 -> "_"',
                               variable=cb6)
        cbox6.pack()
        cb7 = Tk.IntVar()
        form1 = Tk.Frame(masswin)
        cbox7 = Tk.Checkbutton(form1, text='Custom:', variable=cb7)
        cbox7.grid(row=0, column=0)
        ev1 = Tk.StringVar()
        entry1 = Tk.Entry(form1, textvariable=ev1)
        entry1.grid(row=0, column=1)
        label2 = Tk.Label(form1, text=' -> ')
        label2.grid(row=0, column=2)
        ev2 = Tk.StringVar()
        entry2 = Tk.Entry(form1, textvariable=ev2)
        entry2.grid(row=0, column=3)
        form1.pack()
        button1 = Tk.Button(masswin, text="Rename Columns")
        button1["command"] = lambda masswin=masswin, cb1=cb1, cb2=cb2, \
            cb3=cb3, cb4=cb4, cb5=cb5, cb6=cb6, cb7=cb7, ev1=ev1, ev2=ev2: \
            self.massrenamego(masswin, cb1, cb2, cb3, cb4, cb5,
                              cb6, cb7, ev1, ev2)
        button1.pack()

    def massrenamego(self, masswin, cb1, cb2, cb3, cb4,
                     cb5, cb6, cb7, ev1, ev2):
        try:
            if cb1.get() == 1:
                self.filein.columns = \
                    self.filein.columns.str.replace('FINAL FUNCT PROD::', '')
                self.pythonwin.new_code(
                    "datatables[\"" + self.table_name + "\"].columns = " +
                    "datatables[\"" + self.table_name + "\"].columns." +
                    "str.replace('FINAL FUNCT PROD::', '')")
            if cb2.get() == 1:
                self.filein.columns = \
                    self.filein.columns.str.replace('FINAL FUNCT CORR::', '')
                self.pythonwin.new_code(
                    "datatables[\"" + self.table_name + "\"].columns = " +
                    "datatables[\"" + self.table_name + "\"].columns" +
                    ".str.replace('FINAL FUNCT CORR::', '')")
            if cb3.get() == 1:
                self.filein.columns = \
                    self.filein.columns.str.replace('LotData::', '')
                self.pythonwin.new_code(
                    "datatables[\"" + self.table_name + "\"].columns = " +
                    "datatables[\"" + self.table_name + "\"].columns" +
                    ".str.replace('LotData::', '')")
            if cb4.get() == 1:
                self.filein.columns = \
                    self.filein.columns.str.replace('WaferData::', '')
                self.pythonwin.new_code(
                    "datatables[\"" + self.table_name + "\"].columns = " +
                    "datatables[\"" + self.table_name + "\"].columns" +
                    ".str.replace('WaferData::', '')")
            if cb5.get() == 1:
                self.filein.columns = \
                    self.filein.columns.str.replace('DieData::', '')
                self.pythonwin.new_code(
                    "datatables[\"" + self.table_name + "\"].columns = " +
                    "datatables[\"" + self.table_name + "\"].columns" +
                    ".str.replace('DieData::', '')")
            if cb6.get() == 1:
                self.filein.columns = \
                    self.filein.columns.str.replace('::', '_')
                self.pythonwin.new_code(
                    "datatables[\"" + self.table_name + "\"].columns = " +
                    "datatables[\"" + self.table_name + "\"].columns" +
                    ".str.replace('::', '_')")
            if cb7.get() == 1:
                self.filein.columns = \
                    self.filein.columns.str.replace(ev1.get(), ev2.get())
                self.pythonwin.new_code(
                    "datatables[\"" + self.table_name + "\"].columns = " +
                    "datatables[\"" + self.table_name + "\"].columns" +
                    ".str.replace('" + ev1.get() + "', '" + ev2.get() + "')")
            masswin.destroy()
            self.doupdate()
        except Exception as myerr:
            self.tkprint("Mass Rename Exception:", myerr,
                         self.exc_info()[0], self.exc_info()[1])

    def uniquemenu(self):
        uniquewin = Tk.Toplevel()
        label1 = Tk.Label(uniquewin,
                          text="Select Columns to reduce to " +
                          "non-duplicate values")
        label1.grid(column=0, row=0)
        maxlen = np.max([len(col) for col in self.filein.columns])
        uniqcols = self.listbox_and_scrollbar(uniquewin, self.filein.columns,
                                              column=0, row=1,
                                              maxlen=maxlen, multiselect=True)
        button1 = Tk.Button(uniquewin, text="Drop Duplicates")
        button1["command"] = lambda uniquewin=uniquewin, \
            uniqcols=uniqcols: self.uniquego(uniquewin, uniqcols)
        button1.grid(column=0, row=2)

    def uniquego(self, uniquewin, uniqcols):
        try:
            uniquecols = [uniqcols.get(col) for col in uniqcols.curselection()]
            self.subtable = self.filein[uniquecols].drop_duplicates()
            window = Tk.Toplevel()
            pnewname = self.table_registry.get_new_table_name()
            newname = "No-Duplicates Table of " + self.table_name
            try:
                self.table_registry.rename_table(pnewname,
                                                 newname, self.subtable)
            except KeyError:
                newname = pnewname
            window.title(newname)
            self.pythonwin.new_code(
                    "uniquecols = [\"" + "\",\"".join(uniquecols) + "\"]")
            self.pythonwin.new_code(
                    "datatables[\"" + newname + "\"] = datatables[\"" +
                    self.table_name + "\"][uniquecols].drop_duplicates()")
            uniquewin.destroy()
            dashclass(self.subtable, self.basedir, basefile, "child",
                      self.pythonwin, master=window,
                      table_registry=self.table_registry, table_name=newname)
        except Exception as myerr:
            self.tkprint("Drop Duplicates Exception:", myerr,
                         sys.exc_info()[0], sys.exc_info()[1])

    def concatmenu(self):
        pretablelist = self.table_registry.joinable_tables()
        tablelist = []
        maxlen = 20
        for thistable in pretablelist:
            if self.table_name != thistable:
                tablelist.append(thistable)
                if len(thistable) > maxlen:
                    maxlen = len(thistable)
        if len(tablelist) < 1:
            self.tkprint("There are no other tables.  Please open or " +
                         "create something to concatenate with.")
        else:
            concatwin = Tk.Toplevel()
            label1 = Tk.Label(concatwin,
                              text="Select Table(s) to Concatenate " +
                              "under this one")
            label1.grid(column=0, row=0)
            concattablesel = self.listbox_and_scrollbar(concatwin, tablelist,
                                                        column=0, row=1,
                                                        maxlen=maxlen,
                                                        multiselect=True)
            button1 = Tk.Button(concatwin, text="Concatenate")
            button1["command"] = lambda concatwin=concatwin, \
                concattablesel=concattablesel: \
                self.concatgo(concatwin, concattablesel)
            button1.grid(column=0, row=2)

    def concatgo(self, concatwin, concattablesel):
        try:
            preconcattables = [concattablesel.get(tablename)
                               for tablename in concattablesel.curselection()]
            concattables = [self.table_name]
            concathooks = [self.filein]
            for preconcat in preconcattables:
                concattables.append(preconcat)
                concathooks.append(
                        self.table_registry.table_handles[preconcat])
            self.subtable = pd.concat(concathooks)
            window = Tk.Toplevel()
            pnewname = self.table_registry.get_new_table_name()
            newname = "Concatenate Table of " + self.table_name
            try:
                self.table_registry.rename_table(pnewname,
                                                 newname, self.subtable)
            except KeyError:
                newname = pnewname
            window.title(newname)
            self.pythonwin.new_code(
                "concattables = [\n        datatables[\"" + "\"],\n        " +
                "datatables[\"".join(concattables) + "\"]\n]")
            self.pythonwin.new_code(
                "datatables[\"" + newname + "\"] = pd.concat(concattables)")
            concatwin.destroy()
            dashclass(self.subtable, self.basedir, basefile, "child",
                      self.pythonwin, master=window,
                      table_registry=self.table_registry, table_name=newname)
        except Exception as myerr:
            self.tkprint("Concatenate Exception:", myerr,
                         sys.exc_info()[0], sys.exc_info()[1])

    def facetcolsmenu(self):
        facetcolswin = Tk.Toplevel()
        label1 = Tk.Label(facetcolswin,
                          text="Select Columns for Facet Grid by Columns")
        label1.pack()
        form1 = Tk.Frame(facetcolswin)
        label2 = Tk.Label(form1, text="Col Step By")
        label2.grid(column=0, row=0)
        label3 = Tk.Label(form1, text="Hue By")
        label3.grid(column=1, row=0)
        label4 = Tk.Label(form1, text="X Col Scatter")
        label4.grid(column=2, row=0)
        label5 = Tk.Label(form1, text="Y Col Scatter")
        label5.grid(column=3, row=0)
        numericcols = []
        maxlen = 10
        nummaxlen = 10
        for col in self.filein.columns:
            if self.filein[col].dtype == np.float64 or \
              self.filein[col].dtype == np.int64:
                numericcols.append(col)
                if len(col) > nummaxlen:
                    nummaxlen = len(col)
            if len(col) > maxlen:
                maxlen = len(col)
        facetcolcol = self.listbox_and_scrollbar(form1, self.filein.columns,
                                                 column=0, row=1,
                                                 maxlen=maxlen,
                                                 multiselect=False)
        facethuecol = self.listbox_and_scrollbar(form1, self.filein.columns,
                                                 column=1, row=1,
                                                 maxlen=maxlen,
                                                 multiselect=False)
        facetxcol = self.listbox_and_scrollbar(form1, numericcols,
                                               column=2, row=1,
                                               maxlen=nummaxlen,
                                               multiselect=False)
        facetycol = self.listbox_and_scrollbar(form1, numericcols,
                                               column=3, row=1,
                                               maxlen=nummaxlen,
                                               multiselect=False)
        form1.pack()
        button1 = Tk.Button(facetcolswin, text="Make Facet Graphs")
        button1["command"] = lambda facetcolswin=facetcolswin, \
            facetcolcol=facetcolcol, facethuecol=facethuecol, \
            facetxcol=facetxcol, facetycol=facetycol: \
            self.facetcolsgo(facetcolswin, facetcolcol, facethuecol,
                             facetxcol, facetycol)
        button1.pack()

    def facetcolsgo(self, facetcolswin, facetcolcol, facethuecol,
                    facetxcol, facetycol):
        try:
            colcol = facetcolcol.get(facetcolcol.curselection()[0])
            huecol = facethuecol.get(facethuecol.curselection()[0])
            xcol = facetxcol.get(facetxcol.curselection()[0])
            ycol = facetycol.get(facetycol.curselection()[0])
            canfacet = self.filein.sort_values(by=huecol)
            mygraph = sns.FacetGrid(canfacet, col=colcol,
                                    col_wrap=4, hue=huecol)
            (mygraph.map(plt.plot, xcol, ycol, marker='o',
                         linewidth=0)).add_legend()
            self.pythonwin.new_code(
                    "canfacet = datatables[\"" + self.table_name +
                    "\"].sort_values(by=\"" + huecol + "\")")
            self.pythonwin.new_code(
                    "mygraph = sns.FacetGrid(canfacet, col=\"" + colcol +
                    "\", col_wrap=4, hue=\"" + huecol + "\")")
            self.pythonwin.new_code(
                    "(mygraph.map(plt.plot, \"" + xcol + "\", \"" + ycol +
                    "\", marker='o', linewidth=0)).add_legend()")
            facetcolswin.destroy()
            fig = plt.gcf()
            wmapfigout = Tk.Toplevel()
            wmapfigout.wm_title("FacetGrid Plot")
            formrone = Tk.Frame(wmapfigout)
            canvas = FigureCanvasTkAgg(fig, master=formrone)
            canvas.draw()
            canvas._tkcanvas.pack(side="top", fill="both", expand=1)
            toolbar_frame_one = Tk.Frame(formrone)
            toolbarone = NavigationToolbar2Tk(canvas, toolbar_frame_one)
            toolbarone.update()
            toolbarone.grid(row=0, column=0, columnspan=2)
            toolbar_frame_one.pack()
            formrone.pack()
            plt.close(fig)  # otherwise, the next plt.show will show this
        except Exception as myerr:
            self.tkprint("FacetGrid Plot Exception:", myerr,
                         sys.exc_info()[0], sys.exc_info()[1])

    def facetgridmenu(self):
        facetgridswin = Tk.Toplevel()
        label1 = Tk.Label(facetgridswin,
                          text="Select Columns for Facet Grid by " +
                          "Row and Column")
        label1.pack()
        form1 = Tk.Frame(facetgridswin)
        label2 = Tk.Label(form1, text="Col Step By")
        label2.grid(column=0, row=0)
        label6 = Tk.Label(form1, text="Row Step By")
        label6.grid(column=1, row=0)
        label3 = Tk.Label(form1, text="Hue By")
        label3.grid(column=2, row=0)
        label4 = Tk.Label(form1, text="X Col Scatter")
        label4.grid(column=0, row=2)
        label5 = Tk.Label(form1, text="Y Col Scatter")
        label5.grid(column=1, row=2)
        numericcols = []
        maxlen = 10
        nummaxlen = 10
        for col in self.filein.columns:
            if self.filein[col].dtype == np.float64 or \
              self.filein[col].dtype == np.int64:
                numericcols.append(col)
                if len(col) > nummaxlen:
                    nummaxlen = len(col)
            if len(col) > maxlen:
                maxlen = len(col)
        facetgridcol = self.listbox_and_scrollbar(form1, self.filein.columns,
                                                  column=0, row=1,
                                                  maxlen=maxlen,
                                                  multiselect=False)
        facetrowcol = self.listbox_and_scrollbar(form1, self.filein.columns,
                                                 column=1, row=1,
                                                 maxlen=maxlen,
                                                 multiselect=False)
        facethuecol = self.listbox_and_scrollbar(form1, self.filein.columns,
                                                 column=2, row=1,
                                                 maxlen=maxlen,
                                                 multiselect=False)
        facetxcol = self.listbox_and_scrollbar(form1, numericcols,
                                               column=0, row=3,
                                               maxlen=nummaxlen,
                                               multiselect=False)
        facetycol = self.listbox_and_scrollbar(form1, numericcols,
                                               column=1, row=3,
                                               maxlen=nummaxlen,
                                               multiselect=False)
        form1.pack()
        button1 = Tk.Button(facetgridswin, text="Make Facet Graphs")
        button1["command"] = lambda facetgridswin=facetgridswin, \
            facetgridcol=facetgridcol, facetrowcol=facetrowcol, \
            facethuecol=facethuecol, facetxcol=facetxcol, \
            facetycol=facetycol: self.facetgridsgo(facetgridswin,
                                                   facetgridcol,
                                                   facetrowcol,
                                                   facethuecol,
                                                   facetxcol, facetycol)
        button1.pack()

    def facetgridsgo(self, facetgridswin, facetgridcol, facetrowcol,
                     facethuecol, facetxcol, facetycol):
        try:
            colcol = facetgridcol.get(facetgridcol.curselection()[0])
            rowcol = facetrowcol.get(facetrowcol.curselection()[0])
            huecol = facethuecol.get(facethuecol.curselection()[0])
            xcol = facetxcol.get(facetxcol.curselection()[0])
            ycol = facetycol.get(facetycol.curselection()[0])
            facetgo = self.filein.sort_values(by=huecol)
            mygraph = sns.FacetGrid(facetgo, col=colcol, row=rowcol,
                                    hue=huecol)
            (mygraph.map(plt.plot, xcol, ycol, marker='o',
                         linewidth=0)).add_legend()
            self.pythonwin.new_code(
                "mygraph = sns.FacetGrid(datatables[\"" + self.table_name +
                "\"], col=\"" + colcol + "\", row=\"" + rowcol + "\", hue=\"" +
                huecol + "\")")
            self.pythonwin.new_code(
                "(mygraph.map(plt.plot, \"" + xcol + "\", \"" + ycol +
                "\", marker='o', linewidth=0)).add_legend()")
            facetgridswin.destroy()
            fig = plt.gcf()
            wmapfigout = Tk.Toplevel()
            wmapfigout.wm_title("Facetgrid Plot")
            formrone = Tk.Frame(wmapfigout)
            canvas = FigureCanvasTkAgg(fig, master=formrone)
            canvas.draw()
            canvas._tkcanvas.pack(side="top", fill="both", expand=1)
            toolbar_frame_one = Tk.Frame(formrone)
            toolbarone = NavigationToolbar2Tk(canvas, toolbar_frame_one)
            toolbarone.update()
            toolbarone.grid(row=0, column=0, columnspan=2)
            toolbar_frame_one.pack()
            formrone.pack()
            plt.close(fig)  # otherwise, the next plt.show will show this
        except Exception as myerr:
            self.tkprint("FacetGrid Plot Exception:", myerr,
                         sys.exc_info()[0], sys.exc_info()[1])

    def anovamenu(self):
        anovawin = Tk.Toplevel()
        label1 = Tk.Label(anovawin,
                          text="Calculate F-Oneway for multiple targets")
        label1.pack()
        form1 = Tk.Frame(anovawin)
        label2 = Tk.Label(form1, text="Split Column")
        label2.grid(column=0, row=0)
        label3 = Tk.Label(form1, text="Target Columns")
        label3.grid(column=1, row=0)
        numericcols = []
        maxlen = 10
        nummaxlen = 10
        for col in self.filein.columns:
            if self.filein[col].dtype == np.float64 or \
              self.filein[col].dtype == np.int64:
                numericcols.append(col)
                if len(col) > nummaxlen:
                    nummaxlen = len(col)
            if len(col) > maxlen:
                maxlen = len(col)
        anvgpcol = self.listbox_and_scrollbar(form1, self.filein.columns,
                                              column=0, row=1,
                                              maxlen=maxlen, multiselect=False)
        anvtgcols = self.listbox_and_scrollbar(form1, numericcols,
                                               column=1, row=1,
                                               maxlen=nummaxlen,
                                               multiselect=True)
        form1.pack()
        button1 = Tk.Button(anovawin, text="run F-Oneway")
        button1["command"] = lambda anovawin=anovawin, anvgpcol=anvgpcol, \
            anvtgcols=anvtgcols: self.anovago(anovawin, anvgpcol, anvtgcols)
        button1.pack()

    def anovago(self, anovawin, anvgpcol, anvtgcols):
        try:
            anovagroupcol = anvgpcol.get(anvgpcol.curselection()[0])
            anovatargetcols = [anvtgcols.get(col)
                               for col in anvtgcols.curselection()]
            pvalshash = {
                    "Column": [],
                    "Count": [],
                    "F-Value": [],
                    "P-Value": [],
                    "Split Column": []
                }
            for targcol in anovatargetcols:
                fval = np.nan
                pval = 1
                # we need to flatten each group into its own list,
                # f_oneway takes lists of lists.
                flexlist = self.filein.groupby(anovagroupcol)[targcol] \
                    .apply(lambda x: x.dropna().tolist())
                runstring = 'stats.f_oneway('
                counter = 0
                totlen = 0
                for sublist in flexlist:
                    counter = counter + 1
                    runstring += str(sublist) + ","
                    totlen += len(sublist)
                runstring = runstring[:-1] + ')'
                # make sure there were at least two groups
                if counter >= 2:
                    try:
                        (fval, pval) = eval(runstring)
                    except Exception as myerr:
                        fval = np.nan
                        pval = 1
                pvalshash["Column"].append(targcol)
                pvalshash["Count"].append(totlen)
                pvalshash["F-Value"].append(fval)
                pvalshash["P-Value"].append(pval)
                pvalshash["Split Column"].append(anovagroupcol)
            # push to an output table
            self.subtable = pd.DataFrame(pvalshash)
            self.subtable = self.subtable.sort_values(by="P-Value")
            window = Tk.Toplevel()
            pnewname = self.table_registry.get_new_table_name()
            newname = "P-Values Table of " + self.table_name
            try:
                self.table_registry.rename_table(pnewname,
                                                 newname, self.subtable)
            except KeyError:
                newname = pnewname
            window.title(newname)
            anovawin.destroy()
            self.pythonwin.new_code(
                    "pvalshash = {\n" + " " * 8 + "\"Column\" : [],\n" +
                    " " * 8 + "\"Count\"  : [],\n" + " " * 8 +
                    "\"P-Value\" : []\n}\n")
            self.pythonwin.new_code(
                    "anovatargetcols = [\"" + "\",\n    \""
                    .join(anovatargetcols) + "\"]")
            self.pythonwin.new_code(
                    "for targcol in anovatargetcols:\n    pval = 1\n    " +
                    "# we need to flatten each group into its own list, " +
                    "f_oneway takes lists of lists.")
            self.pythonwin.new_code(
                    "    flexlist = datatables[\"" + self.table_name +
                    "\"].groupby(\"" + anovagroupcol + "\")[targcol].apply" +
                    "(lambda x: x.dropna().tolist())")
            self.pythonwin.new_code(
                    "    runstring = 'stats.f_oneway('\n    counter = 0\n" +
                    "    totlen = 0\n    for sublist in flexlist:\n")
            self.pythonwin.new_code(
                    "        counter = counter + 1\n        runstring += " +
                    "str(sublist) + \",\"\n        totlen += len(sublist)\n")
            self.pythonwin.new_code(
                    "    runstring = runstring[:-1] + ')'\n    " +
                    "# make sure there were at least two groups\n    " +
                    "if counter >= 2:")
            self.pythonwin.new_code(
                    "        try:\n            (fval,pval) = eval(runstring)")
            self.pythonwin.new_code(
                    "        except:\n            pval = 1\n")
            self.pythonwin.new_code(
                    "    pvalshash[\"Column\"].append(targcol)\n" +
                    "    pvalshash[\"Count\"].append(totlen)\n" +
                    "    pvalshash[\"P-Value\"].append(pval)")
            self.pythonwin.new_code(
                    "#push to an output table\ndatatables[\"" + newname +
                    "\"] = pd.DataFrame(pvalshash)\ndatatables[\"" +
                    newname + "\"] = datatables[\"" + newname +
                    "\"].sort_values(by=\"P-Value\")")
            dashclass(self.subtable, self.basedir, basefile, "child",
                      self.pythonwin, master=window,
                      table_registry=self.table_registry, table_name=newname)
        except Exception as myerr:
            self.tkprint("F-Oneway Exception:", myerr,
                         sys.exc_info()[0], sys.exc_info()[1])

    def rsqmenu(self):
        rsqwin = Tk.Toplevel()
        label1 = Tk.Label(rsqwin,
                          text="Calculate Linear Fit R^2 for multiple targets")
        label1.pack()
        form1 = Tk.Frame(rsqwin)
        label2 = Tk.Label(form1, text="Source Column(s)")
        label2.grid(column=0, row=0)
        label3 = Tk.Label(form1, text="Target Columns")
        label3.grid(column=1, row=0)
        numericcols = []
        maxlen = 10
        nummaxlen = 10
        for col in self.filein.columns:
            if self.filein[col].dtype == np.float64 or \
              self.filein[col].dtype == np.int64:
                numericcols.append(col)
                if len(col) > nummaxlen:
                    nummaxlen = len(col)
            if len(col) > maxlen:
                maxlen = len(col)
        rsqsrccols = self.listbox_and_scrollbar(form1, numericcols,
                                                column=0, row=1,
                                                maxlen=nummaxlen,
                                                multiselect=True)
        rsqtargcols = self.listbox_and_scrollbar(form1, numericcols,
                                                 column=1, row=1,
                                                 maxlen=nummaxlen,
                                                 multiselect=True)
        form1.pack()
        dupvar = Tk.IntVar()
        dupbox = Tk.Checkbutton(rsqwin,
                                text="Ignore targets and run all the " +
                                "sources against each other", variable=dupvar)
        dupbox.pack()
        label4 = Tk.Label(rsqwin,
                          text="NOTE: For large datasets, there may be a " +
                          "delay.  There is no progress bar " +
                          "and no cancel option.")
        label4.pack()
        button1 = Tk.Button(rsqwin, text="Calculate R^2's")
        button1["command"] = lambda rsqwin=rsqwin, dupvar=dupvar, \
            rsqsrccols=rsqsrccols, rsqtargcols=rsqtargcols: \
            self.rsqgo(rsqwin, dupvar, rsqsrccols, rsqtargcols)
        button1.pack()

    def rsqgo(self, rsqwin, dupvar, rsqsrccols, rsqtargcols):
        try:
            srccols = \
                [rsqsrccols.get(col) for col in rsqsrccols.curselection()]
            if dupvar.get() == 1:
                targcols = srccols
            else:
                targcols = [rsqtargcols.get(col)
                            for col in rsqtargcols.curselection()]
            rsqhash = {
                "X Column": [],
                "Y Column": [],
                "R Squared": [],
                "Slope": [],
                "Intercept": [],
                "P-Value": [],
                "Stderr": [],
                "Count": []
            }
            # if we did X by Y, don't need Y by X.
            reversi = {}
            for srccol in srccols:
                for targcol in targcols:
                    if srccol == targcol:
                        continue
                    if srccol + chr(127) + targcol in reversi:
                        continue
                    reversi[targcol + chr(127) + srccol] = True
                    subtab = self.filein[[srccol, targcol]].dropna()
                    if len(subtab) > 0:
                        myslope, myintercept, rval, pval, stderr = \
                            stats.linregress(subtab[srccol], subtab[targcol])
                        rsqhash["X Column"].append(srccol)
                        rsqhash["Y Column"].append(targcol)
                        rsqhash["R Squared"].append(rval ** 2)
                        rsqhash["Slope"].append(myslope)
                        rsqhash["Intercept"].append(myintercept)
                        rsqhash["P-Value"].append(pval)
                        rsqhash["Stderr"].append(stderr)
                        rsqhash["Count"].append(len(subtab))
            self.subtable = pd.DataFrame(rsqhash)
            # get the columns into a nice order
            self.subtable = self.subtable[["X Column", "Y Column",
                                           "R Squared", "Slope", "Intercept",
                                           "P-Value", "Stderr", "Count"]]
            self.subtable = self.subtable.sort_values(by="R Squared",
                                                      ascending=False)
            window = Tk.Toplevel()
            pnewname = self.table_registry.get_new_table_name()
            newname = "R-Squared Table of " + self.table_name
            try:
                self.table_registry.rename_table(pnewname, newname,
                                                 self.subtable)
            except KeyError:
                newname = pnewname
            window.title(newname)
            rsqwin.destroy()
            self.pythonwin.new_code(
                    "srccols = [\n    \"" + "\",\n    \"".join(srccols) +
                    "\"\n]")
            self.pythonwin.new_code(
                    "targcols = [\n   \"" + "\",\n    \"".join(targcols) +
                    "\"\n]")
            self.pythonwin.new_code(
                    "rsqhash = {\n    \"X Column\" : [],\n    \"Y Column\"" +
                    " : [],\n    \"R Squared\" : [],")
            self.pythonwin.new_code(
                    "    \"Slope\"     : [],\n    \"Intercept\" : [],\n" +
                    "    \"P-Value\"   : [],")
            self.pythonwin.new_code(
                    "    \"Stderr\"    : [],\n    \"Count\"     : []\n}\n" +
                    "# if we did X by Y, don't need Y by X.")
            self.pythonwin.new_code(
                    "reversi = {}\nfor srccol in srccols:\n" +
                    "    for targcol in targcols:\n" +
                    "        if srccol == targcol:")
            self.pythonwin.new_code(
                    "            continue\n        " +
                    "if srccol + chr(127) + targcol in reversi:\n" +
                    "            continue")
            self.pythonwin.new_code(
                    "        reversi[targcol + chr(127) + srccol] = True\n" +
                    "        subtab = datatables[\"" + self.table_name +
                    "\"][[srccol, targcol]].dropna()")
            self.pythonwin.new_code(
                    "        if len(subtab) > 0:\n            myslope, " +
                    "myintercept, rval, pval, stderr = stats.linregress" +
                    "(subtab[srccol], subtab[targcol])")
            self.pythonwin.new_code(
                    "            rsqhash[\"X Column\"].append(srccol)\n" +
                    "            rsqhash[\"Y Column\"].append(targcol)")
            self.pythonwin.new_code(
                    "            rsqhash[\"R Squared\"].append(rval ** 2)\n" +
                    "            rsqhash[\"Slope\"].append(myslope)")
            self.pythonwin.new_code(
                    " " * 12 + "rsqhash[\"Intercept\"].append(myintercept)\n" +
                    "            rsqhash[\"P-Value\"].append(pval)")
            self.pythonwin.new_code(
                    "            rsqhash[\"Stderr\"].append(stderr)\n" +
                    "            rsqhash[\"Count\"].append(len(subtab))")
            self.pythonwin.new_code(
                    "datatables[\"" + newname + "\"] = pd.DataFrame(" +
                    "rsqhash)\n#get the columns into a nice order")
            self.pythonwin.new_code(
                    "datatables[\"" + newname + "\"] = datatables[\"" +
                    newname + "\"][[\"X Column\",\"Y Column\"," +
                    "\"R Squared\",\"Slope\",\"Intercept\",\"P-Value\"," +
                    "\"Stderr\",\"Count\"]]")
            self.pythonwin.new_code(
                    "datatables[\"" + newname + "\"] = datatables[\"" +
                    newname + "\"].sort_values(by=\"R Squared\", " +
                    "ascending=False)")
            dashclass(self.subtable, self.basedir, basefile, "child",
                      self.pythonwin, master=window,
                      table_registry=self.table_registry, table_name=newname)
        except Exception as myerr:
            self.tkprint("R^2's Exception:", myerr,
                         sys.exc_info()[0], sys.exc_info()[1])

    def lofmenu(self):
        lofwin = Tk.Toplevel()
        label1 = Tk.Label(lofwin, text="Check Columns for Outliers")
        label1.pack()
        frame1 = Tk.Frame(lofwin)
        numericcols = []
        maxlen = 10
        nummaxlen = 10
        for col in self.filein.columns:
            if self.filein[col].dtype == np.float64 or \
              self.filein[col].dtype == np.int64:
                numericcols.append(col)
                if len(col) > nummaxlen:
                    nummaxlen = len(col)
            if len(col) > maxlen:
                maxlen = len(col)
        label2 = Tk.Label(frame1, text="Label Columns to Match to Results")
        label2.grid(column=0, row=0)
        label3 = Tk.Label(frame1, text="Columns to check for Outliers")
        label3.grid(column=1, row=0)
        labelcols = self.listbox_and_scrollbar(frame1, self.filein.columns,
                                               column=0, row=1,
                                               maxlen=nummaxlen,
                                               multiselect=True)
        lofcols = self.listbox_and_scrollbar(frame1, numericcols,
                                             column=1, row=1,
                                             maxlen=nummaxlen,
                                             multiselect=True)
        frame1.pack()
        button1 = Tk.Button(lofwin, text="Check for Outliers")
        button1["command"] = lambda lofwin=lofwin, labelcols=labelcols, \
            lofcols=lofcols: self.lofgo(lofwin, labelcols, lofcols)
        button1.pack()

    def lofgo(self, lofwin, labelcols, lofcols):
        try:
            try:
                spcols = [labelcols.get(col)
                          for col in labelcols.curselection()]
            except Exception as myerr:
                spcols = []
            xcols = [lofcols.get(col) for col in lofcols.curselection()]
            self.subtable = self.filein[xcols].dropna()
            classifier = neighbors.LocalOutlierFactor(
                    n_neighbors=len(self.subtable) + 1)
            curcols = ["Outlier Prediction Results"]
            for col in self.subtable.columns:
                curcols.append(col)
            self.subtable["Outlier Prediction Results"] = \
                classifier.fit_predict(self.subtable)
            self.subtable = self.subtable[curcols]
            if len(spcols) > 0:
                self.othertable = self.filein[spcols]
                self.subtable = self.subtable.join(self.othertable, on=None,
                                                   how='left',
                                                   lsuffix="_lefttable",
                                                   rsuffix="_righttable")
            window = Tk.Toplevel()
            pnewname = self.table_registry.get_new_table_name()
            newname = "Outliers Detection Table of " + self.table_name
            try:
                self.table_registry.rename_table(pnewname, newname,
                                                 self.subtable)
            except KeyError:
                newname = pnewname
            window.title(newname)
            lofwin.destroy()
            self.pythonwin.new_code(
                    "xcols = [\n    \"" + "\",\n    \"".join(xcols) + "\"\n]")
            self.pythonwin.new_code(
                    "datatables[\"" + newname + "\"] = datatables[\"" +
                    self.table_name + "\"][xcols].dropna()")
            self.pythonwin.new_code(
                    "classifier = neighbors.LocalOutlierFactor(n_neighbors=" +
                    "len(datatables[\"" + newname + "\"]) + 1)")
            self.pythonwin.new_code(
                    "curcols = [\"Outlier Prediction Results\"]\n" +
                    "for col in datatables[\"" + newname + "\"].columns:\n" +
                    "    curcols.append(col)")
            self.pythonwin.new_code(
                    "datatables[\"" + newname + "\"][\"Outlier Prediction " +
                    "Results\"] = classifier.fit_predict(datatables[\"" +
                    newname + "\"])")
            self.pythonwin.new_code(
                    "datatables[\"" + newname + "\"] = datatables[\"" +
                    newname + "\"][curcols]")
            if len(spcols) > 0:
                self.pythonwin.new_code(
                    "spcols = [\n    \"" + "\",\n    \"".join(spcols) +
                    "\"\n]")
                self.pythonwin.new_code(
                    "temptable = datatables[\"" + self.table_name +
                    "\"][spcols]")
                self.pythonwin.new_code(
                    "datatables[\"" + newname + "\"] = datatables[\"" +
                    newname + "\"].join(temptable, on=None, how='left', " +
                    "lsuffix='_lefttable', rsuffix='_righttable')")
            dashclass(self.subtable, self.basedir, basefile, "child",
                      self.pythonwin, master=window,
                      table_registry=self.table_registry, table_name=newname)
        except Exception as myerr:
            self.tkprint("Outliers Finder Exception:", myerr,
                         sys.exc_info()[0], sys.exc_info()[1])

    def fiftyfiftymenu(self):
        fiftyfiftywin = Tk.Toplevel()
        label1 = Tk.Label(fiftyfiftywin,
                          text="Detect Columns with Fifty/Fifty Splits")
        label1.grid(column=0, row=0)
        numericcols = []
        maxlen = 1
        for col in self.filein.columns:
            if self.filein[col].dtype == np.float64 or \
              self.filein[col].dtype == np.int64:
                numericcols.append(col)
                if len(col) > maxlen:
                    maxlen = len(col)
        label2 = Tk.Label(fiftyfiftywin, text="Select Columns to Screen:")
        label2.grid(row=1, column=0)
        fifcols = self.listbox_and_scrollbar(fiftyfiftywin, numericcols,
                                             column=0, row=2, maxlen=maxlen,
                                             multiselect=True)
        fifbylot = Tk.IntVar()
        fifchk = Tk.Checkbutton(fiftyfiftywin,
                                text="By LotId (LotId column must exist)",
                                variable=fifbylot)
        fifchk.grid(row=3, column=0)
        button1 = Tk.Button(fiftyfiftywin, text="Detect")
        button1["command"] = lambda lfiftyfiftywin=fiftyfiftywin, \
            lfifcols=fifcols: self.fiftyfiftygo(lfiftyfiftywin, lfifcols,
                                               fifbylot)
        button1.grid(row=4, column=0)

    def fiftyfiftygo(self, fiftyfiftywin, fifcols, fifbylot):
        pass

    def fiftyfiftyhelper(self, checkcols, outtab, mytab, lotid=None):
        for checkcol in checkcols:
            colaslist = mytab[checkcol].dropna()
            if len(colaslist) > 0:
                colaslist = colaslist.tolist()
                qulmed = np.percentile(colaslist, 30)
                quasilow = np.percentile(colaslist, 40) - \
                    np.percentile(colaslist, 20)
                quasihi = np.percentile(colaslist, 80) - \
                    np.percentile(colaslist, 60)
                quasiavg = np.mean([quasilow, quasihi])
                quhmed = np.percentile(colaslist, 60)
                if quasiavg > 0:
                    factor = (quhmed - qulmed) / quasiavg
                else:
                    factor = 0
                outtab["Column"].append(checkcol)
                outtab["Count"].append(len(colaslist))
                outtab["Difference Factor 50/50"].append(factor)
                qulmed = np.percentile(colaslist, 20)
                quasilow = np.percentile(colaslist, 25) - \
                    np.percentile(colaslist, 15)
                quasihi = np.percentile(colaslist, 80) - \
                    np.percentile(colaslist, 40)
                quhmed = np.median(colaslist)
                quasiavg = np.mean([quasilow, quasihi])
                if quasiavg > 0:
                    factor = (quhmed - qulmed) / quasiavg
                else:
                    factor = 0
                outtab["Difference Factor 1/3rd / 2/3rd"].append(factor)
                qulmed = np.median(colaslist)
                quasilow = np.percentile(colaslist, 60) - \
                    np.percentile(colaslist, 20)
                quasihi = np.percentile(colaslist, 85) - \
                    np.percentile(colaslist, 75)
                quhmed = np.percentile(colaslist, 80)
                quasiavg = np.mean([quasilow, quasihi])
                if quasiavg > 0:
                    factor = (quhmed - qulmed) / quasiavg
                else:
                    factor = 0
                outtab["Difference Factor 2/3rd / 1/3rd"].append(factor)
                outtab["Quantile75"].append(np.percentile(colaslist, 75))
                if lotid is not None:
                    outtab["LotId"].append(lotid)
        return outtab

    def clusmenu(self):
        cluswin = Tk.Toplevel()
        label1 = Tk.Label(cluswin,
                          text="Make Wafermaps of Auto Detected GFA " +
                          "Clusters\nExpects DieX/DieY columns")
        label1.pack()
        frame1 = Tk.Frame(cluswin)
        numericcols = []
        maxlen = 10
        nummaxlen = 10
        for col in self.filein.columns:
            if self.filein[col].dtype == np.float64 or \
              self.filein[col].dtype == np.int64:
                numericcols.append(col)
                if len(col) > nummaxlen:
                    nummaxlen = len(col)
            if len(col) > maxlen:
                maxlen = len(col)
        label2 = Tk.Label(frame1, text="Cluster-Per Column  ")
        label2.grid(column=0, row=0)
        label3 = Tk.Label(frame1, text="  Column with GFA's")
        label3.grid(column=1, row=0)
        cluslabelcols = self.listbox_and_scrollbar(frame1,
                                                   self.filein.columns,
                                                   column=0, row=1,
                                                   maxlen=maxlen,
                                                   multiselect=False)
        clusvalcols = self.listbox_and_scrollbar(frame1, numericcols,
                                                 column=1, row=1,
                                                 maxlen=nummaxlen,
                                                 multiselect=False)
        frame1.pack()
        frame2 = Tk.Frame(cluswin)
        label4 = Tk.Label(frame2, text="Number of Clusters:")
        label4.grid(column=0, row=0)
        numclus = Tk.StringVar()
        numclus.set("5")
        entry1 = Tk.Entry(frame2, textvariable=numclus)
        entry1.grid(column=1, row=0)
        frame2.pack()
        button1 = Tk.Button(cluswin, text="Find Clusters")
        button1["command"] = lambda cluswin=cluswin, \
            cluslabelcols=cluslabelcols, clusvalcols=clusvalcols, \
            numclus=numclus: self.clusgo(cluswin, cluslabelcols,
                                         clusvalcols, numclus)
        button1.pack()

    def clusgo(self, cluswin, cluslabelcols, clusvalcols, numclus):
        try:
            labelcol = cluslabelcols.get(cluslabelcols.curselection()[0])
            targcol = clusvalcols.get(clusvalcols.curselection()[0])
            instancecol = "DieX_DieY"
            mytable = self.filein
            if instancecol not in self.filein.columns:
                newc = mytable.apply(lambda row: str(row["DieX"]) + "^" +
                                     str(row["DieY"]), axis=1)
                mytable.insert(column=instancecol, value=newc,
                               loc=len(mytable.columns))
            inclus = clusterfind(mytable, targcol, labelcol,
                                 instancecol, numclusters=numclus.get())
            newcolname = "Cluster " + targcol
            countup = 2
            while newcolname in self.filein.columns:
                newcolname = "Cluster " + targcol + " " + str(countup)
                countup += 1
            newc = mytable.apply(lambda row: inclus[row[labelcol]], axis=1)
            mytable.insert(column=newcolname, value=newc,
                           loc=len(mytable.columns))
            cluswin.destroy()
            self.doupdate()
            make_wafermap(mytable, targcol, False, newcolname)
            self.pythonwin.new_code(
                    "instancecol = \"DieX_DieY\"\nmytable =" +
                    " datatables[\"" + self.table_name + "\"]\ntargcol = \"" +
                    targcol + "\"\nlabelcol = \"" + labelcol + "\"")
            self.pythonwin.new_code("numclus = " + numclus.get())
            self.pythonwin.new_code("if instancecol not in mytable.columns:")
            self.pythonwin.new_code(
                    "    newc = mytable.apply(lambda row: " +
                    "str(row[\"DieX\"]) + \"^\" + str(row[\"DieY\"]), axis=1)")
            self.pythonwin.new_code(
                    "    mytable.insert(column=instancecol," +
                    " value=newc, loc=len(mytable.columns))\ninclus = " +
                    "clusterfind(mytable, targcol, labelcol, instancecol," +
                    " numclusters=numclus)")
            self.pythonwin.new_code("newcolname = \"" + newcolname + "\"")
            self.pythonwin.new_code(
                    "newc = mytable.apply(lambda row: " +
                    "inclus[row[labelcol]], axis=1)")
            self.pythonwin.new_code(
                    "mytable.insert(column=newcolname, " +
                    "value=newc, loc=len(mytable.columns))\nmake_wafermap(" +
                    "mytable, targcol, False, newcolname)")
            plt.show()
        except Exception as myerr:
            self.tkprint("Cluster Finding Exception:", myerr,
                         sys.exc_info()[0], sys.exc_info()[1])

    def scrollvalmenu(self):
        scrollvalwin = Tk.Toplevel()
        label1 = Tk.Label(scrollvalwin,
                          text="Scroll rows to where a particular " +
                          "column has a particular value")
        label1.grid(column=0, row=0)
        label2 = Tk.Label(scrollvalwin, text="Select Column with target value")
        label2.grid(column=0, row=1)
        maxlen = np.max([len(col) for col in self.filein.columns])
        scrollvalcol = self.listbox_and_scrollbar(scrollvalwin,
                                                  self.filein.columns,
                                                  column=0, row=2,
                                                  maxlen=maxlen,
                                                  multiselect=False)
        button1 = Tk.Button(scrollvalwin, text="Select Column")
        button1["command"] = lambda scrollvalwin=scrollvalwin, \
            scrollvalcol=scrollvalcol: self.scrollvalmenutwo(scrollvalwin,
                                                             scrollvalcol)
        button1.grid(row=3, column=0)

    def scrollvalmenutwo(self, scrollvalwin, scrollvalcol):
        valcol = None
        try:
            valcol = scrollvalcol.get(scrollvalcol.curselection()[0])
        except Exception as myerr:
            self.tkprint("You did not select a column", myerr)
        if valcol is not None:
            try:
                scrollvalwin.destroy()
                scrollvalwintwo = Tk.Toplevel()
                newmenu = sorted(self.filein[valcol].unique().tolist())
                maxlen = np.max([len(str(x)) for x in newmenu])
                label1 = Tk.Label(scrollvalwintwo,
                                  text="Select the value to scroll to:")
                label1.grid(column=0, row=0)
                scrollvalval = self.listbox_and_scrollbar(scrollvalwintwo,
                                                          newmenu, column=0,
                                                          row=1, maxlen=maxlen,
                                                          multiselect=False)
                button1 = Tk.Button(scrollvalwintwo, text="Scroll")
                button1["command"] = lambda scrollvalwintwo=scrollvalwintwo, \
                    valcol=valcol, scrollvalval=scrollvalval: \
                    self.scrollvalgo(scrollvalwintwo, valcol, scrollvalval)
                button1.grid(column=0, row=2)
            except Exception as myerr:
                self.tkprint("Scroll to Value Menu Exception", myerr,
                             sys.exc_info()[0], sys.exc_info()[1])

    def scrollvalgo(self, scrollvalwintwo, valcol, scrollvalval):
        valval = None
        try:
            valval = scrollvalval.get(scrollvalval.curselection()[0])
        except Exception as myerr:
            self.tkprint("You did not select a value to scroll to", myerr)
        if valval is not None:
            try:
                scrollvalwintwo.destroy()
                for counter, (index, row) in enumerate(self.filein.iterrows()):
                    if row[valcol] == valval:
                        self.imin = counter
                        if self.imin > len(self.filein) - self.screensize:
                            self.imin = len(self.filein) - self.screensize
                        self.imax = self.imin + self.screensize
                        self.doupdate()
                        break
            except Exception as myerr:
                self.tkprint("Scroll to Value Exception", sys.exc_info()[0],
                             sys.exc_info()[1], myerr)

    def scrollcolmenu(self):
        scrollcolwin = Tk.Toplevel()
        label1 = Tk.Label(scrollcolwin,
                          text="Scroll columns to a target column")
        label1.grid(row=0, column=0)
        maxlen = np.max([len(col) for col in self.filein.columns])
        scrolltargcol = self.listbox_and_scrollbar(scrollcolwin,
                                                   self.filein.columns,
                                                   column=0, row=1,
                                                   maxlen=maxlen,
                                                   multiselect=False)
        button1 = Tk.Button(scrollcolwin, text="Scroll")
        button1["command"] = lambda scrollcolwin=scrollcolwin, \
            scrolltargcol=scrolltargcol: self.scrollcolgo(scrollcolwin,
                                                          scrolltargcol)
        button1.grid(row=2, column=0)

    def scrollcolgo(self, scrollcolwin, scrolltargcol):
        scrollcol = None
        try:
            scrollcol = scrolltargcol.get(scrolltargcol.curselection()[0])
        except Exception as myerr:
            self.tkprint("You did not select a column.", myerr)
        if scrollcol is not None:
            for counter, col in enumerate(self.filein.columns):
                if col == scrollcol:
                    self.jmin = counter - self.leftheader
                    if self.jmin > len(self.filein.columns) - self.screentabs:
                        self.jmin = len(self.filein.columns) - self.screentabs
                    if self.jmin < 0:
                        self.jmin = 0
                    self.jmax = self.jmin + self.screentabs
                    self.doupdate()
                    break
            scrollcolwin.destroy()


##############################################################################
class PythonWin(Tk.Frame):
    """

    Create a scrollable window that echoes all the Python
    commands the GUI generates

    """
    def __init__(self, master=None):
        Tk.Frame.__init__(self, master)
        self.mainwin = Tk.Toplevel()
        self.mainwin.minsize(width=800, height=200)
        self.mainwin.wm_title("Python Code Echo")
        self.pythonecho = Tk.StringVar()
        self.pythontrack = ""
        self.pyletters = "AA"
        self.mypos = 0
        self.windowheight = 30
        self.pythonlabel = Tk.Label(self.mainwin,
                                    textvariable=self.pythonecho,
                                    justify=Tk.LEFT)
        self.pythonlabel.pack(anchor="w")
        self.button1 = Tk.Button(self.mainwin, text="Up")
        self.button1["command"] = self.goup
        self.button1.pack()
        self.button2 = Tk.Button(self.mainwin, text="Down")
        self.button2["command"] = self.godown
        self.button2.pack()
        self.button3 = Tk.Button(self.mainwin, text="Save")
        self.button3["command"] = self.filesaveas
        self.button3.pack()

        self.mainwin.protocol("WM_DELETE_WINDOW", self.confirm)
        self.mainwin.lower()

    def confirm(self):
        confirmwin = Tk.Toplevel()
        label1 = Tk.Label(confirmwin, text="If you close this window, you " +
                          "will not be able to\nget it back without " +
                          "restarting the program.")
        label1.pack()
        form1 = Tk.Frame(confirmwin)  # type: Frame
        button1 = Tk.Button(form1, text="Don't Close")
        button1["command"] = lambda: confirmwin.destroy()
        button1.grid(column=0, row=0)
        button2 = Tk.Button(form1, text="Understood, so close it please.")
        button2["command"] = lambda lconfirmwin=confirmwin: \
            self.dualdestroy(lconfirmwin)
        button2.grid(column=1, row=0)
        form1.pack()

    def dualdestroy(self, confirmwin):
        confirmwin.destroy()
        self.mainwin.destroy()

    def new_code(self, codestr):
        self.pythontrack += codestr + '\n'
        mylen = len(self.pythontrack.split("\n"))
        self.mypos = mylen - self.windowheight
        if self.mypos < 0:
            self.mypos = 0
        self.doupdate()

    def goup(self):
        self.mypos -= 1
        if self.mypos < 0:
            self.mypos = 0
        self.doupdate()

    def godown(self):
        self.mypos += 1
        mylen = len(self.pythontrack.split("\n"))
        if self.mypos > mylen - self.windowheight:
            self.mypos = mylen - self.windowheight
        if self.mypos < 0:
            self.mypos = 0
        self.doupdate()

    def doupdate(self):
        pmypythontrack = self.pythontrack.split("\n")
        mypythontrack = []
        for counter, value in enumerate(pmypythontrack):
            addstr = str(counter + 1)
            while len(addstr) < 5:
                addstr = " " + addstr
            addstr += ">"
            mypythontrack.append(addstr + " " + value)
        self.pythonecho.set("\n".join(mypythontrack[self.mypos:self.mypos +
                                                    self.windowheight]))

    def filesaveas(self):
        pwfiletypes = (("Text files", "*.txt"), ("All files", "*.*"))
        pwbasefile = filedialog.asksaveasfile(mode='w', filetypes=pwfiletypes)
        # asksaveasfile return `None` if dialog closed with "cancel".
        if pwbasefile is None:
            return
        pwbasefile = pwbasefile.name
        if pwbasefile[-4:].lower() != ".txt" and "." not in pwbasefile:
            pwbasefile += ".txt"
        with open(pwbasefile, 'w') as filehand:
            filehand.write(self.pythontrack)


if __name__ == "__main__":
    tkroot = Tk.Tk()

    # add file selection here
    basedir = r'c:\temp\\'
    filetypes = (("CSV files", "*.csv"), ("All files", "*.*"))
    basefile = filedialog.askopenfilename(initialdir='.',
                                          filetypes=filetypes)

    # read in basefile
    if not os.path.isfile(basefile):
        print("Initial File to Open not found:", basefile)
        sys.exit()
    forcedtype = {
        "Lot": str,
        "Lot Id": str,
        "LotId": str,
        "LOTID": str
    }
    filein = pd.read_csv(basefile, dtype=forcedtype, low_memory=False)

    treg = table_registry()
    newname = treg.get_new_table_name()
    path, myname = os.path.split(basefile)
    treg.rename_table(newname, myname, table_reference=filein)
    tkroot.title("Table: " + myname)

    pythonwinhandle = PythonWin()

    myform = dashclass(filein, basedir, basefile, "parent",
                       pythonwin=pythonwinhandle, master=tkroot,
                       table_registry=treg, table_name=myname)

    pythonwinhandle.new_code(
            "#autogenerated code, it wouldn't hurt to check" +
            " it for correctness.\n#Not all imports may be necessary for " +
            "all scripts\n#You may need to add a plt.show() to the end or " +
            "otherwise use your results.\n\nimport os\nimport sys\n" +
            "import shutil\nimport matplotlib.pyplot as plt\n" +
            "import pandas as pd\nimport numpy as np\n" +
            "import statsmodels.api as sm\nimport seaborn as sns\n")
    pythonwinhandle.new_code(
            "from scipy import stats\nfrom matplotlib " +
            "import gridspec\nfrom statsmodels.formula.api import ols\n" +
            "from sklearn import neighbors\n\nfrom " +
            "recently_updated_makegraphs_script import oneprobplot\n" +
            "from recently_updated_yieldmosaic import " +
            "colorgen, yieldmosaic_function")
    pythonwinhandle.new_code(
            "from waterfallchart import waterfallchart\n" +
            "from pymaps import make_wafermap\nfrom pydiemaps import " +
            "make_wafermap as make_diemap")
    pythonwinhandle.new_code(
            "from bycolorgraph import bycolorgraph\n" +
            "from trendator_quickturn import convert_to_epoch\nfrom " +
            "clusterfind import clusterfind\n")

    pythonwinhandle.new_code("datatables = dict()")
    pythonwinhandle.new_code("datatables[\"" + myname + "\"] = " +
                             "pd.read_csv(\"" + basefile +
                             "\", low_memory=False)")

    # set a color scheme for FacetGrid Here.
    # That way, the user can override it later.
    tab20_for_earlier_values = [
        # R G B
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229]
    ]
    # this looks ugly, but it makes the above look nicer.
    # Matplotlib takes fractions 0-1
    for index in range(len(tab20_for_earlier_values)):
        for jndex in range(len(tab20_for_earlier_values[index])):
            tab20_for_earlier_values[index][jndex] /= 255
    cmap = plt.cm.jet
    tab20_for_earlier = cmap.from_list('tab20_for_earlier',
                                       tab20_for_earlier_values,
                                       len(tab20_for_earlier_values))
    sns.set_palette(tab20_for_earlier_values)

    tkroot.mainloop()
