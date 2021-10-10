#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dirdeep directory tree spanner

Where is the space on the disk drive going?

Created on Mon Feb 15 09:52:26 2021

@author: bbuck
"""

import os
import pandas as pd

basedir = 'c:/'
dirdeep = {
    'File': [],
    'Directory': [],
    'Size': []
    }

for counter in range(1, 8):
    dirdeep['dirsub_' + str(counter)] = []
    
def recurse(basedir, dirdeep):
    print(basedir)
    try:
        mydirs = [d.name for d in os.scandir(basedir) if os.path.isdir(basedir + d.name) and not d.name.endswith('~') and not os.path.islink(basedir + d.name)]
    except PermissionError:
        print("Permission Error")
        mydirs = []
    except OSError:
        print("OS ERROR")
        mydirs = []
    for mydir in mydirs:
        dirdeep = recurse(basedir + mydir + '/', dirdeep)
    try:
        myfiles = [f.name for f in os.scandir(basedir) if os.path.isfile(basedir + f.name) and not f.name.endswith('~') and not os.path.islink(basedir + f.name)]
    except PermissionError:
        print("Files Permission Error")
        return dirdeep
    except OSError:
        return dirdeep
    for myfile in myfiles:
        try:
            mysize = os.path.getsize(basedir + myfile)
        except PermissionError:
            print("can't get size", myfile)
        dirdeep['File'].append(myfile)
        dirdeep['Directory'].append(basedir)
        dirdeep['Size'].append(mysize)
        brokenbase = basedir.split('/')
        for counter in range(1, 8):
            if len(brokenbase) >= counter:
                dirdeep['dirsub_' + str(counter)].append(brokenbase[counter - 1])
            else:
                dirdeep['dirsub_' + str(counter)].append('')
    return dirdeep

dirdeep = recurse(basedir, dirdeep)

#pd.DataFrame(dirdeep).to_csv('~/Documents/filelist.csv', index=False)
pd.DataFrame(dirdeep).to_csv('d:/filelist.csv', index=False)