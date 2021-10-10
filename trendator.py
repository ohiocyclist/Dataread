"""
trendator.py

"""
import pandas as pd
from scipy import stats
from scipy import special as scipysp #erfc for probplots
import numpy as np
import time
import subprocess
import re
import statistics
import matplotlib.pyplot as plt

def movingaverage(window_size, smootharray):
    """
    smooth out a moving average trend line
    Just a raw average of window_size not an ewma
    
    """
    # smooth forward and backwards
    back = int(window_size / 2)
    forward = window_size - back
    retvals = []
    for x in range(len(smootharray)):
        start = x - back
        if start < 0:
            start = 0
        end = x + forward
        if end > len(smootharray) - 1:
            end = len(smootharray) - 1
        avg = 0
        count = 0
        for y in range(start, end):
            if pd.notnull(smootharray[y]):
                avg = avg + smootharray[y]
                count = count + 1
        if count > 0:
            retvals.append(avg / count)
        else:
            retvals.append(smootharray[x])
    return retvals
    
#### rest of module To Be Implemented