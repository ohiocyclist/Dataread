import os
import re
import sys
import decimal
import pandas as pd
import numpy as np

import probplot

def oneprobplot(onelot, thiscol, directory, makethismap=True, uslref=False,
                lslref=False, withboxplots=True, keepplot=False,
                splitcol="Split", **kwargs):
    
    (thisplt, isok, scaletop, scalebot, ppax) = \
        probplot.probplot(onelot, thiscol, splitcol,
                          connect_points=True,
                          withboxplots=withboxplots, **kwargs)
        
        # skip some stuff for adding lsl/usl reference lines
        
