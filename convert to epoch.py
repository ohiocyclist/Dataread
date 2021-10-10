#!/usr/bin/env python
# coding: utf-8

# In[1]:


def convert_to_epoch(trendcsv, dtcol='StartDateTime', epochname='Unix_Epoch_Time', pattern='%m/%d/%Y %I:%M:%S %p', newloc=3):
    if dtcol not in trendcsv.columns:
        print(">>> convert to epoch warninig: could not find your date column.  Will attempt to discover one.")
        for thiscol in trendcsv.columns:
            if 'date' in thiscol.lower():
                dtcol = thiscol
                break
    if dtcol not in trendcsv.columns:
        print(">>> convert to epoch error: could not find any sort of date column")
        return trendcsv
    trendcsv.dropna(subset=[dtcol], inplace=True)
    trendcsv.insert(value=range(len(trendcsv)), column='newdex', loc=len(trendcsv.columns))
    trendcsv.set_index(['newdex'], inplace=True, drop=True, append=False)
    del trendcsv.index.name
    
    datecol = trendcsv.loc[:, dtcol]
    for tconvert in datecol:
        try:
            epoch = int(time.mktime(time.strptime(str(tconvert), pattern)))
        except ValueError:
            pattern = "%Y-%m-%d %H:%M:%S"
        break
    for tconvert in datecol:
        try:
            epoch = int(time.mktime(time.strptime(str(tconvert), pattern)))
        except ValueError:
            pattern = "%Y/%m/%d %H:%M:%S"
        break
    for tconvert in datecol:
        try:
            epoch = int(time.mktime(time.strptime(str(tconvert), pattern)))
        except ValueError:
            pattern = "%Y/%m/%d %I:%M:%S %p"
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


# In[ ]:




