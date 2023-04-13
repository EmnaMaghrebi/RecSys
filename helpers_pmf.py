import numpy as np
import pandas as pd


#map values to discrete ratings 
# x: value to be mapped
def condition(x):
    if x<-2:
        return 1
    elif x>=-2 and x<=-0.5:
        return 2
    elif x>-0.5 and x<=0.5:
        return 3
    elif x>0.5 and x<=2:
        return 4
    else:
        return 5
