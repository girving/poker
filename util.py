'''Miscellaneous utilities'''

from __future__ import division,absolute_import

from numpy import *
from fractions import Fraction

def amap(f,x):
    x = asanyarray(x)
    return array(map(f,x.ravel())).view(type(x)).reshape(x.shape)

def fractions(x):
    return amap(Fraction,asarray(x,object))
