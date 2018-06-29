# -*- coding: utf-8 -*-
"""
Created on Tue May  8 14:46:45 2018

@author: Administrator
"""

from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "1984-01-01/to/2013-12-31",
    "expver": "1",
    "grid": "0.125/0.125",
    "levelist": "500/600/700/775/850/925/1000",
    'area': "40/115/38/117",
    "levtype": "pl",
    "param": "130.128/157.128/131.128/132.128",
    "step": "0",
    "stream": "oper",
    "time": "00:00:00/06:00:00/12:00:00/18:00:00",
    "type": "an",
    'format': "netcdf",
    "target": r"E:\python3.6\test\output_interim_beijing.nc",
})