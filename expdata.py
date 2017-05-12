
import numpy as np 

def setexperimentdata(problem): 
    problemlist = np.array([0,1 ])
    IN = np.array([4,13 ])
    HID = np.array([6,6 ])
    OUT = np.array([2,3 ])
     baseNet = [IN[problem], HID[problem], OUT[problem]] 
 

    if problem == 0:
       TrDat= np.loadtxt("iristrain.txt") #  iris dataset
       TsDat = np.loadtxt("iristest.txt") #  

    elif problem == 1:
       TrDat = np.loadtxt("winetrain.txt") #  Wine data
       TsDat = np.loadtxt("winetest.txt") #  
	 
 
    return [TrainData, TestData, baseNet]
