
import numpy as np 

def setexperimentdata(problem): 
    problemlist = np.array([0,1 ])
    IN = np.array([4,13 ])
    HID = np.array([6,6 ])
    OUT = np.array([2,3 ])
     baseNet = [IN[problem], HID[problem], OUT[problem]] 
 

    if problem == 0:
       TrDat= np.loadtxt("Iris/train.txt") #  iris dataset
       TsDat = np.loadtxt("Iris/test.txt") #  

    elif problem == 1:
       TrDat = np.loadtxt("Wine/train.txt") #  Wine data
       TsDat = np.loadtxt("Wine/test.txt") #  
	 
 
    return [TrainData, TestData, baseNet]
