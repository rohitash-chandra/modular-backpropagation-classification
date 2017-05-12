
import numpy as np 

def setexperimentdata(problem): 
    problemlist = np.array([0,1,2,3,4,5,6,7,8,9,10, 11, 12])
    IN = np.array([4,13,9,13,15, 4, 9, 34, 16, 4, 4, 4, 24])
    HID = np.array([6,6,6,16,20, 5, 30, 8, 6, 5, 8, 14, 14])
    OUT = np.array([2,3,1,1,1, 1, 1, 1, 7, 3, 3, 4, 4])
    filenames = ["Data/Iris/", "Data/Wine/", "Data/Cancer/", "Data/Heart/",  "Data/CreditApproval/", "Data/Baloon/", "Data/TicTac/", "Data/Ions/", "Data/Zoo/", "Data/Lenses/", "Data/Balance/", "Data/Robot-Four/", "Data/Robot-TwentyFour/"]
    baseNet = [IN[problem], HID[problem], OUT[problem]] 
 

    if problem == 0:
       TrDat= np.loadtxt("Data/Iris/train.txt") #  iris dataset
       TsDat = np.loadtxt("Data/Iris/test.txt") #  

    elif problem == 1:
       TrDat = np.loadtxt("Data/Wine/train.txt") #  Wine data
       TsDat = np.loadtxt("Data/Wine/test.txt") #  
	
    elif problem == 2:
       TrDat = np.loadtxt("Data/Cancer/train.txt") #  Ccncer data
       TsDat = np.loadtxt("Data/Cancer/test.txt") #  
		
    elif problem == 3:
       TrDat = np.loadtxt("Data/Heart/train.txt") #  Heart data
       TsDat = np.loadtxt("Data/Heart/test.txt") #   

    elif problem == 4:
       TrDat = np.loadtxt("Data/CreditApproval/train.txt") #  Credit Approval data
       TsDat = np.loadtxt("Data/CreditApproval/test.txt") # 

    elif problem == 5:
       TrDat = np.loadtxt("Data/Baloon/train.txt") #  Baloon data
       TsDat = np.loadtxt("Data/Baloon/test.txt") # 

    elif problem == 6:
       TrDat = np.loadtxt("Data/TicTac/train.txt") #  Tic Tac Toe data
       TsDat = np.loadtxt("Data/TicTac/test.txt") # 
	
    elif problem == 7:
       TrDat = np.loadtxt("Data/Ions/train.txt") #  Ionsphere data
       TsDat = np.loadtxt("Data/Ions/test.txt") # 
	
    elif problem == 8:
       TrDat = np.loadtxt("Data/Zoo/train.txt") #  Zoo data
       TsDat = np.loadtxt("Data/Zoo/test.txt") # 
	
    elif problem == 9:
       TrDat = np.loadtxt("Data/Lenses/train.txt") #  Lenses data
       TsDat = np.loadtxt("Data/Lenses/test.txt") # 

    elif problem == 10:
       TrDat = np.loadtxt("Data/Balance/train.txt") #  Balance data
       TsDat = np.loadtxt("Data/Balance/test.txt") # 

    elif problem == 11:
       TrDat = np.loadtxt("Data/Robot-Four/train.txt") #  Balance data
       TsDat = np.loadtxt("Data/Robot-Four/test.txt") # 
	
    elif problem == 12:
       TrDat = np.loadtxt("Data/Robot-TwentyFour/train.txt") #  Balance data
       TsDat = np.loadtxt("Data/Robot-TwentyFour/test.txt") #  


    Hidden = HID[problem]
    Input = IN[problem]
    Output = OUT[problem]
		
    traindt = TrDat[:,np.array(range(0,IN[problem]))]	
    dt = np.amax(traindt, axis=0)
    dt =[1 if x==0 else x for x in dt]
    tds = abs(traindt/dt)
	
    testdt = TsDat[:,np.array(range(0,IN[problem]))]	
    dst = np.amax(testdt, axis=0)
    dst =[1 if x==0 else x for x in dst]
    tdst = abs(testdt/dst)

		
    if(problem == 2): # adjust dataset 
	outTraindat = TrDat[:,np.array(range(IN[problem],IN[problem]+OUT[problem]))]
	outTraindat[outTraindat ==2] = 0 
	outTraindat [outTraindat ==4] = 1
	TrainData  = np.concatenate(( tds[:,range(0,IN[problem])], outTraindat[:,range(0,OUT[problem])]), axis=1)

	outTestdata = TsDat[:,np.array(range(IN[problem],IN[problem]+OUT[problem]))]
	outTestdata[outTestdata==2]=0
	outTestdata[outTestdata==4]=1
	TestData  = np.concatenate(( tdst[:,range(0,IN[problem])],outTestdata[:,range(0,OUT[problem])]) , axis=1)

    elif(problem == 3):  # adjust dataset 
	outTraindat = TrDat[:,np.array(range(IN[problem],IN[problem]+OUT[problem]))]
	outTraindat[outTraindat >1] = 1 
	TrainData  = np.concatenate(( tds[:,range(0,IN[problem])], outTraindat[:,range(0,OUT[problem])]), axis=1)

	outTestdata = TsDat[:,np.array(range(IN[problem],IN[problem]+OUT[problem]))]
	outTestdata[outTestdata>1]=1
	TestData  = np.concatenate(( tdst[:,range(0,IN[problem])],outTestdata[:,range(0,OUT[problem])]) , axis=1)
     
    elif(problem == 5):  # adjust dataset 
	TrainData  =  TrDat
	TestData  = TsDat
    else:  # adjust dataset 
	TrainData  = np.concatenate(( tds[:,range(0,IN[problem])], TrDat[:,range(IN[problem],IN[problem]+OUT[problem])]), axis=1)
	TestData  = np.concatenate(( tdst[:,range(0,IN[problem])], TsDat[:,range(IN[problem],IN[problem]+OUT[problem])]), axis=1)

    return [TrainData, TestData, baseNet]
