# Rohitash Chandra, 2017 c.rohitash@gmail.conm

#!/usr/bin/python

# built using: https://github.com/rohitash-chandra/VanillaFNN-Python


#Sigmoid units used in hidden and output layer. gradient descent and stocastic gradient descent functions implemented with momentum

 

from expdata import setexperimentdata


import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time

#An example of a class
class Network:

    def __init__(self, Topo, Train, Test, MaxTime,  MinPer):

    #def __init__(self, Topo, Train, Test, MaxTime, Samples, MinPer):
        self.Top  = Topo  # NN topology [input, hidden, output]
        self.Max = MaxTime # max epocs
        self.TrainData = Train
        self.TestData = Test
        self.NumSamples = Train.shape[0]

        self.lrate  = 0 # will be updated later with BP call

        self.momenRate = 0
        self.useNesterovMomen = 0 #use nestmomentum 1, not use is 0

        self.minPerf = MinPer
                                        #initialize weights ( W1 W2 ) and bias ( b1 b2 ) of the network
    	np.random.seed()
	self.W1 = np.random.randn(self.Top[0]  , self.Top[1])  / np.sqrt(self.Top[0] )
        self.B1 = np.random.randn(1  , self.Top[1])  / np.sqrt(self.Top[1] ) # bias first layer
        self.BestB1 = self.B1
        self.BestW1 = self.W1
    	self.W2 = np.random.randn(self.Top[1] , self.Top[2]) / np.sqrt(self.Top[1] )
        self.B2 = np.random.randn(1  , self.Top[2])  / np.sqrt(self.Top[1] ) # bias second layer
        self.BestB2 = self.B2
        self.BestW2 = self.W2
        self.hidout = np.zeros((1, self.Top[1] )) # output of first hidden layer
        self.out = np.zeros((1, self.Top[2])) #  output last layer

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def printNet(self):
        print self.Top
        print self.W1

    def sampleEr(self,actualout):
        error = np.subtract(self.out, actualout)
        sqerror= np.sum(np.square(error))/self.Top[2]
        #print sqerror
        return sqerror

    def ForwardPass(self, X ):
         z1 = X.dot(self.W1) - self.B1
         self.hidout = self.sigmoid(z1) # output of first hidden layer
         z2 = self.hidout.dot(self.W2)  - self.B2
         self.out = self.sigmoid(z2)  # output second hidden layer



    def BackwardPassMomentum(self, Input, desired, vanilla):
            out_delta =   (desired - self.out)*(self.out*(1-self.out))
            hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1-self.hidout))

            if vanilla == 1: #no momentum
                self.W2+= (self.hidout.T.dot(out_delta) * self.lrate)
                self.B2+=  (-1 * self.lrate * out_delta)
                self.W1 += (Input.T.dot(hid_delta) * self.lrate)
                self.B1+=  (-1 * self.lrate * hid_delta)

            else:
 	                  # momentum http://cs231n.github.io/neural-networks-3/#sgd
            	self.W2 += ( self.W2 *self.momenRate) + (self.hidout.T.dot(out_delta) * self.lrate)       # velocity update
            	self.W1 += ( self.W1 *self.momenRate) + (Input.T.dot(hid_delta) * self.lrate)
                self.B2 += ( self.B2 *self.momenRate) + (-1 * self.lrate * out_delta)       # velocity update
            	self.B1 += ( self.B1 *self.momenRate) + (-1 * self.lrate * hid_delta)





    def TestNetwork(self, phase,   erTolerance):
        Input = np.zeros((1, self.Top[0])) # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        nOutput = np.zeros((1, self.Top[2]))
        if phase == 1:
           Data = self.TestData
        if phase == 0:
           Data = self.TrainData
        clasPerf = 0
     	sse = 0
        testSize = Data.shape[0]
        self.W1 = self.BestW1
        self.W2 = self.BestW2 #load best knowledge
        self.B1 = self.BestB1
        self.B2 = self.BestB2 #load best knowledge

        for s in xrange(0, testSize):

                Input[:]  =   Data[s,0:self.Top[0]]
                Desired[:] =  Data[s,self.Top[0]:]

                self.ForwardPass(Input )
                sse = sse+ self.sampleEr(Desired)


                if(np.isclose(self.out, Desired, atol=erTolerance).any()):
                   clasPerf =  clasPerf +1

   	return ( sse/testSize, float(clasPerf)/testSize * 100 )


    def saveKnowledge(self):
        self.BestW1 = self.W1
        self.BestW2 = self.W2
        self.BestB1 = self.B1
        self.BestB2 = self.B2

    def BP_GD(self, learnRate, mRate,    stocastic, vanilla, depth): # BP with SGD (Stocastic BP)
        self.lrate = learnRate
        self.momenRate = mRate

        Input = np.zeros((1, self.Top[0])) # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        #Er = []#np.zeros((1, self.Max))
        epoch = 0
        bestmse = 100
        bestTrain = 0
        #while  epoch < self.Max and bestTrain < self.minPerf :
        while epoch < depth:
            sse = 0
            for s in xrange(0, self.NumSamples):

                if(stocastic):
                   pat = random.randint(0, self.NumSamples-1)
                else:
                   pat = s

                Input[:]  =  self.TrainData[pat,0:self.Top[0]]
                Desired[:] = self.TrainData[pat,self.Top[0]:]



                self.ForwardPass(Input )
                self.BackwardPassMomentum(Input , Desired, vanilla)
                sse = sse+ self.sampleEr(Desired)

            mse = np.sqrt(sse/self.NumSamples*self.Top[2])

            if mse < bestmse:
               bestmse = mse
               self.saveKnowledge()
               (x,bestTrain) = self.TestNetwork(0,  0.2)


            #Er = np.append(Er, mse)


            epoch=epoch+1

        return (mse,bestmse, bestTrain, epoch)

#--------------------------------------------------------------------------------------------------------





class MTnetwork: # Multi-Task leaning using Stocastic GD

    def __init__(self, mtaskNet, trainData, testData, maxTime, minPerf, learnRate, numModules, transKnow):
          #trainData and testData could also be different datasets. this example considers one dataset
	self.transKnowlege = transKnow
	self.trainData = trainData
	self.testData = testData
	self.maxTime = maxTime
	self.minCriteria = minPerf
        self.numModules = numModules # number of network modules (i.e tasks with shared knowledge representation)
                           # need to define network toplogies for the different tasks.

        self.mtaskNet = mtaskNet

        self.learnRate = learnRate
        self.trainTolerance = 0.20 # [eg 0.15 output would be seen as 0] [ 0.81 would be seen as 1]
        self.testTolerance = 0.49


    def transferKnowledge(self, Wprev, Wnext): # transfer knowledge (weights from given layer) from  Task n (Wprev) to Task n+1 (Wnext)
        x=0
        y = 0
        Wnext[x:x+Wprev.shape[0], y:y+Wprev.shape[1]] = Wprev                                   #(Netlist[n].W1 ->  Netlist[n+1].W1)
        return Wnext

    def mainAlg(self):

        mRate = 0.05

        stocastic = 1 # 0 for vanilla BP. 1 for Stocastic BP
        vanilla = 1 # 1 for Vanilla Gradient Descent, 0 for Gradient Descent with momentum


        Netlist = [None]*10  # create list of Network objects ( just max size of 10 for now )


        trainPerf = np.zeros(self.numModules)
        trainMSE =  np.zeros(self.numModules)
        testPerf = np.zeros(self.numModules)
        testMSE =  np.zeros(self.numModules)

        erPlot = np.random.randn(self.maxTime ,self.numModules)
         # plot of convergence for each module (Netlist[n] )

        depthSearch = 5 #declare


        for n in xrange(0, self.numModules):
            module = self.mtaskNet[n]
            taskfeatures = module[0]
            trdata = taskdata(self.trainData, module[0],  module[2] )   # make the partitions for task data
            testdata = taskdata(self.testData, module[0],  module[2] )
            Netlist[n] = Network(self.mtaskNet[n], trdata, testdata, depthSearch,  self.minCriteria)


        cycles = 0
        index = 0
        current = 0
        while(current) <(self.maxTime): #*self.numModules
            cycles =cycles + 1

            for n in xrange(0, self.numModules-1):
                #if n == 0:
                #depthSearch = 10-(n*3) # random.randint(1, 5) #stocastic depth
                #print depthSearch
                #else:
                   #depthSearch = 1
                current = current + depthSearch
                #print current, depthSearch
            	(erPlot[index, n],  trainMSE[n], trainPerf[n], Epochs) = Netlist[n].BP_GD(self.learnRate, mRate,   stocastic, vanilla, depthSearch)

                if(self.transKnowlege ==1):
            		Netlist[n+1].W1 = self.transferKnowledge(Netlist[n].W1, Netlist[n+1].W1)
            		Netlist[n+1].B1 = self.transferKnowledge(Netlist[n].B1, Netlist[n+1].B1)
            		Netlist[n+1].W2 = self.transferKnowledge(Netlist[n].W2, Netlist[n+1].W2)
            		Netlist[n+1].B2 = self.transferKnowledge(Netlist[n].B2, Netlist[n+1].B2)
            (erPlot[index, self.numModules-1],  trainMSE[self.numModules-1], trainPerf[self.numModules-1], Epochs) = Netlist[self.numModules-1].BP_GD(self.learnRate, mRate, stocastic, vanilla, depthSearch) # BP for last module
            index = index + 1
            #print trainPerf, cycles
        for n in xrange(0, self.numModules):
            (testMSE[n], testPerf[n]) = Netlist[n].TestNetwork(1, self.testTolerance) # 1 in argument means to use testdata


        return (erPlot, trainMSE, trainPerf, testMSE, testPerf)

def normalisedata(data, inputsize, outsize): # normalise the data between [0,1]  (not used in this prog)
    traindt = data[:,np.array(range(0,inputsize))]
    dt = np.amax(traindt, axis=0)
    tds = abs(traindt/dt)
    return np.concatenate(( tds[:,range(0,inputsize)], data[:,range(inputsize,inputsize+outsize)]), axis=1)



def taskdata(data, taskfeatures, output): # group taskdata from main data source
    datacols = data.shape[1]
    featuregroup = data[:,0:taskfeatures]
    return np.concatenate(( featuregroup[:,range(0,taskfeatures)], data[:,range(datacols-output,datacols)]), axis=1)



# ------------------------------------------------------------------------------------------------------



def main():



	#os.remove('out2_randomdepth.txt')
        np.random.seed()
	maxEpoch = np.array([500,500 ])
        learnRate = 0.5
        fileout1 =  open('out1_res.txt','a')
        fileout2 =  open('out2_res.txt','a')

        moduledecomp = [0.25, 0.5, 0.75, 1]  # decide what will be number of features for each group of taskdata correpond to module


	for problem in range(0, 13):
	        [TrainData, TestData, base] = setexperimentdata(problem)

                MaxTime = maxEpoch[problem]



		TrSamples =  np.size(TrainData,0)
		TestSize = np.size(TestData,0)




 		MaxRun = 30  # number of experimental runs

        	MinCriteria = 97 #stop when learn 95 percent

       	        numModules = 4 # first decide number of  modules (or ensembles for comparison)
                inputfeatures = base[0] # total num inputfeatures for the prob

                mtaskNet =   np.array([base, base,base,base])
        	for i in xrange(1, numModules):

             	     mtaskNet[i-1][0]  =  moduledecomp[i-1] * inputfeatures
             	     mtaskNet[i][1] += (i*2) # in this example, we have fixed numner  output neurons. input for each task is termined by feature group size.
                                         # we adapt the number of hidden neurons for each task.
        	print mtaskNet # print network topology of all the modules that make the respective tasks. Note in this example, the tasks aredifferent network topologies given by hiddent number of hidden layers.

        	trainPerf = np.random.randn(MaxRun,numModules)
        	testPerf =  np.random.randn(MaxRun,numModules)
        	meanTrain =  np.zeros(numModules)
        	stdTrain =  np.zeros(numModules)
        	meanTest =  np.zeros(numModules)
        	stdTest =  np.zeros(numModules)
                x =   np.zeros(numModules)

        	trainMSE =  np.random.randn(MaxRun,numModules)
        	testMSE =  np.random.randn(MaxRun,numModules)
        	Epochs =  np.zeros(MaxRun)
        	Time =  np.zeros(MaxRun)



        	for transKnow in xrange(1, 2  ): # transKnow = 0 # 1 is for MT knowledge transfer. 0 is for no transfer (simple ensemble learning)
        		for run in xrange(0, MaxRun  ):
        			mt = MTnetwork(mtaskNet, TrainData, TestData, MaxTime,MinCriteria,learnRate, numModules,transKnow)
        			(erPlot, trainMSE[run,:], trainPerf[run,:], testMSE[run,:], testPerf[run,:]) = mt.mainAlg()
                		x = [problem, transKnow, run]
                                print x, testPerf[run,:]
                                #np.savetxt(fileout1, (problem, transKnow, run ), fmt='%1.1f')
                                #np.savetxt(fileout1, (testPerf[run,:]), fmt='%1.2f',  newline=' ')


       			for module in xrange(0, numModules ):
            			meanTrain[module] = np.mean(trainPerf[:,module])
       	    			stdTrain[module] = np.std(trainPerf[:,module])
            			meanTest[module] = np.mean(testPerf[:,module])
       	    			stdTest[module] = np.std(testPerf[:,module])


        		print meanTrain
        		print stdTrain
       			print meanTest
        		print stdTest
                        np.savetxt(fileout2, (problem, transKnow ), fmt='%1.1f'     )
			np.savetxt(fileout2, (meanTrain,stdTrain,meanTest,stdTest ), fmt='%1.2f')

if __name__ == "__main__": main()
