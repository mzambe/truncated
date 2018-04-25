import os
import numpy as np
import math

def diag(arr):
    d = len(arr)
    rm = np.matrix(np.zeros((d, d), float))
    for i in range(d):
        rm[i, i] = arr[i]
    
    return rm

def matSqrt(A):
    decA = np.linalg.svd(A)
    midA = diag(np.sqrt(decA[1]))
    rm = decA[0]*midA*decA[2]
    
    return rm

class box:
    def __init__(self, dimensions, parameters):
        
        self.d    = dimensions
        self.par  = parameters

    def incl(self, point):
        for i in range(self.d):
            if (point[i] < self.par[i][0]) or (point[i] > self.par[i][1]):
                return False
        
        return True

class space:
    def __init__(self, dimensions, parameters=None):
        
        self.d    = dimensions
        self.par  = parameters

    def incl(self, point):
        return True


class multivariateGaussian:
    def __init__(self, dimensions, mean, covariance, truncation=None):
        self.d    = dimensions
        self.mean = mean
        self.std  = matSqrt(covariance)
        self.var  = covariance
        if (truncation == None):
            self.cset = space(self.d, [])
        else:
            self.cset = truncation
    
    def getSample(self):
        bincl = False
        while(not bincl):
            xs = np.zeros((self.d, 1))
            for i in range(self.d):
                xs[i] = np.random.normal(0, 1)

            xf = self.std * xs
            xf = xf + self.mean
            bincl = self.cset.incl(xf)

        return xf
    
class truncatedLearning:
    def __init__(self, dimensions, truncation, samples, epsilon=0.1, bigConstant=10.0):
        
        self.d       = dimensions
        self.eps     = epsilon
        self.cons    = 10.0
        self.iterNum = 400
        
        self.cset = truncation
        self.smp  = samples
        self.mean = np.zeros((self.d, 1), float)
        self.var  = diag(np.ones(self.d, float))
        self.std  = matSqrt(self.var)
        self.varT = self.var.getI()
        self.varn = self.varT * self.mean
  
    def updateReal(self):
        self.var  = self.varT.getI()
        self.mean = self.var * self.varn
        
        self.std  = matSqrt(self.var)
        
    def updateFake(self):
        self.varT = self.var.getI()
        self.varn = self.varT * self.mean
        
        self.std  = matSqrt(self.var)
    
    def getReal(self):
        return self.mean, self.var

    def getFake(self):
        return self.varn, self.varT
    
    def getFakeSample(self):
        bincl = False
        
        xs = np.zeros((self.d, 1))
        for i in range(self.d):
            xs[i] = np.random.normal(0, 1)

        xf = self.std * xs
        xf = xf + self.mean
        bincl = self.cset.incl(xf)

        return (float(bincl)), (float(bincl))*xf

    def initialEstimations(self):
        # Empirical Mean Estimation
        meanSampl = self.cons*(self.d / ((self.eps)**2))
        
        meanEs = self.mean
        for i in range(int(meanSampl)):
            sampl = self.smp.getSample()
            meanEs = meanEs + sampl
        
        meanEs = (1.0/float(meanSampl))*meanEs
        
        # Empirical Covariance Estimation
        covSampl = self.cons*(((self.d)**2) / ((self.eps)**2))
        
        varEs = self.var
        for i in range(int(covSampl)):
            sampl = self.smp.getSample()
            varEs = varEs + (sampl * sampl.getT())
        
        varEs = (1.0/float(covSampl))*varEs
        
        self.mean = meanEs
        self.var  = varEs
        
        self.updateFake()
        
    def estimateLikelihood(self):
        sampl = self.smp.getSample()
        
        iterNum = 200

        # Estimate Likelihood
        diffx = sampl - self.mean
        likelihoodEst = (0.5)*(diffx.getT() * self.varT * diffx)
        
        cnt = 1
        for i in range(iterNum):
            sampl = self.smp.getSample()

            # Estimate Likelihood
            diffx = sampl - self.mean
            likelihoodEst = likelihoodEst + (0.5)*(diffx.getT() * self.varT * diffx)
            
            cnt = cnt + 1

        likelihoodEst = (1.0/float(cnt))*likelihoodEst

        # Estimate Truncated Mass
        bincl, fsamp = self.getFakeSample()
        
        countIncl = int(bincl)
        
        # Estimate Mass
        cnt = 1
        
        while (countIncl <= iterNum/10):
            bincl, fsamp = self.getFakeSample()

            if bincl:
                countIncl = countIncl + 1
                
                # Estimate Mass
                cnt = cnt + 1

        mass = (1.0/float(cnt))*float(countIncl)
        logMass = math.log(mass)

        return likelihoodEst + logMass
        
    def estimateFakeMeanGradient(self):
        bincl, fsamp = self.getFakeSample()
        
        countIncl = int(bincl)
        #print('bincl:', bincl)

        # Estimate Gradient
        varTGRD2 = (0.5)*(fsamp * fsamp.getT())
        varnGRD2 = -fsamp
        
        while (countIncl <= self.iterNum):
            bincl, fsamp = self.getFakeSample()

            if bincl:
                countIncl = countIncl + 1
                
                # Estimate Gradient
                varTGRD2 = varTGRD2 + (0.5)*(fsamp * fsamp.getT())
                varnGRD2 = varnGRD2 - fsamp

        varTGRD2 = (1.0/float(countIncl))*varTGRD2
        varnGRD2 = (1.0/float(countIncl))*varnGRD2
        
        #print('countIncl:', countIncl)
        return varTGRD2, varnGRD2
    
    def estimateRealMeanGradient(self):
        sampl = self.smp.getSample()

        # Estimate Gradient
        varTGRD1 = (0.5)*(sampl * sampl.getT())
        varnGRD1 = - sampl
        
        cnt = 1
        
        for i in range(self.iterNum):
            sampl = self.smp.getSample()

            # Estimate Gradient
            varTGRD1 = varTGRD1 + (0.5)*(sampl * sampl.getT())
            varnGRD1 = varnGRD1 - sampl
            
            cnt = cnt + 1

        varTGRD1 = (1.0/float(cnt))*varTGRD1
        varnGRD1 = (1.0/float(cnt))*varnGRD1

        return varTGRD1, varnGRD1
    
    def projectVarT(self):
        Tw, Tv = np.linalg.eig(self.varT)
        midT = diag(np.maximum(Tw, 0.001))
        rm = Tv * midT * (Tv.getT())
        
        #print(self.varT, midA)
    
        self.varT = rm
    
    def basicSGD(self, iterNum=5000, displayStep = 10):
        lamb = 1
        
        self.mean = np.zeros((2, 1), float)
        self.var = np.matrix('5.0 4.0; 4.0 5.0')
        
        self.updateFake()
        ll = self.estimateLikelihood()
        print('Optimal  Likelihood:', ll)
        
        self.initialEstimations()
        #iterNum = int(self.cons*(((self.d)**2) / ((self.eps)**2)))
        
        fvarT = self.varT
        fvarn = self.varn
        
        print('Step ',0,':')
        print(self.mean)
        print(self.var)
        print(self.varn)
        print(self.varT)
        
        for i in range(1, iterNum + 1):
            if i <= 3000:
                eta = 0.2 #3.0/(float(i)**(0.5))#1.0/(lamb*i)
            elif i <= 8000:
                eta = 0.01# min(eta, 10.0/(float(i) - 495.0))
            else:
                eta = 0.001
            
            # Estimate Gradient
            varTGRD1, varnGRD1 = self.estimateRealMeanGradient()
            varTGRD2, varnGRD2 = self.estimateFakeMeanGradient()
            
            varTGRD = varTGRD1 - varTGRD2
            varnGRD = varnGRD1 - varnGRD2
            
            # Update Fake Variables
            self.varT = self.varT - eta * varTGRD
            self.varn = self.varn - eta * varnGRD
            
            self.projectVarT()
            
            # Update Real Variables
            self.updateReal()
            
            # Update Final Estimation
            fvarT = fvarT + self.varT
            fvarn = fvarn + self.varn
            
            if ((i - 1) % displayStep == 0):
                loglikelihood = self.estimateLikelihood()
                print('Step ',i,':', eta)
                print('mean:', self.mean)
                print('variance:', self.var)
                print('nu:', self.varn)
                print('T:', self.varT)
                print('gradient n:', varnGRD)
                print('gradient T:', varTGRD)
                print('LogLikelihood:', loglikelihood)
                print()
                
                countNonZero = 0
            
        self.varT = (1.0/iterNum)*fvarT
        self.varn = (1.0/iterNum)*fvarn
        
        self.updateReal()

def printSamples(l):
    print('{')
    for i in range(len(l)):
        if (i > 0):
            print(',',)
        print('{',float(l[i][0]),',',float(l[i][1]),'}')
    print('}')
    
mn = np.zeros((2, 1), float)
var = np.matrix('5.0 4.0; 4.0 5.0')
cset = box(2, [(0, 2), (0, 2)])
sset = space(2)
mlt = multivariateGaussian(2, mn, var, cset)

l = []
for i in range(1000):
    l.append(mlt.getSample())

#printSamples(l)

trunc = truncatedLearning(2, cset, mlt, epsilon=0.1)
trunc.basicSGD(iterNum=12000, displayStep=100)
mn, vr = trunc.getReal()

print(mn)
print(vr)
print('LLikelihood:', trunc.estimateLikelihood())