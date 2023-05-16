import math
import numpy as np
from  sklearn.metrics.pairwise import euclidean_distances

def preprocessing(array):
    if len(array)>5:
        result=[]
        for i in range(len(array)-2):
            result.append(array[i] / 4 + array[i+1] / 2 + array[i+2] / 4)
        return result 
    return array

def NormVDist(array):
    step=(array[-1]-array[0])/(len(array)-1)

    dist=[]
    current=array[0]
    for item in array:
        dist.append(abs(item-current))
        current+=step
    
    return array

def getVIPs(array):
    threshold = 0.15
    VIPlist=[ [0,0,0],[len(array)-1,0,1] ]
    waitinglist=[]

    if len(array)>2:
        Dist=NormVDist(array)
        possVIPindex=1
        possVIPdist=Dist[1]
        for i in range(2,len(array)-1):
            if Dist[i] > possVIPdist:
                possVIPdist=Dist[i]
                possVIPindex=i
        dewlIndex=-1
        if possVIPindex>threshold:
            waitinglist.append([possVIPindex,possVIPdist,0,len(array)-1])
            dewlIndex=0
        
        while dewlIndex>=0:
            possVIP=waitinglist[dewlIndex]
            newVip=[possVIP[0],possVIP[1],len(VIPlist)]
            VIPlist.append(newVip)
            waitinglist.pop(dewlIndex)
            startIndex=possVIP[2]
            endIndex=possVIP[3]
            middleIndex=possVIP[0]

            if middleIndex>(startIndex+1):
                Dist=NormVDist(array[startIndex:middleIndex+1])
                possVIPindex = startIndex+1
                possVIPdist = Dist[1]
                for i in range(startIndex+2,middleIndex):
                    if Dist[i-startIndex] > possVIPdist:
                        possVIPdist=Dist[i-startIndex]
                        possVIPindex=i
                if possVIPdist>threshold:
                    possVIP=[possVIPindex, possVIPdist, startIndex, middleIndex]
                    waitinglist.append(possVIP)
            
            if endIndex>(middleIndex+1):
                Dist=NormVDist(array[middleIndex:endIndex+1])
                possVIPindex = middleIndex+1
                possVIPdist = Dist[1]
                for i in range(middleIndex+2,endIndex):
                    if Dist[i-middleIndex] > possVIPdist:
                        possVIPdist=Dist[i-middleIndex]
                        possVIPindex=i
                if possVIPdist>threshold:
                    possVIP=[possVIPindex, possVIPdist, middleIndex, endIndex]
                    waitinglist.append(possVIP)
            
            dewlIndex=-1
            tmpdist=-1
            for i,item in enumerate(waitinglist):
                if item[1]>tmpdist:
                    tmpdist=item[1]
                    dewlIndex=i
    
    sorted(VIPlist,key=lambda item:item[0])
    return VIPlist

def getIndicators(ts,VIPlist):
    dimension = 8
    Xrange=len(ts)-1
    indicatorArray=np.zeros((len(VIPlist),dimension))
    nearbyShapeInterval=[-2,-1,1,2]
    nearbyPatternInterval=[-1,1]

    for i,item in enumerate(VIPlist):
        if Xrange>0:
            indicatorArray[i][0]=item[0]/Xrange
        
        indicatorArray[i][1]=ts[item[0]]

        for j in range(len(nearbyShapeInterval)):
            index = item[0]+nearbyShapeInterval[j]
            if index>=0 and index<len(ts):
                indicatorArray[i][2+j]=(ts[index]-ts[item[0]])*Xrange
        
        for j in range(len(nearbyPatternInterval)):
            VIPindex=i+nearbyPatternInterval[j]
            if VIPindex>=0 and VIPindex<len(VIPlist):
                indicatorArray[i][2+len(nearbyShapeInterval)+j]= \
                    (ts[item[0]]-ts[VIPlist[VIPindex][0]])/(item[0]-VIPlist[VIPindex][0])*Xrange
        
    
    return indicatorArray

def eucDist(ts1,ts2):
    sum_=0
    for i in range(len(ts1)):
        sum_+=(ts1[i]-ts2[i])**2
    
    return math.sqrt(sum_)

def DTWDist(IndicatorsI,IndicatorsJ):
    costMatrix=np.zeros((len(IndicatorsI),len(IndicatorsJ)))

    costMatrix[0][0]=eucDist(IndicatorsI[0],IndicatorsJ[0])
    for j in range(1,len(IndicatorsJ)):
        costMatrix[0][j] = costMatrix[0][j-1] + eucDist(IndicatorsI[0], IndicatorsJ[j])

    for i in range(1,len(IndicatorsI)):
        costMatrix[i][0] = costMatrix[i-1][0] + eucDist(IndicatorsI[i], IndicatorsJ[0])

        for j in range(1,len(IndicatorsJ)):
            costMatrix[i][j]=min(costMatrix[i-1][j-1],min(costMatrix[i][j-1], costMatrix[i-1][j]))#+euclidean_distances(IndicatorsI[i].reshape(1,-1),IndicatorsJ[j].reshape(1,-1))[0][0]
    
    return costMatrix[-1][-1]

def MVIPDistance(src,tar):
    src=preprocessing(src)
    tar=preprocessing(tar)

    srcVIPlist = getVIPs(src)
    tarVIPlist = getVIPs(tar)

    srcIndicators = getIndicators(src, srcVIPlist)
    tarIndicators = getIndicators(tar, tarVIPlist)

    return DTWDist(srcIndicators,tarIndicators)

