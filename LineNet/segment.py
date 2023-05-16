from email.mime import base
import numpy as np
from sklearn import linear_model
import math
import time

MAX_ERROR = 10
NUM_OF_SEGMENTS = 10
MAXCOST = 1000

def  getMergeCost(data,length):
    x=np.array( range(int(length)) ).reshape(-1,1)
    y=data.reshape(-1,1)
    
    regr=linear_model.LinearRegression(n_jobs=-1)
    regr.fit(x.reshape(-1,1),y.reshape(-1,1))
    a,b=regr.coef_,regr.intercept_
    y_pred=a*x+b
    cost=0.5*np.sum((y-y_pred)**2)
    cost=0
    return cost


def linearNormalize(inputTrend):
    max_val=np.max(inputTrend)
    min_val=np.min(inputTrend)
    if max_val==min_val:
        return inputTrend
    inputTrend=(inputTrend-min_val)/(max_val-min_val)
    return inputTrend

'''startX endX costToMergeNext startY endY length slope intercept'''

def getBottomUp(inputTrend):
    sList=[]
    for i in range(len(inputTrend)-1):
        temp=np.zeros(8,dtype=np.float64)
        temp[0]=i
        temp[1]=i+1
        if i!=len(inputTrend)-2:
            temp[2]=getMergeCost(inputTrend[i:i+3],3)
        else: 
            temp[2]=MAXCOST
        temp[3]=inputTrend[i]
        temp[4]=inputTrend[i+1]
        temp[6]=inputTrend[i+1]-inputTrend[i]
        temp[7]=inputTrend[i]-temp[6]*i
        temp[5]=math.sqrt(1+(inputTrend[i]-inputTrend[i+1])**2 )
        sList.append(temp)

    while True:
        pos=-1
        pos_val=MAXCOST
        for i,seg in enumerate(sList):
            if pos==-1 or pos_val>seg[2]:
                pos=i 
                pos_val=seg[2]

        if len(sList)<=NUM_OF_SEGMENTS or pos_val>=MAX_ERROR:
            break
        
        sList[pos][1]=sList[pos+1][1]
        sList[pos][4]=sList[pos+1][4]
        sList[pos][5]=math.sqrt((sList[pos][1]-sList[pos][0])**2+(inputTrend[int(sList[pos][1])]-inputTrend[int(sList[pos][0])])**2) 
        X=np.array(range(int(sList[pos][0]),int(sList[pos][1]))).reshape(-1,1)
        Y=inputTrend[int(sList[pos][0]):int(sList[pos][1])].reshape(-1,1)
        regr=linear_model.LinearRegression(n_jobs=-1)
        regr.fit(X.reshape(-1,1),Y.reshape(-1,1))
        a,b=regr.coef_,regr.intercept_
        sList[pos][6]=a
        sList[pos][7]=b
        sList.pop(pos+1)

        if pos==len(sList)-1:
            sList[pos][2]=MAXCOST
        else:
            sList[pos][2]=getMergeCost(inputTrend[int(sList[pos][0]):int(sList[pos+1][1])+1],sList[pos+1][1]-sList[pos][0]+1)* \
                (sList[pos+1][1]-sList[pos][0])
        
        if pos!=0:
            sList[pos-1][2]=getMergeCost(inputTrend[int(sList[pos-1][0]):int(sList[pos][1])+1],sList[pos][1]-sList[pos-1][0]+1)* \
                (sList[pos][1]-sList[pos-1][0])

    return sList

def getSimilarityRate(s1,s2,inputTrend1,inputTrend2,useTrend=True):
    y11,y12=s1[3],s1[4]
    y21,y22=s2[3],s2[4]
    mid1=(y11+y12)/2
    mid2=(y21+y22)/2
    distance=math.sqrt(2*(mid1-mid2)**2)
    width1=s1[1]-s1[0]
    width2=s2[1]-s2[0]

    for i in range(6):
        tempx1=s1[0]+width1*i/6
        tempx2=s2[0]+width2*i/6
        tempy1=s1[6]*tempx1+s1[7]
        tempy2=s2[6]*tempx2+s2[7]
        distance+=(tempy1-tempy2)**2
    
    distance=math.sqrt(distance/6)
    return 1-distance



def getMatchingSimilarity(l1,l2,type_,totalLength1,totalLength2,inputTrend1,inputTrend2):
    if type_==0:
        s1=l1[0]
        s2=l2[0]
        similarityRate = getSimilarityRate(s1,s2,inputTrend1,inputTrend2)
        weight=(s1[5]+s2[5])/(totalLength1+totalLength2)
        return similarityRate*weight

    if type_==1:
        s=l2[0]
        l=l1 
    else: 
        s=l1[0]
        l=l2
    
    listLength = 0
    for item in l: 
        listLength+=item[5]
    listWidth=l[-1][1]-l[0][0]
    currentStart = s[0]
    sWidth=s[1]-s[0]
    averageSimilarity=0

    for i in range(len(l)):
        tempS=np.zeros(8)
        tempS[0]=currentStart
        tempS[3]=tempS[0]*s[6]+s[7]
        tempS[1]=tempS[0]+l[i][5]/listLength*sWidth
        tempS[4]=tempS[1]*s[6]+s[7]
        tempS[6]=s[6]
        tempS[7]=s[7]
        averageSimilarity+=getSimilarityRate(s,tempS,inputTrend1,inputTrend2,False)
        currentStart+=l[i][5]/listLength*sWidth
    
    weight = (listLength + s[5]) / (totalLength1 + totalLength2)
    return averageSimilarity/len(l)*weight



def getMaxAtIJ(similarities,i,j,baseLevel):
    #rt = similarities[i][j][baseLevel - j]
    rt=np.max(similarities[i][j][baseLevel-j+1:baseLevel+i])
    '''for k in range(baseLevel-j+1,baseLevel+i):
        if similarities[i][j][k] > rt:
                rt = similarities[i][j][k]'''
    return rt


def getDTWSimilarityV2(t1,t2,inputTrend1,inputTrend2):
    matchingSimilarities=np.zeros((len(t1),len(t2),len(t1)+len(t2)-1),dtype=np.float64)
    accumulatedSimilarity=np.zeros((len(t1),len(t2),len(t1)+len(t2)-1),dtype=np.float64)
    accumulatedMax=np.zeros((len(t1),len(t2)),dtype=np.float64)

    totalLength1 = 0
    for item in t1:
        totalLength1+=item[5]
    
    totalLength2 = 0
    for item in t2:
        totalLength2+=item[5]

    baseLevel=len(t2)-1
    for i in range(len(t1)):
        for j in range(len(t2)):

            matchingSimilarities[i][j][baseLevel]=getMatchingSimilarity(t1[i:i+1],t2[j:j+1],0,totalLength1,totalLength2,inputTrend1,inputTrend2)
            
            for k in range(i):
                matchingSimilarities[i][j][baseLevel + k + 1]=getMatchingSimilarity(t1[i-k:i+1],t2[j:j+1],1,totalLength1,totalLength2,inputTrend1,inputTrend2)
            for k in range(j):
                matchingSimilarities[i][j][baseLevel - k - 1]=getMatchingSimilarity(t1[i:i+1],t2[j-k:j+1],2,totalLength1,totalLength2,inputTrend1,inputTrend2)
            
    accumulatedSimilarity[0][0][baseLevel] = matchingSimilarities[0][0][baseLevel]
    accumulatedMax[0][0] = accumulatedSimilarity[0][0][baseLevel]

    for i in range(len(t1)):
        for k in range(i+1):
            accumulatedSimilarity[i][0][baseLevel + k]=matchingSimilarities[i][0][baseLevel + i]
        accumulatedMax[i][0]=matchingSimilarities[i][0][baseLevel + i]
    
    for i in range(len(t2)):
        for k in range(i+1):
            accumulatedSimilarity[0][i][baseLevel - k]=matchingSimilarities[0][i][baseLevel - i]
        accumulatedMax[0][i]=matchingSimilarities[0][i][baseLevel - i]

    for i in range(1,len(t1)):
        for j in range(1,len(t2)):
            accumulatedSimilarity[i][j][baseLevel] = accumulatedMax[i-1][j-1] + matchingSimilarities[i][j][baseLevel]

            for k in range(1,i):
                accumulatedSimilarity[i][j][baseLevel + k] = accumulatedMax[i-k-1][j-1] + matchingSimilarities[i][j][baseLevel + k]
            
            for k in range(1,j):
                accumulatedSimilarity[i][j][baseLevel - k] = accumulatedMax[i-1][j-k-1] + matchingSimilarities[i][j][baseLevel - k]
            
            accumulatedMax[i][j] = getMaxAtIJ(accumulatedSimilarity,i,j,baseLevel)

    return accumulatedMax[len(t1)-1][len(t2)-1]
    

def SegmentationDistance(src,dst):
    
    t1=getBottomUp(linearNormalize(src))
    t2=getBottomUp(linearNormalize(dst))
    sim=getDTWSimilarityV2(t1,t2,src,dst)

    return 1-sim
