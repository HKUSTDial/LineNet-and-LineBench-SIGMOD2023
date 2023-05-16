import os
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn

from sklearn.cluster import AgglomerativeClustering



@torch.no_grad()
def triplet_select_all(config,names,status,name_map,datas,anchors):

    samples=[]

    for i,name in enumerate(anchors):

        anchorIndex = names.index(name)
        anchorData = datas[anchorIndex].reshape(
            1, -1, config.DATA.IMG_H, config.DATA.IMG_W)
        posList=[]
        negList=[]
        idi=name_map[name]
        for k, name_ in enumerate(names):
            idk=name_map[name_]
            if status[idi][idk]==1:
                posList.append(k)
            if status[idi][idk]==-1:
                negList.append(k)

        if len(posList)==0 or len(negList)==0:
            continue

        posIndex=random.choice(posList)
        negIndex=random.choice(negList)

        posData = datas[posIndex].reshape(
            1, -1, config.DATA.IMG_H, config.DATA.IMG_W)
        negData = datas[negIndex].reshape(
            1, -1, config.DATA.IMG_H, config.DATA.IMG_W)
        sample = torch.cat((anchorData, posData, negData))
        samples.append(sample)
    
    return samples


@torch.no_grad()
def triplet_select_hard(config,vec,names,status,name_map,datas,anchors):
    samples = []

    for i, name in enumerate(anchors):

        anchorIndex = names.index(name)
        anchorData = datas[anchorIndex].reshape(
            1, -1, config.DATA.IMG_H, config.DATA.IMG_W)

        maxVal = 0
        minVal = 99999999
        maxIndex = None
        minIndex = None

        posSum=0
        negSum=0
        posCnt=0
        negCnt=0
        id_i=name_map[name]

        for k, name_ in enumerate(names):
            id_k=name_map[name_]

            if status[id_i][id_k]==1:
                dist = torch.pairwise_distance(vec[anchorIndex].reshape(1,-1), vec[k].reshape(1,-1))[0]
                posCnt+=1
                posSum+=dist
                if maxIndex is None or dist > maxVal:
                    maxVal = dist
                    maxIndex = k
                    
            if status[id_i][id_k]==-1:
                dist = torch.pairwise_distance(vec[anchorIndex].reshape(1,-1), vec[k].reshape(1,-1))[0]
                negCnt+=1
                negSum+=dist
                if minIndex is None or dist < minVal:
                    minVal = dist
                    minIndex = k
        
        if maxIndex is None or minIndex is None:

            continue

        posData = datas[maxIndex].reshape(
            1, -1, config.DATA.IMG_H, config.DATA.IMG_W)
        negData = datas[minIndex].reshape(
            1, -1, config.DATA.IMG_H, config.DATA.IMG_W)
        sample = torch.cat((anchorData, posData, negData))
        samples.append(sample)
    
    return samples

@torch.no_grad()
def triplet_select_semi_hard(config,vec,names,status,name_map,datas,anchors):
    samples = []

    for i, name in enumerate(anchors):

        anchorIndex = names.index(name)
        anchorData = datas[anchorIndex].reshape(
            1, -1, config.DATA.IMG_H, config.DATA.IMG_W)

        maxVal = 0
        maxIndex = None

        posSum=0
        negSum=0
        posCnt=0
        negCnt=0
        id_i=name_map[name]
        neg_list=[]

        for k, name_ in enumerate(names):
            id_k=name_map[name_]

            if status[id_i][id_k]==1:
                dist = torch.pairwise_distance(vec[anchorIndex].reshape(1,-1), vec[k].reshape(1,-1))[0]
                posCnt+=1
                posSum+=dist
                if maxIndex is None or dist > maxVal:
                    maxVal = dist
                    maxIndex = k
        
        for k,name_ in enumerate(names):
            id_k=name_map[name_]

            if status[id_i][id_k]==-1:
                dist = torch.pairwise_distance(vec[anchorIndex].reshape(1,-1), vec[k].reshape(1,-1))[0]
                if dist<maxVal+config.TRAIN.MARGIN:
                    neg_list.append(k)
                    negSum+=dist
        
        if maxIndex is None or len(neg_list)==0:
            continue
        
        negIndex=random.choice(neg_list)

        posData = datas[maxIndex].reshape(
            1, -1, config.DATA.IMG_H, config.DATA.IMG_W)
        negData = datas[negIndex].reshape(
            1, -1, config.DATA.IMG_H, config.DATA.IMG_W)
        sample = torch.cat((anchorData, posData, negData))
        samples.append(sample)
    
    return samples
        

@torch.no_grad()
def triplet_select_diversified(config,vec,clusters,series,names,status,name_map,datas,NormDM):

    alpha=config.TRAIN.ALPHA
    beta=config.TRAIN.BETA
    theta=1-alpha-beta

    idx=[name_map[name] for name in names]
    D=torch.norm(vec[:, None]-vec, dim=2, p=2)
    R=NormDM[idx][:,idx]
    aggcls=AgglomerativeClustering(n_clusters=clusters,affinity='precomputed',linkage='complete').fit(D)

    labels=aggcls.labels_
    
    cluster_items=[[] for i in range(clusters)]
    anchors=[]
        
    
    for i,c in enumerate(labels):
        cluster_items[c].append(i)
    
    for candidates in cluster_items:
        anchors.append(random.choice(candidates))
    
    samples=[]
    ids=[name_map[name] for name in names]
    
    for anchorIndex in anchors:
        id_a=ids[anchorIndex]
        cur_pos=[]
        cur_neg=[]
        for kth in range(config.TRAIN.TRIPLET_PER_CLUSTER):
            posMax=None
            posIndex=None
            for i,name in enumerate(names):
                id_i=ids[i]
                if i in cur_pos or status[id_a][id_i]!=1:
                    continue
                div=0
                for pos in cur_pos:
                    div+=R[pos][i]
                weight=-alpha*R[anchorIndex][i]+beta*D[anchorIndex][i]+theta*div
                if posMax is None or posMax<weight:
                    posMax=weight
                    posIndex=i
            if posIndex is None:
                break
            cur_pos.append(posIndex)
            pos_dist=D[anchorIndex][posIndex]

            negMax=None
            negIndex=None
            for i,name in enumerate(names):
                id_i=ids[i]
                if i in cur_pos or status[id_a][id_i]!=-1 or D[anchorIndex][i]>pos_dist+config.TRAIN.MARGIN:
                    continue
                div=0
                for neg in cur_neg:
                    div+=R[neg][i]
                weight=alpha*R[anchorIndex][i]-beta*D[anchorIndex][i]+theta*div
                if negMax is None or negMax<weight:
                    negMax=weight
                    negIndex=i
        
            if negIndex is None:
                continue
            cur_neg.append(negIndex)

            sample=torch.cat((datas[anchorIndex].unsqueeze(0),datas[posIndex].unsqueeze(0),datas[negIndex].unsqueeze(0)))
            samples.append(sample)
    
    return samples