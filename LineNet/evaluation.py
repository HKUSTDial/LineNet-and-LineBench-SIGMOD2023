
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, cosine_distances
from scipy.stats import wasserstein_distance
import os
import pandas as pd
from dtaidistance import dtw
from segment import SegmentationDistance
from mvip import MVIPDistance
import time

class ComputeSimilarityGroundTruth(object):
    def __init__(self, data_folder, distance_metric):
        self.data_folder = data_folder 
        self.distance_metric = distance_metric 
        self.ImgSeriesData = []
        self.ImgSeriesName = []

    def normalize_simple(self, data: np.ndarray):
        data -= np.min(data)
        if np.max(data)==0:
            return data
        return data / np.max(data)

    def top_k_indices(self, array: np.ndarray, k: int) -> tuple:
        indices = np.argpartition(array, -k)[-k:]
        indices = indices[np.argsort(-array[indices])]
        top_k_indices = np.unravel_index(indices, array.shape)
        return top_k_indices

    def bottom_k_indices(self, array: np.ndarray, k: int) -> tuple:
        indices = np.argpartition(array, k)[:k]
        indices = indices[np.argsort(array[indices])]
        bottom_k_indices = np.unravel_index(indices, array.shape)
        return bottom_k_indices

    def get_simImg_by_imgID(self, DistMatrix, item_name: str, topk: int):
        item_id = np.where(self.ImgSeriesName == item_name)[0][0]
        topk_inds = self.bottom_k_indices(DistMatrix[item_id], topk + 1)
        return list(self.ImgSeriesName[topk_inds][1:])  # skip the 0-index of itself.
    
    def get_id_from_name(self,item_name:str):
        item_id = np.where(self.ImgSeriesName == item_name)[0][0]
        return item_id

    def build_dist_matrix(self,trains=[],load=False,path='TrendLineBenchmark/saved/',interpolation=False):
        csv_folder = os.listdir(self.data_folder)

        self.ImgSeriesName=[]
        self.ImgSeriesData=[]

        if interpolation:
            path=path+'interpolation/'

        if load==True:
            if self.distance_metric == 'dtw':
                self.ImgSeriesName=np.load(path+'names_dtai.npy')
                DistMatrix=np.load(path+'DistMatrix_dtai.npy')
                return DistMatrix
            elif self.distance_metric == 'emd':
                self.ImgSeriesName=np.load(path+'names_emd.npy')
                DistMatrix=np.load(path+'DistMatrix_emd.npy')
                return DistMatrix
            elif self.distance_metric == "segment":
                self.ImgSeriesName=np.load(path+'names_seg.npy')
                DistMatrix=np.load(path+'DistMatrix_seg.npy')
                return DistMatrix
            elif self.distance_metric == "mvip":
                self.ImgSeriesName=np.load(path+'names_mvip.npy')
                DistMatrix=np.load(path+'DistMatrix_mvip.npy')
                return DistMatrix
            

        for csv_file in csv_folder:
            if '.csv' in csv_file and ( csv_file.replace('csv','png') in trains or trains==[]) :
                df = pd.read_csv(self.data_folder + csv_file)
                X = self.normalize_simple(df[df.columns.values[1]])  # read & normalize the data
                X = np.nan_to_num(X)

                if interpolation:
                    axis=np.arange(X.shape[0])
                    axis=axis*(1024//X.shape[0])
                    X=np.interp(np.arange(1024),axis,X)

                self.ImgSeriesData.append(np.array(X,dtype=np.float64))
                self.ImgSeriesName.append(csv_file)

        self.ImgSeriesName = np.array(self.ImgSeriesName)

        if self.distance_metric == 'euclidean_distances':
            DistMatrix = euclidean_distances(self.ImgSeriesData, self.ImgSeriesData)
            return DistMatrix
        elif self.distance_metric == 'cosine_distances':
            DistMatrix = cosine_distances(self.ImgSeriesData, self.ImgSeriesData)
            return DistMatrix
        elif self.distance_metric == 'dtw':
            

            tmp=[]
            for i,item in enumerate(self.ImgSeriesData):
                tmp.append(np.array(item,dtype=np.float64))
            self.ImgSeriesData=tmp
            DistMatrix=dtw.distance_matrix_fast(self.ImgSeriesData,parallel=True)
            np.save(path+"names_dtai.npy",self.ImgSeriesName)
            np.save(path+"DistMatrix_dtai.npy",DistMatrix)
            return DistMatrix
        elif self.distance_metric == 'emd':

            DistMatrix = np.zeros((len(self.ImgSeriesData),len(self.ImgSeriesData)))
            for i,itemI in enumerate(self.ImgSeriesData):
                for j,itemJ in enumerate(self.ImgSeriesData):
                    if i>j:
                        DistMatrix[i][j]=DistMatrix[j][i]
                    elif i<j:
                        DistMatrix[i][j]=wasserstein_distance(itemI,itemJ)

            np.save(path+"names_emd.npy",self.ImgSeriesName)
            np.save(path+"DistMatrix_emd.npy",DistMatrix)
            return DistMatrix
        elif self.distance_metric == "segment":

            DistMatrix = np.zeros((len(self.ImgSeriesData),len(self.ImgSeriesData)))
            for i,itemI in enumerate(self.ImgSeriesData):
                for j,itemJ in enumerate(self.ImgSeriesData):
                    if i>j:
                        DistMatrix[i][j]=DistMatrix[j][i]
                    elif i<j:
                        begin=time.time()
                        DistMatrix[i][j]=SegmentationDistance(itemI,itemJ)

            np.save(path+"names_seg.npy",self.ImgSeriesName)
            np.save(path+"DistMatrix_seg.npy",DistMatrix)
            return DistMatrix
        elif self.distance_metric == "mvip":

            DistMatrix = np.zeros((len(self.ImgSeriesData),len(self.ImgSeriesData)))
            for i,itemI in enumerate(self.ImgSeriesData):
                for j,itemJ in enumerate(self.ImgSeriesData):
                    if i>j:
                        DistMatrix[i][j]=DistMatrix[j][i]
                    elif i<j:
                        begin=time.time()
                        DistMatrix[i][j]=MVIPDistance(itemI,itemJ)

            np.save(path+"names_mvip.npy",self.ImgSeriesName)
            np.save(path+"DistMatrix_mvip.npy",DistMatrix)
            return DistMatrix
        else:
            assert("please specify the right distance function.")
    
    def load_data(self):
        for csv_file in self.ImgSeriesName:
            df = pd.read_csv(self.data_folder + csv_file)
            X = self.normalize_simple(df[df.columns.values[1]])  # read & normalize the data
            X = np.nan_to_num(X)

            self.ImgSeriesData.append(np.array(X,dtype=np.float64))



class MeanAvgPrecisionK(object):
    def __init__(self):
        pass

    def apk(self, actual, predicted, k=10):
        """
        Computes the average precision at k.
        This function computes the average prescision at k between two lists of
        items.
        Parameters
        ----------
        actual : list
                 A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The average precision at k over the input lists
        """
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)

    def mapk(self, actual, predicted, k=10):
        """
        Computes the mean average precision at k.
        This function computes the mean average prescision at k between two lists
        of lists of items.
        Parameters
        ----------
        actual : list
                 A list of lists of elements that are to be predicted
                 (order doesn't matter in the lists)
        predicted : list
                    A list of lists of predicted elements
                    (order matters in the lists)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The mean average precision at k over the input lists
        """
        return np.mean([self.apk(a, p, k) for a, p in zip(actual, predicted)])

def analysis(vec,origin,DMTest,ComputeSim,knn,records,topk):
    totalPrecision=0
    totalMAPK=0
    totalHR10=0
    totalHR50=0
    totalR10_50=0
    totalR10_100=0
    MAPK=MeanAvgPrecisionK()

    for i,item in enumerate(vec):

        topk_list=ComputeSim.get_simImg_by_imgID(DMTest,item_name=records[i][:-3]+'csv',topk=topk)
        predicted=knn.kneighbors([item],return_distance=False)

        predicted_ids=[]
        related=0
        hr10=0
        hr50=0
        r10_50=0
        r10_100=0
        
        for j,num in enumerate(predicted[0]):
            if j==0:
                continue
            predicted_ids.append(records[num][:-3]+'csv')
            if records[num][:-3]+'csv' in topk_list:
                related+=1
        
        for j,item in enumerate(predicted_ids[:10]):
            if item in topk_list:
                hr10+=1
        
        for j,item in enumerate(predicted_ids[:50]):
            if item in topk_list:
                hr50+=1
        
        for j,item in enumerate(topk_list[:10]):
            if item in predicted_ids[:50]:
                r10_50+=1
            if item in predicted_ids:
                r10_100+=1
            
        totalPrecision+=related/topk
        totalMAPK+=MAPK.mapk(actual=[topk_list[:topk]],predicted=[predicted_ids],k=topk)
        totalHR10+=hr10/10
        totalHR50+=hr50/50
        totalR10_50+=r10_50/10
        totalR10_100+=r10_100/10


    print(totalMAPK/len(vec),totalPrecision/len(vec))
    print('hr10:',totalHR10/len(vec),'hr50',totalHR50/len(vec),'r10@50',totalR10_50/len(vec),'r10@100',totalR10_100/len(vec))

    return totalMAPK/len(vec),totalPrecision/len(vec),totalHR10/len(vec),totalHR50/len(vec),totalR10_50/len(vec),totalR10_100/len(vec)