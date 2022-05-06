import numpy as np 
import collections
import time
import datetime
import mlxtend
from collections import defaultdict
from sklearn.naive_bayes import CategoricalNB
from mlxtend.classifier import EnsembleVoteClassifier
class SISA:
    def __init__(self, X_train, y_train, shards=1, slices=None, seed = None):
        self.X_train = X_train
        self.y_train = y_train
        self.N, self.D = X_train.shape
        self.shards = shards
        self.seed = seed
        self.estimators = []
        self.slices = slices
        self.dict_train_time = defaultdict(dict)
        for i in range(self.shards):
            if self.slices != 0:
                for j in range(self.slices):
                    self.dict_train_time[i][j] = 0
            else:
                self.dict_train_time[i] = 0
        self.train_time_per_shard = defaultdict(dict)
        for i in range(self.shards):
            self.train_time_per_shard[i] = 0
        self.affected_shards = []

    def gen_random_seq(self, size): # sequence to delete random rows in our data
        delete_rows = np.random.choice(range(self.N), size=(size, 1), replace=False) # range start, stop + 1; # Max number of rows we can delete
        return delete_rows

    def fit(self, unlearn_requests=None):
        np.random.seed(self.seed)
        indexer = np.arange(self.N) # interval for index based on size of X_train data
        y_indexer = np.arange(self.N)
        x_shards = np.array_split(indexer, self.shards)
        y_shards = np.array_split(y_indexer, self.shards)
        if self.shards != 0:
            # Initial fit without unlearning requests
            flag = not np.any(unlearn_requests)
            if flag:
                for i, (x_shard, y_shard) in enumerate(zip(x_shards, y_shards)):
                    naive_b_wlearner = CategoricalNB(min_categories=256)
                    if self.slices == None:
                        start = datetime.datetime.now()
                        X_train_b = self.X_train[x_shard]
                        y_train_b = self.y_train[y_shard]
                        naive_b_wlearner.fit(X_train_b, y_train_b, classes=np.unique(y_train_b))
                        end = datetime.datetime.now()
                        diff = (end -start)
                        execution_time = diff.total_seconds() * 1000
                        self.dict_train_time[i] = execution_time
                        self.estimators.append(naive_b_wlearner)
                    else:
                        slices_in_x_shard = np.array_split(indexer, self.slices) # slight innaccuracy because array_split will in some cases remove a tiny bit of data on the split; negligible
                        slices_in_y_shard = np.array_split(y_indexer, self.slices)
                        for j, (ind_slice_x, ind_slice_y) in enumerate(zip(slices_in_x_shard, slices_in_y_shard)):
                            start = datetime.datetime.now()
                            X_train_slice = self.X_train[ind_slice_x]
                            y_train_slice = self.y_train[ind_slice_y]
                            naive_b_wlearner.partial_fit(X_train_slice, y_train_slice, classes=np.unique(y_train_slice))
                            end = datetime.datetime.now()
                            diff = (end -start)
                            execution_time = diff.total_seconds() * 1000
                            self.dict_train_time[i][j] = execution_time
                        self.estimators.append(naive_b_wlearner)

                # Case for retraining only the affected shards
            else: 
                for x_shard, y_shard in zip(x_shards, y_shards):
                    bool_val = False
                    for item in x_shard:
                        for unlearn_request in unlearn_requests:
                            if unlearn_request[0] == item:
                                bool_val = True
                    if bool_val:
                        self.affected_shards.append(x_shard)
                affected_shards_x = np.array(self.affected_shards)

                for i, (x_shard,y_shard) in enumerate(zip(x_shards, y_shards)):
                    flag = False
                    for aff_x_shard in affected_shards_x:
                        if flag == True:
                            continue
                        elif np.array_equal(x_shard,aff_x_shard) != True:
                            for j in range(self.slices):
                                self.dict_train_time[i][j] = 0
                            continue
                        else:
                            flag = True
                            naive_b_wlearner = CategoricalNB(min_categories=256)
                            if self.slices == None:
                                start = datetime.datetime.now()
                                X_train_copy = np.copy(self.X_train)
                                y_train_copy = np.copy(self.y_train)
                                X_train_b_1 = X_train_copy[x_shard]
                                y_train_b_1 = y_train_copy[y_shard]
                                X_train_b = np.delete(X_train_b_1, x_shard, axis=0)
                                if X_train_b.size == 0:
                                    y_train_b = np.delete(y_train_b_1, y_shard)
                                    self.estimators.pop(i)
                                    continue
                                y_train_b = np.delete(y_train_b_1, y_shard, axis=0)
                                naive_b_wlearner.fit(X_train_b, y_train_b)
                                end = datetime.datetime.now()
                                diff = (end -start)
                                execution_time = diff.total_seconds() * 1000
                                self.dict_train_time[i] = execution_time
                                self.nb_learners[i] = naive_b_wlearner
                            else:
                                slices_in_x_shard = np.array_split(x_shard, self.slices) # slight innaccuracy because array_split will in some cases remove a tiny bit of data on the split; negligible
                                slices_in_y_shard = np.array_split(y_shard, self.slices)
                                # Currently in one slice at each shard 
                                for unlearn_request in unlearn_requests:
                                    for j, (x_ind_slice, y_ind_slice) in enumerate(zip(slices_in_x_shard, slices_in_y_shard)):
                                        start = datetime.datetime.now()
                                        if unlearn_request in x_ind_slice: # just omit the slice from the training of our model!!!!!
                                            self.dict_train_time[i][j] = 0
                                            continue          
                                        else: # feeding slices that are not in the unlearn request to our model
                                            X_train_slice = self.X_train[x_ind_slice]
                                            y_train_slice = self.y_train[y_ind_slice]
                                            naive_b_wlearner.partial_fit(X_train_slice, y_train_slice, classes=np.unique(y_train_slice))
                                        end = datetime.datetime.now()
                                        diff = (end -start)
                                        execution_time = diff.total_seconds() * 1000
                                        self.dict_train_time[i][j] = execution_time
                                self.estimators[i] = naive_b_wlearner

    def predict(self, X_test, X_train_copy, y_train_copy): # Sort of the aggregation method
        #y_test_hats = np.empty((len(self.nb_learners), len(X_test)))
        maj_vote = EnsembleVoteClassifier(clfs=self.estimators, voting='hard', fit_base_estimators=False) # Majority vote
        maj_vote.fit(X_train_copy, y_train_copy)
        for i in range(self.shards):
            sum = 0 
            for j in range(self.slices): # Sums up all the time it took to train each slice for our weak learner per shard
                sum+=self.dict_train_time[i][j]
            self.train_time_per_shard[i] = sum 
        for i in range(self.shards):
            print ('Execution time for training Shard %d : %d ms' %(i, self.train_time_per_shard[i]))
        self.reset_time()
        self.affected_shards = []
        return maj_vote.predict(X_test)
    
    def reset_time(self):
        for i in range(self.shards):
            self.train_time_per_shard[i] = 0

    def reset_learners(self):
        self.estimators = []
        self.fit()
        # Reset after intial fit to now calculate retrain time with starting point at 0 for all shards
        for i in range(self.shards):
            if self.slices != 0:
                for j in range(self.slices):
                    self.dict_train_time[i][j] = 0
            else:
                self.dict_train_time[i] = 0
