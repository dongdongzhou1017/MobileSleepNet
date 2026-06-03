import numpy as np

class kFoldGenerator():
    '''
    Data Generator
    '''
    k = -1      # the fold number
    x_list = [] # x list with length=k
    y_list = [] # x list with length=k

    # Initializate
    def __init__(self, x, y, k, n): # k
        if len(x) != len(y):
            assert False, 'Data generator: Length of x or y is not equal to k.'
        self.k = k # self.k = len(x) 100
        self.n = n
        self.x_list = x
        self.y_list = y

    # Get i-th fold
    def getFold(self, i): # i: 0~24  training set: 96 subjects  test set: 4 subjects
        fold_len = self.n //self.k # 4 4 subjects per fold, 24 folds for training, 1 fold for validation
        isFirst = True
        isValFirst = True
        train_data = None
        train_targets = None
        val_data = None
        val_targets = None
        
        for p in range(self.k): # 0~24 25
            if p != i:
                for j in range(fold_len):
                    subject_idx = p * fold_len + j  # 修正索引计算
                    if subject_idx >= len(self.x_list):  # 添加边界检查
                        break
                    if isFirst:
                        train_data = self.x_list[subject_idx] # 1. sub1,2
                        train_targets = self.y_list[subject_idx]
                        isFirst = False
                    else:
                        train_data = np.concatenate((train_data, self.x_list[subject_idx]))
                        train_targets = np.concatenate((train_targets, self.y_list[subject_idx]))
            else:
                for j in range(fold_len):
                    subject_idx = p * fold_len + j  # 修正索引计算
                    if subject_idx >= len(self.x_list):  # 添加边界检查
                        break
                    if isValFirst:
                        val_data = self.x_list[subject_idx] # 1. sub0
                        val_targets = self.y_list[subject_idx] 
                        isValFirst = False
                    else:
                        val_data = np.concatenate((val_data, self.x_list[subject_idx]))
                        val_targets = np.concatenate((val_targets, self.y_list[subject_idx]))
        # return train_data[:,1:3,:], train_targets, val_data[:,1:3,:], val_targets
        # 确保返回数据的类型正确
        if train_data is not None:
            train_data = train_data.astype(np.float32)
        if train_targets is not None:
            train_targets = train_targets.astype(np.float32)
        if val_data is not None:
            val_data = val_data.astype(np.float32)
        if val_targets is not None:
            val_targets = val_targets.astype(np.float32)
            
        return train_data, train_targets, val_data, val_targets



