import numpy as np
from random import randrange
from scipy.spatial import distance
import matplotlib.pyplot as plt


# explicit function to normalize array Frobenius norm
def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    # if norm of the matrix is not 0. Otherwise, leave it be
    if norm != 0:
        matrix = matrix / norm  # normalized matrix
    return matrix


# explicit function to normalize array [Frobenius norm]
def normalize_3d(matrix):
    for i in range(len(matrix)):
        matrix[i] = normalize_2d(matrix[i])
    return matrix


# Split a dataset into k folds
# dataset: 3d numpy array
# return: a np array of 3d numpy array, each 3d array is a subset of the dataset,aka 4d array
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        fold = np.array(fold)
        dataset_split.append(fold)
    return np.array(dataset_split)





# calculate the  distance between two matrices
def compute_distance(m1, m2):
    #Euclidean distance
    #distance = np.linalg.norm(m1 - m2) #same as numpy.sqrt(numpy.sum((A - B)**2))
    #Cosine distance
    dist = distance.cosine(m1.reshape(1, -1), m2.reshape(1, -1))
    return dist


# Locate the most similar neighbors
# train: dataset 3d array
# test_mat: the data point whose neighnors are to be obtained 2d array [still have the labels]
def get_neighbors(train, test_mat, num_neighbors):
    distances = list()
    for train_mat in train:
        dist = compute_distance(test_mat[:, :-1], train_mat[:, :-1])  # exclude the labels
        distances.append((train_mat, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# create a 3d dataset with X: 3d array, y: 1d array, X and y have the same length the returned dataset is a 3d array
# where each 2d element contains each one in X with a vectorized y appended as the last row
def create_dataset(X, y):
    a = X
    b = y
    list = []
    for i in range(len(b)):
        zeros = np.zeros((np.shape(a[0])[0], 1))

        zeros[0] = b[i]

        list.append(np.concatenate((a[i], zeros), axis=1))

    return np.array(list)


# Make a classification prediction with neighbors
# train: dataset 3d array
# test_mat: the data point whose neighnors are to be obtained 2d array
def predict_classification(train, test_mat, num_neighbors):
    neighbors = get_neighbors(train, test_mat, num_neighbors)
    output_values = [mat[0][-1] for mat in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# Calculate accuracy percentage
# actual and predicted are lists
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
# dataset: 3d array, [X,y]
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for i in range(len(folds)):
        fold = folds[i]
        train_set = list(folds)

        train_set.pop(i)
        train_set = np.array(train_set)  # convert the list of 3d array to 4d array
        train_set = train_set.reshape(-1, train_set.shape[-2], train_set.shape[-1])  # flatten 4d array to 3d array

        test_set = np.copy(fold)

        # print(train_set.shape)
        # test_set still have the labels
        predicted = algorithm(train_set, test_set, *args)
        actual = [mat[0][-1] for mat in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# kNN Algorithm
# train: dataset numpy 3d array
# test:  3d array contains the np 2d data points whose neighbors are to be obtained
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for mat in test:
        output = predict_classification(train, mat, num_neighbors)
        predictions.append(output)
    return predictions


dataset = np.load('dataset_smile_challenge.npy', allow_pickle=True).item()

# for training and testing data:
dataset_train = dataset['train']
dataset_test = dataset['test']

# for deep features.
# all the features are of numpy array
deep_features = dataset_train['deep_features']
# conv1d backbone based features for ECG signal.
# >deep_features['ECG_features_C']
# transformer backbone basde features for ECG signal
# >deep_features['ECG_features_T']

# for hand-crafted features.
handcrafted_features = dataset_train['hand_crafted_features']
# handcrafted features for ECG signal
# >print(handcrafted_features['ECG_features'].shape)
# handcrafted features for GSR signal.
# >print(handcrafted_features['GSR_features'].shape)

# for labels(y value).
labels = dataset_train['labels']  # labels.
# convert labels to an array of 0 and 1 to have binary classification.
# label-[0] => 0   and  label-[1, 2,...,6] => 1
labels = labels > 0
labels = labels.astype(int)
print(labels)

# preprocessing for x values
# ignore missing data for now and all the missing data is now represetned as 0
# iterate the 3d arrays on the 1st dimension and perform L2 normanlization

# ECG_dp has a shape of (2070, 60, 320)
# ECG_all (2070,60,328
# X (2070,60,340)
# X contains all featues


#PLOT ECG features HR
# plt.plot(handcrafted_features['ECG_features'][3][:,1])
# plt.ylabel('some numbers')
# plt.show()


ECG_dp = np.concatenate((deep_features['ECG_features_C'], deep_features['ECG_features_T']), -1)
ECG_all = np.concatenate((ECG_dp, handcrafted_features['ECG_features']), -1)
X = np.concatenate((ECG_all, handcrafted_features['GSR_features']), -1)

# L2 normalization
X = normalize_3d(X)

# Test distance function
# >>
# X = X[:50]
# X1 = X[0]
# for m in X:
#     dist = euclidean_distance(X1,m)
#     print(dist)

# Test get neighbours function
# >>
# X = X[:10]
# print(X)
# neighbors = get_neighbors(X, X[0], 3)
# for neighbor in neighbors:
#     print('neighbor')
#     print(neighbor.shape)

# append y label to X
D = create_dataset(X, labels)
# Test predict_classification
# prediction = predict_classification(D,D[-1],3)
# print(prediction)

# Test cross validation
# lst = cross_validation_split(D,2)
# print(lst)

# simulation dataset
# SD = np.array([[[100,100,100,1],[100,100,100,0]],[[99,99,99,1],[99,99,99,0]],[[99,100,99,1],[99,99,99,0]],[[-99,-99,-99,0],[-99,-99,-99,0]],[[-99,-99,-100,0],[-100,-99,-99,0]]])
n_folds = 5
num_neighbors_list = [3, 5, 7, 9, 11]
for i in range(len(num_neighbors_list)):
    scores = evaluate_algorithm(D, k_nearest_neighbors, n_folds, num_neighbors_list[i])
    print('Num of neighbors: %s' % num_neighbors_list[i])
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
    print()
