# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 22:49:00 2018

@author: nrgurram
"""


import numpy as np
# Read User-TV show matrix from data set 
userItemMatrix = np.loadtxt("./dataset/user-shows.txt", dtype='i', delimiter=' ');
#print(userItemMatrix)

#TVShows = np.loadtxt("./dataset/sample_shows.txt", dtype='str')
tvShowsFile =  open("./dataset/shows.txt", "r")
tvShows = [];
for line in tvShowsFile:
    tvShows.append(line.strip().strip("\""))
#print(tvShows)

# -------------- USER-USER Filtering ---------------
user20 = userItemMatrix[1:2][0]
# inverse L2 norm of user20
l2user20 = 1/np.sqrt(np.sum(user20)); 
#inverse L2 norm of all users
l2allUsers = np.diag(1/np.sqrt(np.sum(userItemMatrix, axis=1)));

#Get similarity between U(20) and other users by taking cosine similarity
# [1 x 1] * [ 1 x I] * [I x U] * [U x U] -> [1 x U]
user20ToAllUsersSimilarity = (np.dot(np.dot(np.dot(l2user20, user20), np.transpose(userItemMatrix)), l2allUsers));

#User20 item preferences based on his similarity with other users and their preference
# [1 x U] * [U x I] -> [1 x I]
user20ItemPrefAsPerUU = np.dot(user20ToAllUsersSimilarity, userItemMatrix)

# Get TV shows scores in descending order
user20ItemPrefUUSortedIndices = np.argsort(user20ItemPrefAsPerUU)[::-1]

# Get top 10 movies for user 20
count = 0;
user20Top10UU = [];
for index in user20ItemPrefUUSortedIndices:
    if(user20[index] == 0):
        user20Top10UU.append(index);
        count+=1;
    if(count == 10):
        break;


print(user20Top10UU);

# ----------- ITEM-ITEM Filtering---------
#inverse L2 norm for all items
l2allItems = np.diag(1/np.sqrt(np.sum(userItemMatrix, axis=0)))

#item-item similarity
# [I x I] * [I x U] * [U x I] * [I x I]
itemItemSimilarity = np.dot(np.dot(np.dot(l2allItems, np.transpose(userItemMatrix)), userItemMatrix), l2allItems)

#User20 item preferences based on similarity bewteen what he watched actually and their similarity with other items
# [1 x I] * [I x I]
user20ItemPrefAsPerII = np.dot(user20, itemItemSimilarity)

user20ItemPrefIISortedIndices = np.argsort(user20ItemPrefAsPerII)[::-1]

#Get top 10 movies for user 20 based on Item-Item 
count = 0;
user20Top10II = [];
for index in user20ItemPrefIISortedIndices:
    if(user20[index] == 0):
        user20Top10II.append(index);
        count+=1;
    if(count == 10):
        break;
        
print(user20Top10II);

# These values are predicted by different algorithms using MediaLite Library
itemKNNTop10 = [234, 48, 37, 543, 490, 477, 280, 553, 489, 222];
wrmfTop10 = [48, 77, 192, 208, 280, 195, 207, 222, 219, 489];

# Merge all top 10 shows into one list
allTop10 = list(set(user20Top10UU + user20Top10II + itemKNNTop10 + wrmfTop10));
print(allTop10)
allTop10ShowNames = [tvShows[i] for i in allTop10]
print(allTop10ShowNames)

# Calculate kendal Tau distance between all predictions 
import scipy.stats as stats
allPredictions = np.array([user20Top10II, user20Top10UU, itemKNNTop10, wrmfTop10]);
print(allPredictions)
for pred1 in allPredictions:
    KTauPred1 = [];
    for pred2 in allPredictions:
        (tau, p) = stats.kendalltau(pred1, pred2);
        KTauPred1.append(round(tau,3));
    print(KTauPred1);