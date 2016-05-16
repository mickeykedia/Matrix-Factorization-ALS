import numpy as np
from pyspark import SparkContext
import datetime
def parseMovieFile(line):
    if line[0] ==  0:
        return {'movie':int(line[1].rstrip(':'))}
    else:
        arr = line[1].split(',')
        d = {int(arr[0].encode('utf-8')): float(arr[1])}
        return d
    
def merge_dicts(x, y):
    x.update(y)
    return x
    

sc = SparkContext(appName="Saving to correct format")
sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId",'')
sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", '')

# Change the name of the pickle file here
rdd = sc.pickleFile("s3n://netflix-dataset-pickle/rdd2.pickle")

def map_to_item_user_rating(d):
    movie = d['movie']
    del d['movie']
    return (movie, d)

def flat_map_user_ratings(line):
    ret_arr = []
    for k in line[1]:
        ret_arr.append((k, {line[0]:line[1][k]}))
    return ret_arr

item_user_ratings = rdd.map(map_to_item_user_rating)
user_ratings = item_user_ratings.flatMap(flat_map_user_ratings)
user_item_ratings = user_ratings.reduceByKey(merge_dicts)

print "Number of Items:"+str(item_user_ratings.count())
print "Number of Users:" + str(user_item_ratings.count())
print "Number of Ratings:" + str(user_ratings.count())



lambda_ = sc.broadcast(0.1) # Regularization parameter
n_factors = sc.broadcast(3) # nfactors of User matrix and Item matrix
n_iterations = 10

Items = item_user_ratings.map(lambda line: (line[0], 5 * np.random.rand(1, n_factors.value)))
Items_broadcast = sc.broadcast({
  k: v for (k, v) in Items.collect()
})

def Update_User(userTuple):
    '''
    This function calculates (userID, Users[i]) using:
        'Users[i] = inverse(Items*Items^T + I*lambda) * Items * A[i]^T'
    Dot product calculations are done differently than normal to allow for sparsity. Rather 
    than row of left matrix times column of right matrix, sum result of column of left matrix  
    * rows of right matrix (skipping items for which user doesn't have a rating).
    '''
    Itemssquare = np.zeros([n_factors.value,n_factors.value])
    for matrixA_item_Tuple in userTuple[1]:
        itemRow = Items_broadcast.value[matrixA_item_Tuple][0]
        for i in range(n_factors.value):
            for j in range(n_factors.value):
                Itemssquare[i,j] += float(itemRow[i]) * float(itemRow[j])
    leftMatrix = np.linalg.inv(Itemssquare + lambda_.value * np.eye(n_factors.value))
    rightMatrix = np.zeros([1,n_factors.value])
    for matrixA_item_Tuple in userTuple[1]:
        for i in range(n_factors.value):
            rightMatrix[0][i] += Items_broadcast.value[matrixA_item_Tuple][0][i] * userTuple[1][matrixA_item_Tuple]
    newUserRow = np.dot(leftMatrix, rightMatrix.T).T
    return (userTuple[0], newUserRow)

Users = user_item_ratings.map(Update_User)

Users_broadcast = sc.broadcast({
  k: v for (k, v) in Users.collect()
})


def Update_Item(itemTuple):
    '''
    This function calculates (userID, Users[i]) using:
        'Users[i] = inverse(Items*Items^T + I*lambda) * Items * A[i]^T'
    Dot product calculations are done differently than normal to allow for sparsity. Rather 
    than row of left matrix times column of right matrix, sum result of column of left matrix  
    * rows of right matrix (skipping items for which user doesn't have a rating).
    '''
    Userssquare = np.zeros([n_factors.value,n_factors.value])
    for matrixA_user_Tuple in itemTuple[1]:
        userRow = Users_broadcast.value[matrixA_user_Tuple][0]
        for i in range(n_factors.value):
            for j in range(n_factors.value):
                Userssquare[i,j] += float(userRow[i]) * float(userRow[j])
    leftMatrix = np.linalg.inv(Userssquare + lambda_.value * np.eye(n_factors.value))
    rightMatrix = np.zeros([1,n_factors.value])
    for matrixA_user_Tuple in itemTuple[1]:
        for i in range(n_factors.value):
            rightMatrix[0][i] += Users_broadcast.value[matrixA_user_Tuple][0][i] * itemTuple[1][matrixA_user_Tuple]
    newItemRow = np.dot(leftMatrix, rightMatrix.T).T
    return (itemTuple[0], newItemRow)

Items = item_user_ratings.map(Update_Item)

Items_broadcast = sc.broadcast({
  k: v for (k, v) in Items.collect()
})


def getRowSumSquares(userTuple):
    userRow = Users_broadcast.value[userTuple[0]]
    rowSSE = 0.0
    for matrixA_item_Tuple in userTuple[1]:
        predictedRating = 0.0
        for i in range(n_factors.value):
            predictedRating += userRow[0][i] * Items_broadcast.value[matrixA_item_Tuple][0][i]
        SE = (userTuple[1][matrixA_item_Tuple] - predictedRating) ** 2
        rowSSE += SE
    return rowSSE


SSE = user_item_ratings.map(getRowSumSquares).reduce(lambda a, b: a + b) 
Count = item_user_ratings.count()
MSE = SSE / Count
print "MSE:", MSE

for iter in range(n_iterations):
    Users = user_item_ratings.map(Update_User)
    Users_broadcast = sc.broadcast({k: v for (k, v) in Users.collect()})
    Items = item_user_ratings.map(Update_Item)
    Items_broadcast = sc.broadcast({k: v for (k, v) in Items.collect()})
    SSE = user_item_ratings.map(getRowSumSquares).reduce(lambda a, b: a + b)
    MSE = SSE / Count
    print "MSE:", MSE
