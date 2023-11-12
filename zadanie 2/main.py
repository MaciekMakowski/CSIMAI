import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, balanced_accuracy_score



df = pd.read_csv('australian.dat', delimiter=" ")

df2 =df.copy()

size = len(df)

data = []

diff = []



for column in df:

    diff.append([column])

    data.append([column])



def randomMissing():

    i = np.random.randint(size-(size * 0.1))

    return i



def createMissing(data, i):

    for index in range(round(size * 0.1)):

        data[index + i] = np.nan







def findMinMax(data):

    return [min(data), max(data)]





def countDiff(data):

    return [len(np.unique(data)), np.unique(data)]



def deviation(data):

    return(np.std(data))



def classesDeviation():

    for uniClass in df["CLASS"].unique():

        clStd = []

        cl = df.where(df["CLASS"] == uniClass)

        cl.dropna(inplace=True)

        for element in cl:

            clStd.append(deviation(cl[element]))

        data.append(clStd)





def betterPrint(data):

    for item in data:

        if(isinstance(item[0], str)):

            print("\nTitle")

            print(item[0])

            print("Min/Max")

            print(item[1])

            print("Number of unique")

            print(item[2][0])

            print("Unique values")

            print(item[2][1])

            print("Deviation")

            print(item[3])

        elif(isinstance(item[0],float)):

            print(item)





i = randomMissing()



for item in data:

    item.append(findMinMax(df[item[0]]))

    item.append(countDiff(df[item[0]]))

    item.append(deviation(df[item[0]]))

    createMissing(df2[item[0]], i)



classesDeviation()

betterPrint(data)



