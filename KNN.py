import csv
import numpy
import scipy
from sklearn import neighbors


def loadData():
    with open('test1.csv', 'rt', encoding='utf-8') as csvfile:
        lines = csv.reader(csvfile)
        trainSet = list()
        labelSet = []
        for row in lines:
            x = list(row)[3:52]
            y = list(row)[56]
            trainSet.append(x)
            labelSet.append(y)

    return {'data': trainSet, 'target': labelSet}



knn = neighbors.KNeighborsClassifier()

data = loadData()
knn.fit(data['data'], data['target'])
with open('pre.csv', 'rt', encoding='utf-8') as csvfile:
    lines = csv.reader(csvfile)
    ac = 0
    total = 0
    for row in lines:
        x = list(row)[3:52]
        prelabel = numpy.array(x).reshape(1, -1)
        presresult = knn.predict(prelabel)
        if str(row[56]) == str(presresult[0]):
            ac += 1
        total += 1
        print('预测值：', presresult[0])
        print(' 真实值：', row[56])
    print('total=',total,'ac=',ac)
    print('准确率：',str(ac/float(total)*100),'%')


