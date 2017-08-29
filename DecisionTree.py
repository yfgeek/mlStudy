import csv
import numpy
import scipy
from sklearn import neighbors, preprocessing, tree
from sklearn.feature_extraction import DictVectorizer


def loadData():
    with open('2014data.csv', 'rt', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        # print(headers)
        trainSet = list()
        labelSet = []
        rowDict = {}
        for row in reader:
            y = list(row)[54]
            labelSet.append(y)
            for i in range(3, 52):
                rowDict[headers[i]] = row[i]
            trainSet.append(rowDict)
    return {'data': trainSet, 'target': labelSet}



datas = loadData()
vec = DictVectorizer()
dummyX = vec.fit_transform(datas['data']) .toarray()

print(" dummyX" +str(dummyX))
print(vec.get_feature_names())

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(datas['target'])
print("dummY" + str(dummyY))

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY);
print("clf" + str(clf))

with open("allEletronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)


with open('pre.csv', 'rt', encoding='utf-8') as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
        x = list(row)[3:52]
        prelabel = numpy.array(x).reshape(1, -1)
        presresult = clf.predict(prelabel)
        print('预测值：', str(presresult))
        print(' 真实值：', row[54])


