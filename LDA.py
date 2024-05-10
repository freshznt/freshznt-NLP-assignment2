import os
from typing import List, Dict
from gensim import corpora, models
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import jieba


class LDAC(object):
    def __init__(self, tokenMode: str, docNum: int, docLength: int, topicNum: int, crossValNum: int):
        self.tokenMode = tokenMode
        self.docNum = docNum
        self.docLength = docLength
        self.topicNum = topicNum
        self.crossValNum = crossValNum
        self.Dataset: List[resDoc] = []

    def preprocessData(self, stopWordsPath: str, corpusPath: str):
        stopWords = []
        for file in os.listdir(stopWordsPath):
            with open(os.path.join(stopWordsPath, file), 'r', encoding='utf-8') as stopWordFile:
                stopWords.extend([line.strip() for line in stopWordFile.readlines()])
        txtNameWithDatas = []
        for filePath in os.listdir(corpusPath):
            with open(os.path.join(corpusPath, filePath), 'r', encoding='utf-8') as corpusFile:
                rawTxt = corpusFile.read()
                rawTxt = rawTxt.replace('----〖新语丝电子文库(www.xys.org)〗', '')
                rawTxt = rawTxt.replace('本书来自www.cr173.com免费txt小说下载站', '')
                txtData = rawTxt.replace('更多更新免费电子书请关注www.cr173.com', '')
                if self.tokenMode == 'word':
                    txtData = list(jieba.lcut(txtData))
                words = [word for word in txtData if word not in stopWords and not word.isspace()]

                words = np.array(words)
                txtData = [words[i:i + self.docLength] for i in
                           range(0, len(words) - len(words) % self.docLength, self.docLength)]
                txtNameWithDatas.append((filePath.split('.txt')[0], txtData))

        sizeArray = np.array([len(tuple[1]) for tuple in txtNameWithDatas])
        numArrayFloat = self.docNum * sizeArray / sizeArray.sum()
        numArrayInt = np.floor(numArrayFloat)
        while numArrayInt.sum() < self.docNum:
            maxErrorIdx = np.argmax(numArrayFloat - numArrayInt)
            numArrayInt[maxErrorIdx] += 1
        for i, (label, docs) in enumerate(txtNameWithDatas):
            nowParagNums = numArrayInt[i]
            ticIdxArr = np.random.choice(range(len(docs)), size=int(nowParagNums), replace=False)
            self.Dataset.extend([resDoc(label, docs[paragIdx]) for paragIdx in ticIdxArr])

    def LDAClassify(self):
        trainAccuracySum = 0
        testAccuracySum = 0
        np.random.shuffle(self.Dataset)
        for iVal in range(self.crossValNum):
            groupSize = len(self.Dataset) // self.crossValNum
            startIdx = iVal * groupSize
            endIdx = startIdx + groupSize
            testSet = self.Dataset[startIdx:endIdx]
            trainSet = self.Dataset[:startIdx] + self.Dataset[endIdx:]

            #LDA model
            trainDataset = [data.para for data in trainSet]
            trainDictionary = corpora.Dictionary(trainDataset)
            trainCorpus = [trainDictionary.doc2bow(text) for text in trainDataset]
            ldaModel = models.LdaModel(corpus=trainCorpus, id2word=trainDictionary, num_topics=self.topicNum)

            # Classifier
            docLabel = [data.label for data in trainSet]
            prob_para = np.array(ldaModel.get_document_topics(trainCorpus, minimum_probability=0.0))[:, :, 1]
            model = SVC(kernel='linear', probability=True)
            model.fit(prob_para, docLabel)
            trainAccuracy = model.score(prob_para, docLabel)
            print('Train accuracy is {:.4f}'.format(trainAccuracy))
            testDataset = [data.para for data in testSet]
            testLabel = [data.label for data in testSet]
            testCorpus = [trainDictionary.doc2bow(text) for text in testDataset]
            testProbability = np.array(ldaModel.get_document_topics(testCorpus, minimum_probability=0.0))[:, :, 1]
            testAccuracy = model.score(testProbability, testLabel)
            print('Test accuracy is {:.4f}'.format(testAccuracy))

            trainAccuracySum += trainAccuracy
            testAccuracySum += testAccuracy

        return trainAccuracySum / self.crossValNum, testAccuracySum / self.crossValNum

class resDoc(object):
    def __init__(self, label: str, para: list):
        self.label = label
        self.para = para


if __name__ == '__main__':
    rootPath = './data/'
    corpusFilePath = os.path.join(rootPath, 'corpus_utf8')
    stopWordsFilePath = os.path.join(rootPath, 'stopwords')

    docNum = 1000
    modeList = ['char', 'word']
    docLengthMap = {'char': [20, 100, 500, 1000, 3000], 'word': [20, 100, 500, 1000]}
    topicNumList = [30, 40, 60, 70, 80, 90]
    crossValNum = 10
    resultFilePath = './LDAC.xlsx'
    resFileWriter = pd.ExcelWriter(resultFilePath, mode='w')

    for mode in modeList:
        resultTrainDict, resultTestDict = {}, {}
        for docLength in docLengthMap[mode]:
            resultTrainDict[str(docLength)], resultTestDict[str(docLength)] = [], []
            for topicNum in topicNumList:
                ldaCModel = LDAC(tokenMode=mode, docNum=docNum, docLength=docLength,
                                 topicNum=topicNum, crossValNum=crossValNum)
                ldaCModel.preprocessData(stopWordsPath=stopWordsFilePath, corpusPath=corpusFilePath)
                meanTrain, meanTest = ldaCModel.LDAClassify()
                print(f'Mode: {mode}, docLength: {docLength}, topicNum: {topicNum}:')
                print(f'Average train accuracy: {meanTrain:.4f}, average test accuracy: {meanTest:.4f}')
                resultTrainDict[str(docLength)].append(meanTrain)
                resultTestDict[str(docLength)].append(meanTest)

        pd.DataFrame(resultTrainDict, index=topicNumList).to_excel(resFileWriter, sheet_name=mode + 'train', index=True)
        pd.DataFrame(resultTestDict, index=topicNumList).to_excel(resFileWriter, sheet_name=mode + 'test', index=True)

    resFileWriter.close()