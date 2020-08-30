#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 21:02:55 2020

@author: dhanji
"""

from tensorflow.keras.layers import Input,Dense,LSTM,Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import tensorflow as tf
import re
import spacy
from spacy.lang.en import English
from sklearn.model_selection import train_test_split

TAG_RE = re.compile(r'<[^>]+>')

class DataPreprocessing:

    def __init__(self,filePath):
        self.filePath = filePath
        self.uniqueWords = set()
        self.nlp = English()
        self.file = open(filePath,'r')

    def preProcessText(self,sen):
        valueToBePredict = sen[-2]
        # Removing html tags
        sen = sen[0:sen.find('\t')-1]
        sentence = self.remove_tags(sen.lower())
        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = self.removeStopWordAndDoLemmatize(sentence)
        return [sentence,valueToBePredict]
    
    def removeStopWordAndDoLemmatize(self,sen):
        filterText = []
        for word in sen.split(' '):
            if self.nlp.vocab[word].is_stop == False:
                filterText.append(word)
        doc = self.nlp(' '.join(filterText))
        return ' '.join([token.lemma_ for token in doc])
    
    def remove_tags(self,text):
        return TAG_RE.sub('', text)

    def getUniqueWord(self,x):
        uniqueWords = set()
        for sentence in x:
            for word in text_to_word_sequence(sentence):
                uniqueWords.add(word)
        return uniqueWords
   
    def getPreProcessData(self):
        lines = self.file.readlines();
        x = []
        y = []
        for sen in lines:
            xx,yy = self.preProcessText(sen)
            x.append(xx)
            y.append(int(yy))
        uniqueWord = self.getUniqueWord(x)
        return x,y,uniqueWord

class SentimentAnalysis:
    def __init__(self):
        self.intToWordMap = {}
        self.wordsToIntMap = {}
        self.maxLengthOfSentence = 0
                
    def mapWordsToIntAndViceVersa(self,uniqueVocab):
        index = 0
        for value in uniqueVocab:
            self.intToWordMap[index] = value
            self.wordsToIntMap[value] = index
            index = index + 1
    
    def setMaxLengthOfSentence(self,x):
        for sentence in x:
            words = text_to_word_sequence(sentence)
            if len(words) > self.maxLengthOfSentence:
                self.maxLengthOfSentence = len(words)
    
    def mapEachWordWithOneHotEncoding(self,x,y):
        oneHotRepresentation = np.eye(vocabCount,dtype=np.uint8)
        self.encodedSeq = []
        self.sentenceArray = []
        index = 0
        for sentence in x:
            words = text_to_word_sequence(sentence)
            encodSeq = []
            for word in words:
                encodSeq.append(oneHotRepresentation[self.wordsToIntMap[word]])
            length = len(encodSeq)
            if length < self.maxLengthOfSentence:
                for k in range(0,self.maxLengthOfSentence-length):
                    encodSeq.append(np.zeros((1,vocabCount))[0])
                self.sentenceArray.append(encodSeq)
            else:
                self.sentenceArray.append(encodSeq)
            index = index + 1
        self.sentenceArray = np.array(self.sentenceArray).reshape(-1,self.maxLengthOfSentence,vocabCount)

    def splitDataIntoTrainAndTest(self,x,y):
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.sentenceArray,y,test_size=0.20,random_state=42)
    
    def trainModel(self,vocabCount):
        inputLayer = Input(shape=(None,vocabCount))
        embeddingLayer = Embedding(vocabCount,vocabCount,input_length=self.maxLengthOfSentence)
        lstmLayer = LSTM(128)(inputLayer)
        output = Dense(1,activation='sigmoid')(lstmLayer)
        self.model = Model(inputLayer,output)
        self.model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])
        self.model.fit(self.x_train,self.y_train,epochs=200)
        
filePath = './Dataset/amazon_cells_labelled.txt'
preProcessData = DataPreprocessing(filePath)
x,y,uniqueWords = preProcessData.getPreProcessData()
vocabCount = len(uniqueWords)

classifier = SentimentAnalysis()
classifier.mapWordsToIntAndViceVersa(uniqueWords)
classifier.setMaxLengthOfSentence(x)
classifier.mapEachWordWithOneHotEncoding(x,y)
classifier.splitDataIntoTrainAndTest(x,y)
classifier.trainModel(vocabCount)

predictedValue = classifier.model.predict(classifier.x_test)
predictedValue = np.where(predictedValue > 0.5,1,0)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(classifier.y_test,predictedValue)
