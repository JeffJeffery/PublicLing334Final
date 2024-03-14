import math
import re
import os
import json

#this code was adapted from not only assigment 3 but also a CS 348 assigment
class Bayes_Classifier:

    # planning:
    # remove punctuation
    # plus one smoothing
    # make everything lowercase

    # fields tracking features
    totalFeatures = 0
    totalPositiveFeatures = 0
    totalNegativeFeatures = 0
    positiveFeatureDict = {}
    negativeFeatureDict = {}

    # fields for tracking total postive and negative and total reviews
    totalReviews = 0
    totalPositiveReviews = 0
    totalNegativeReviews = 0

    # fields for removing things to make better predicitons
    stopWords = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
                 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with']

    # Smoothing Value
    alpha = 1
    savePath = '../bot/model.json'

    def __init__(self, train_dir='data/train'):
        self.classes = os.listdir(train_dir)
        self.train_data = {c: os.path.join(train_dir, c) for c in self.classes}
    

    def train(self):
        for classification in self.classes:
            c_docs = open(self.train_data[classification]).readlines()
            for doc in c_docs:
				# first decide if we are looking at 5 star or 1 star reviews
                self.totalReviews += 1
                if classification == 'whiny':
                    workingDict = self.negativeFeatureDict
                    workingTotal = "Negative"
                    self.totalNegativeReviews += 1
                else:
                    workingDict = self.positiveFeatureDict
                    workingTotal = "Positive"
                    self.totalPositiveReviews += 1

				# lets break it into words:
                words = doc.split()

				# now for every word or feature, change relevant tallies
                for word in words:
					# first normalize the word
                    word = word.lower()

					# if it is in stop words, dont worry about it:
                    if word in self.stopWords:
                        continue
                    if word in workingDict:
                        workingDict[word] += 1
                    else:
                        workingDict[word] = 1

					# now update total counts both of our specific outcome and the total
                    if workingTotal == "Negative":
                           self.totalNegativeFeatures += 1
                    else:
                        self.totalPositiveFeatures += 1
                    self.totalFeatures += 1
        print("Done Training")

    def classify(self, line):
		# we dont need to look at score, That would be cheating!

		# lets break it into words:
        words = line.split()
        
		# compute the prob that the reivew is negative and that it is positive, return the highest one
		# here i am weighting the negative reviews as if they were 4 times as likely as actually
		# this is due to the positive bias in our set and the incorrect classification of reviews as postive
        posReviewProb = (self.totalPositiveReviews * 5) / (self.totalReviews + (self.totalPositiveReviews * 4))
		# sum of the logs
        negReviewProb = (self.totalNegativeReviews ) / (self.totalReviews +  (self.totalPositiveReviews * 4))

		# intialize the reviews, we will be taking the log of eveything to allow us to add, we will fix it later
        ProbReviewPositive = 0
        ProbReviewNegative = 0

        for word in words:
			# first normalize the word
            word = word.lower()

			# if it is in stop words, dont worry about it:
            if word in self.stopWords:
                continue

			# compute the positive prob
            posProb = math.log(self.positiveProb(word))
			# compute the negative prob
            negProb = math.log(self.negativeProb(word))

			# add it to respective totals
            ProbReviewPositive += posProb
            ProbReviewNegative += negProb

		# finally multiply by the probability of a reivew being that class to get the final result
        ProbReviewPositive += math.log(posReviewProb)
        ProbReviewNegative += math.log(negReviewProb)
        predicteScore = "not" if ProbReviewPositive > ProbReviewNegative else "whiny"
        return predicteScore


    def positiveProb(self, word):
        numbWordInPositive = self.positiveFeatureDict[word] if word in self.positiveFeatureDict else 0
        prob = (numbWordInPositive + self.alpha) / \
            (self.totalPositiveFeatures + (self.alpha * self.totalFeatures))
        return prob

    def negativeProb(self, word):
        numbWordInNegative = self.negativeFeatureDict[word] if word in self.negativeFeatureDict else 0
        prob = (numbWordInNegative + self.alpha) / \
            (self.totalNegativeFeatures + (self.alpha * self.totalFeatures))
        return prob

    def saveModel(self):
        modelParameters = {}
        modelParameters["totalFeatures"] = self.totalFeatures
        modelParameters["totalPositiveFeatures"] = self.totalPositiveFeatures
        modelParameters["totalNegativeFeatures"] = self.totalNegativeFeatures
        modelParameters["positiveFeatureDict"] = self.positiveFeatureDict
        modelParameters["negativeFeatureDict"] = self.negativeFeatureDict
        modelParameters["totalReviews"] = self.totalReviews
        modelParameters["totalPositiveReviews"] = self.totalPositiveReviews
        modelParameters["totalNegativeReviews"] = self.totalNegativeReviews
        modelParameters["stopWords"] = self.stopWords
        modelParameters["alpha"] = self.alpha
        with open(self.savePath, 'w') as f:
            json.dump(modelParameters, f)
        print("Model saved!")

    def loadModel(self):
        f = open(self.savePath)
        modelParameters = json.load(f)
        self.totalFeatures = modelParameters["totalFeatures"]
        self.totalPositiveFeatures = modelParameters["totalPositiveFeatures"]
        self.totalNegativeFeatures = modelParameters["totalNegativeFeatures"]
        self.positiveFeatureDict = modelParameters["positiveFeatureDict"]
        self.negativeFeatureDict = modelParameters["negativeFeatureDict"]
        self.totalReviews = modelParameters["totalReviews"]
        self.totalPositiveReviews = modelParameters["totalPositiveReviews"]
        self.totalNegativeReviews = modelParameters["totalNegativeReviews"]
        self.stopWords = modelParameters["stopWords"]
        self.alpha = modelParameters["alpha"]