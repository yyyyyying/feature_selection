# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def SvmClassifier(sampled_df):
	print 'SVM'
	x,y = getXY(sampled_df)
	#print collections.Counter(list(y))	
	clf = svm.SVC(kernel='linear')
	accuracy = cross_val_score(clf, x,y, cv=5)
	precision = cross_val_score(clf, x,y, cv=5, scoring='precision')
	f1 = cross_val_score(clf, x,y, cv=5, scoring='f1')
	recall = cross_val_score(clf, x,y, cv=5, scoring='recall')
	roc_auc = cross_val_score(clf, x,y, cv=5, scoring='roc_auc')
	
	print np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(f1),np.mean(roc_auc)


def lassoClassifier(sampled_df):
	print 'lasso'
	x,y = getXY(sampled_df)
	#print collections.Counter(list(y))	
	features = list(sampled_df.columns.values)[2:-1]
	clf = linear_model.LogisticRegression(penalty='l1')
	accuracy = cross_val_score(clf, x,y, cv=5)
	precision = cross_val_score(clf, x,y, cv=5, scoring='precision')
	f1 = cross_val_score(clf, x,y, cv=5, scoring='f1')
	recall = cross_val_score(clf, x,y, cv=5, scoring='recall')
	roc_auc = cross_val_score(clf, x,y, cv=5, scoring='roc_auc')
	
	print np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(f1),np.mean(roc_auc)
	
def DTClassifier(sampled_df,f_size):
	print 'DT'
	x,y = getXY(sampled_df)
	clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=f_size)
	accuracy = cross_val_score(clf, x,y, cv=5)
	precision = cross_val_score(clf, x,y, cv=5, scoring='precision')
	f1 = cross_val_score(clf, x,y, cv=5, scoring='f1')
	recall = cross_val_score(clf, x,y, cv=5, scoring='recall')
	roc_auc = cross_val_score(clf, x,y, cv=5, scoring='roc_auc')
	
	print np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(f1),np.mean(roc_auc)

def GNBayesClassifier(sampled_df):
	print 'GaussianNB'
	x,y = getXY(sampled_df)
	clf = GaussianNB()
	accuracy = cross_val_score(clf, x,y, cv=5)
	precision = cross_val_score(clf, x,y, cv=5, scoring='precision')
	f1 = cross_val_score(clf, x,y, cv=5, scoring='f1')
	recall = cross_val_score(clf, x,y, cv=5, scoring='recall')
	roc_auc = cross_val_score(clf, x,y, cv=5, scoring='roc_auc')
	
	print np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(f1),np.mean(roc_auc)

def KNNClassifier(sampled_df):
	print 'KNN'
	x,y = getXY(sampled_df)
	clf = KNeighborsClassifier()
	accuracy = cross_val_score(clf, x,y, cv=5)
	precision = cross_val_score(clf, x,y, cv=5, scoring='precision')
	f1 = cross_val_score(clf, x,y, cv=5, scoring='f1')
	recall = cross_val_score(clf, x,y, cv=5, scoring='recall')
	roc_auc = cross_val_score(clf, x,y, cv=5, scoring='roc_auc')
	
	print np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(f1),np.mean(roc_auc)
	
def LRClassifier(sampled_df):
	print 'LR'
	x,y = getXY(sampled_df)
	clf = LogisticRegression(penalty='l1')
	accuracy = cross_val_score(clf, x,y, cv=5)
	precision = cross_val_score(clf, x,y, cv=5, scoring='precision')
	f1 = cross_val_score(clf, x,y, cv=5, scoring='f1')
	recall = cross_val_score(clf, x,y, cv=5, scoring='recall')
	roc_auc = cross_val_score(clf, x,y, cv=5, scoring='roc_auc')
	
	print np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(f1),np.mean(roc_auc)
	
def getXY(df):
	def replaceLabel(x):
		x = int(x)
		tmp = 1 if x == 4 else -1
		return tmp
	headers = list(df.columns)
	start = headers.index('user_topic')
	end = headers.index('Class')
	x = df.ix[:,start + 1:end].as_matrix()
	y = df.ix[:,end].apply(replaceLabel).as_matrix()

	return x,y	
				
def main(feature,csvname,num=10):
	
	feature = [i + 1 for i in feature]
	print feature
	#csvname = './white_old_goalfile.csv'
	goal_df = pd.read_csv(csvname,dtype={"user_topic":str,"Class":str})
	
	headers = list(goal_df.columns)
	start = headers.index('user_topic')
	end = headers.index('Class')
	goal_df = goal_df.ix[:,start:end+1]
	
	feature = [0] + feature + [-1]

	#print feature 
	df = goal_df.ix[:,feature]	
	
   	#df = goal_df
	#num = 2695
	df.to_csv('./test.csv',index=False)
	
	SvmClassifier(df)
	lassoClassifier(df)
	DTClassifier(df,num)
	#GNBayesClassifier(df)
	KNNClassifier(df)
	LRClassifier(df)
		

if __name__ == '__main__':
	feature = [26, 448, 47, 818, 180, 610, 1708, 698, 968, 100]
	csvname = './doc2vec/white_old_goal_fill.csv'
	#csvname = './test/iris.csv'
	main(feature,csvname)