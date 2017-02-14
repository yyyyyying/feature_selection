#coding=utf-8

## Notice: PCA can only extract feature num which is the minimun of sample_num and feature num
import numpy as np
import threading
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn.feature_selection import RFE
import sampling_method
import math


class svmvoter(threading.Thread):
	'''
	http://scikit-learn.org/stable/modules/feature_selection.html
		
	'''
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num



	def run(self):
		self.svm()



	def svm(self):
		#print 'svm'
		x,y = getXY(self.sampled_df)
		clf = svm.SVC(kernel='linear')
		clf.fit(x,y)
		score = clf.coef_[0]
		score = list(score)
		t = sorted(range(len(score)),key=lambda k:score[k],reverse=True)
		self.topics = t[:self.num]
		pass


	
	def getTopic(self):
		return self.topics


class lassovoter(threading.Thread):
	'''
	http://scikit-learn.org/stable/modules/feature_selection.html
		
	'''
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num



	def run(self):
		self.l1()



	def l1(self):
		#print 'lasso'
		x,y = getXY(self.sampled_df)
		#print x,y
		clf = linear_model.LogisticRegression(penalty='l1')
		clf.fit(x,y)
		score = clf.coef_[0]
		score = list(score)
		t = sorted(range(len(score)),key=lambda k:score[k],reverse=True)
		#print score
		self.topics = t[:self.num]
		pass


	
	def getTopic(self):
		return self.topics

		
		
class dtvoter(threading.Thread):
	'''
	http://scikit-learn.org/stable/modules/feature_selection.html
		
	'''
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num



	def run(self):
		self.dt()



	def dt(self):
		#print 'dt'
		x,y = getXY(self.sampled_df)
		clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=self.num)
		clf.fit(x,y)
		score = clf.feature_importances_
		t = sorted(range(len(score)),key=lambda k:score[k],reverse=True)
		self.topics = t[:self.num]
		pass


	
	def getTopic(self):
		return self.topics


class Kbesetvoter(threading.Thread):
	'''
	http://scikit-learn.org/stable/modules/feature_selection.html
		
	'''
	def __init__(self,sampled_df,f_num,function=mutual_info_classif):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num
		self.func = mutual_info_classif

	def run(self):
		self.kbest()

	def kbest(self):
		x,y = getXY(self.sampled_df)
		kb = SelectKBest(self.func, k=self.num)
		kb.fit_transform(x, y)
		score = kb.scores_ 
		t = sorted(range(len(score)),key=lambda k:score[k],reverse=True)
		self.topics = t[:self.num]


	
	def getTopic(self):
		return self.topics

class VarianceVoter(threading.Thread):
	
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num
	
	def run(self):
		self.variance()


	def variance(self):
		x,y = getXY(self.sampled_df)
		v = list(np.var(x,axis=0))
		index = sorted(range(len(v)),key = lambda i:v[i],reverse=True)
		#print index
		self.topics = index[:self.num]
	
	def getTopic(self):
		return self.topics
		
		
class CorelationVoter(threading.Thread):
	
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num
	
	def run(self):
		self.corelation()


	def corelation(self):
		x,y = getXY(self.sampled_df)
		row,cols = x.shape
		v = [ np.corrcoef(x[:,c],y)[0,1] for c in range(cols)]
		index = sorted(range(len(v)),key = lambda i:v[i] if not math.isnan(v[i]) else -2,reverse=True)
		self.topics = index[:self.num]
	
	def getTopic(self):
		return self.topics

class WrapperVoter(threading.Thread):
	
	def __init__(self,sampled_df,f_num,base=svm.SVC(kernel="linear",C=1)):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num
		self.base = base
	
	def run(self):
		self.rfe()


	def rfe(self):
		x,y = getXY(self.sampled_df)
		rfe = RFE(estimator=self.base,n_features_to_select=self.num)
		rfe.fit(x,y)
		self.topics = list(rfe.get_support(indices=True))

	
	def getTopic(self):
		return self.topics		
		

class GBDTVoter(threading.Thread):
	
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num
	
	def run(self):
		self.gbdt()


	def gbdt(self):
		x,y = getXY(self.sampled_df)
		dt = GradientBoostingClassifier()
		dt.fit(x,y)
		#print dt.feature_importances_
		score = list(dt.feature_importances_)  
		t = sorted(range(len(score)),key=lambda k:score[k],reverse=True)
		self.topics = t[:self.num]

	
	def getTopic(self):
		return self.topics			

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
	
class RndLassovoter(threading.Thread):
	'''
	http://scikit-learn.org/stable/modules/feature_selection.html
		
	'''
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num



	def run(self):
		self.l1()



	def l1(self):
		x,y = getXY(self.sampled_df)
		clf = linear_model.RandomizedLogisticRegression()
		clf.fit(x,y)
		score = list(clf.scores_)
		t = sorted(range(len(score)),key=lambda k:score[k],reverse=True)
		self.topics = t[:self.num]
		pass


	
	def getTopic(self):
		return self.topics

class rfvoter(threading.Thread):
	'''
	http://scikit-learn.org/stable/modules/feature_selection.html
		
	'''
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num



	def run(self):
		self.rf()



	def rf(self):
		x,y = getXY(self.sampled_df)
		clf = RandomForestClassifier(criterion='entropy',max_depth=self.num)
		clf.fit(x,y)
		score = clf.feature_importances_
		t = sorted(range(len(score)),key=lambda k:score[k],reverse=True)
		self.topics = t[:self.num]
		pass


	
	def getTopic(self):
		return self.topics

def get_method(type=0):

	method_list = [svmvoter,lassovoter,dtvoter,Kbesetvoter,sampling_method.EntropyVoterSimple,VarianceVoter,CorelationVoter,WrapperVoter,RndLassovoter,GBDTVoter,rfvoter]

	return method_list[type]
	
	

if __name__ == '__main__':
	#filename = ('./women_goal_fill.csv')
	filename = ('./en_file/white_old_goal_fill.csv')
	df = pd.read_csv(filename)
	#pv = Kbesetvoter(df,10)
	#pv.kbest()
	#pv = lassovoter(df,10)
	#pv.l1()
	#pv = svmvoter(df,10)
	#pv.svm()
	#pv = dtvoter(df,10)
	#pv.dt()
	#pv = VarianceVoter(df,10)
	#pv.variance()
	#pv = CorelationVoter(df,10)
	#pv.corelation()
	#pv = WrapperVoter(df,10)
	#print pv.rfe()
	#pv = RndLassovoter(df,10)
	#pv.l1()
	#pv = GBDTVoter(df,10)
	#pv.gbdt()
	pv = rfvoter(df,10)
	pv.rf()
	print pv.getTopic()

