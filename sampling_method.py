# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import random
import pandas as pd
import os
from collections import Counter
import math
import threading
from sklearn import tree
from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn import svm
import pydotplus
import collections
import merge
import sys

import pandas as pd
import fs

def log_info(df):
	size = df.shape[0]
	print 'info',df.Class.value_counts()

class SvmVoter(threading.Thread):
	def __init__(self,sampled_df):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []

	def run(self):
		self.svm()
	
	def svm(self):
		print 'svm'
		def replaceLabel(x):
			x = int(x)
			tmp = 1 if x == 4 else -1
			return tmp
		## the class label may need to be 1,-1
		x = np.array(self.sampled_df.ix[:,2:-1])
		labels = self.sampled_df.ix[:,-1].apply(replaceLabel)
		y = np.array(labels)
		#print collections.Counter(list(y))	
		clf = svm.SVC()
		scores = cross_val_score(clf, x,y, cv=5)
		print np.mean(scores)
		#weight = clf.coef_
		#print np.sort(-weight)
		#print np.argsort(-weight)

	def getTopic(self):
		return self.topics

class lassoVoter(threading.Thread):
	def __init__(self,sampled_df):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []

	def run(self):
		self.lasso()
	
	def lasso(self):
		print 'lasso'
		def replaceLabel(x):
			x = int(x)
			tmp = 1 if x == 4 else -1
			return tmp
		## the class label may need to be 1,-1
		x = np.array(self.sampled_df.ix[:,2:-1])
		labels = self.sampled_df.ix[:,-1].apply(replaceLabel)
		y = np.array(labels)
		#print collections.Counter(list(y))	
		features = list(self.sampled_df.columns.values)[2:-1]
		clf = linear_model.LogisticRegression(penalty='l1')
		#clf.fit(x,y)
		scores = cross_val_score(clf, x,y, cv=5)
		print np.mean(scores)
		'''
		weight =  clf.coef_
                fs = np.argsort(-weight)[0][0:15]
		#print -np.sort(-weight)[0][0:15]
		#print fs
	 
		for i in fs:
			self.topics.append(i+2)
		#self.topics.append(-1)
		#classifier(self.topics,self.sampled_df,1)
		'''

	def getTopic(self):
		return self.topics
	

class TreeVoter(threading.Thread):
	
	def __init__(self,sampled_df,num=10):

		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = num

	def run(self):
		self.dt()

	def dt(self):
		print 'dt'
		x = np.array(self.sampled_df.ix[:,2:-1])
		y = np.array(self.sampled_df.ix[:,-1])
		features = list(self.sampled_df.columns.values)[2:-1]
		clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=self.num)
		#clf.fit(x,y)
		scores = cross_val_score(clf, x,y, cv=5)
		print np.mean(scores)
		'''
		dot_data = tree.export_graphviz(clf,feature_names=features, out_file=None) 
		graph = pydotplus.graph_from_dot_data(dot_data) 
		graph.write_pdf("./test.pdf") 
		node_list = clf.tree_.feature
		#self.topics = [features[e] for e in node_list if e != -2]
		weight = clf.feature_importances_
   		fs = np.argsort(-weight)[0:15]
		#print(-np.sort(-weight)[0:15])
		#print(fs)
    		for i in fs:
			self.topics.append(i+2)
		#self.topics.append(-1)
		#classifier(self.topics,self.sampled_df,2)
		'''
		
	
	def getTopic(self):
		return self.topics
	
class EntropyVoter(threading.Thread):
	
	def __init__(self,sampled_df):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []	

	def run(self):
		self.entropy(self.sampled_df)
	
	def entropy(self,sampled_df):
		topic_nums = len(sampled_df.irow(0))-1
		topic_index = []
		for i in range(1,topic_nums):
			topic_index.append(i)
			#print(topic_index)

		
		choose_new = []
		choose_old = []
		samples = len(sampled_df)
		k = range(10)
		log_info(sampled_df)
		#每组根据信息增益IG(Y;Q)=H(Y)-H(Y|Q)贪心选出IG最大的10个query
		#每次迭代选出一个query，我是直接计算H(Y|Q)，选最小的。有个问题是当信息增益可能一样时，默认index最小的query（待改进）
		for iter in k:
			print("-----iter ----",iter+1)
			new_index = topic_index[1]
			max = 2
			for topic_i in topic_index:
				#print("----choose topic ---",topic_i)
				choose_new = choose_old[:]
				choose_new.append(sampled_df.iloc[:,topic_i])
				#print(choose_new)
				sum_total = 0
				for k1, group in sampled_df.groupby(choose_new):
					#print("------group-----",k1,len(group))
					p_group = len(group)/len(sampled_df)
					p = group.Class.value_counts()/len(group)
					sum =0
					for i in p:
						#print("i=",i)
						if(i==1 or i==0):
							pi = 0
						else:
							pi = -i*math.log(i)
						sum += pi
						#print(pi,sum)
					sum_total += p_group*sum
					#print("----sum_total=",sum_total)

				if(sum_total<max):
					max = sum_total
					new_index = topic_i
					#print("-------------------------min----------- ",new_index,max)

			k = new_index   #k = argmax(info_gian_c_i)
			print("----------select--",k,"-------",sampled_df.iloc[:,k].name)
			choose_old.append(sampled_df.iloc[:,k])
			self.topics.append(k)
			topic_index.remove(k)	
	
	def getTopic(self):
		return self.topics

class EntropyVoterSimple(threading.Thread):

	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num

	def run(self):
		self.entropy(self.sampled_df)

	def entropy(self,sampled_df):
	
		headers = list(sampled_df.columns)
		start = headers.index('user_topic')
		end = headers.index('Class')
		sampled_df = sampled_df.ix[:,start:end+1]
				
		#print f_num
		topic_nums = len(sampled_df.iloc[0])-1
		topic_index = []
		for i in range(1,topic_nums):
			topic_index.append(i)
			#print(topic_index)


		#print(sampled_df.iloc[:,1].name)
		samples = len(sampled_df)
		k = range(1)
		#log_info(sampled_df)
		#IG(Y;Q)=H(Y)-H(Y|Q) greedy to choose 10 query that has biggest IG
		#before : iterate 10 times : each time calculate all ,choose the smallest H(Y|Q)  
		#2017/01/05 modify iterate one time:greedy to choose 10 query that has biggest IG
		for iter in k:			
			h_i=[]
			#print("-----iter ----",iter+1)	
			count = 0
			for topic_i in topic_index:
				#print("----choose topic ---",topic_i)
				choose_new = []
				#choose_new.append(sampled_df.iloc[:,topic_i])
				choose_new = sampled_df.iloc[:,topic_i]
				#print(choose_new)
				sum_total = 0
				#choose_new.any()
				if(choose_new.any()):
					for k1, group in sampled_df.groupby(choose_new):			
						#print("------group-----",k1,len(group),len(sampled_df))
						p_group = len(group)/len(sampled_df)
						p = group.Class.value_counts()/len(group)
						sum =0
						for i in p:
							#print("i=",i)
							if(i==1 or i==0):
								pi = 0
							else:
								pi = -i*math.log(i)
							sum += pi
							#print(pi,sum)
						sum_total += p_group*sum
						#print("----sum_total=",sum_total,'=',p_group,'*',sum)
				else:
					#count+=1
					sum_total = 5				
				#print sum_total
				h_i.append(sum_total)
			#print count	
			#print h_i
			t = sorted(range(len(h_i)),key=lambda k:h_i[k],reverse=False)
			self.topics = t[:self.num]
			test = [h_i[i] for i in t[:self.num]]
			#print self.topics
			#print test
			#print self.topics



	def getTopic(self):
		return self.topics

def emsemble_sampling(ti,en,probs_file,origin_file,type=0,f_size=10):

    
    #print f_size
    time = range(ti)
    all_topic = []
    voters = []
    fs_method = fs.get_method(type)
    print str(fs_method)
    #do 10 times, according to the attribute probability prediction to sampling each time
    for t in time:
		print("----------------------iteration------------------- no.",t+1)

		if(en):
			sampled_df = merge.get_file(probs_file,origin_file)
		else:
			sampled_df = pd.read_csv(origin_file,dtype={"user_topic":str,"Class":str})
		#test 
		#sampled_df.to_csv('./test_0.csv')
		
		##feature_selection	
		#fs_method = fs.get_method(t)
		#print str(fs_method)		
		voter = fs_method(sampled_df,f_size)		
		voters.append(voter)
		
    for v in voters:
        v.setDaemon(True)
        v.start()
		
    for v in voters:
        #v.setDaemon(True)
        v.join()
    for v in voters:
       all_topic += v.getTopic()

    feature = []
    for i in Counter(all_topic).most_common(f_size):
        a = str(i[0]+1)+"+"+str(i[1])+"+"+sampled_df.iloc[:,i[0]+1].name
        print(a)       
        feature.append(i[0])   
    #print(feature)	
    return feature


def main():

    #all = []
    #path = r'./test_simple'
    rule = {'Ethnicity':'White','Age':[40,50,70,90,100]}
    filename = './white_old'
    feature_size = 200
	
    probs_file = filename + '.pro'
    origin_file = './user_topic_origin.csv'
    origin_fill = filename + '_origin_fill.csv'
    goal_file = filename + '_goal.csv'	
    goal_fill = filename + '_goal_fill.csv'
    origin_file = origin_fill    #choose fill>
    goal_file = goal_fill
	
    feature = emsemble_sampling(probs_file,origin_file,feature_size)
    classifier(feature,goal_file,i)
	
   

if __name__ == '__main__':
    main()
