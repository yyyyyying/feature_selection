#coding=utf-8
import merge
import pandas as pd
import sampling_method
import classifier
import sys
#sys.path.append('/home/linlt/code/new_crawl/fill')
sys.path.append('/home/linlt/code/cluster')
import fill


def main():
	
	######## 1 prepare 4 files (1 probs + 1 fill + 2 goal), just change the rule and filename 
	### 1.1 get probs_file 
	#rule = {'Occupation':'Student'}	
	#rule = {'Gender':'Female'}
	#rule = {'Ethnicity':'White'}
	
	rule = {'Ethnicity':'White','Age':[40,50,60,70,80,90,100,110]}	
	filename = './new_crawl0213/white_old'		
	probs_file = filename +  '.pro'
	threshold = 0.5
	#merge.get_probs_file(rule,probs_file)
	
	### 1.2 get goal_file and goal_fill_file
	#origin_file = './user_topic_origin.csv'
	origin_file = './new_crawl/topic_matric_origin_balan.csv'
	#origin_file = './new_crawl/topic_matric_origin.csv'
	origin_fill = filename + '_origin_fill'+str(threshold)+'.csv'
	goal_file = filename + '_goal.csv'	
	goal_fill = filename + '_goal_fill'+str(threshold)+'.csv'
	
	#if you run feature selection , put the annotation  
	#goal_file = filename + '_balan.csv'	
	#merge.preprocess(origin_file,goal_file)	
	#merge.get_goal_file(probs_file,goal_file,origin_file)
	
	df = pd.read_csv(origin_file)		
	df = fill.fill(rule,df,'rake',threshold)	
	df.to_csv(origin_fill,index=False)		
	merge.get_goal_file(probs_file,goal_fill,origin_fill)	
		
	####### 2. begin ensemble_FeatureSelection 	
	
	f_size = 10
	time =1
	en = False	
	type_list = [4,5,6,3,0,2,1,8,9,7]
	#type_list = [0,1,2,3,4,5,6,7,8,9]
	#type_list = [7]
	#[0 svmvoter, 1 lassovoter, 2 dtvoter, 3 Kbesetvoter,
	# 4 sampling_method.EntropyVoterSimple, 5 VarianceVoter, 6 CorelationVoter, 7 WrapperVoter, 8 RndLassovoter, 9 GBDTvoter]
	
	origin_file = origin_fill    #choose fill>
	goal_file = goal_fill 
	
	print '------',time,' time----------ensemble:',en,'------'
	print goal_file
	for method_type in type_list:
		if(en):
			feature = sampling_method.emsemble_sampling(time,en,probs_file,origin_file,method_type,f_size)
		else:
			feature = sampling_method.emsemble_sampling(time,en,probs_file,goal_file,method_type,f_size)
		classifier.main(feature,goal_file,f_size)
		
		

if __name__ == '__main__':
	main()
