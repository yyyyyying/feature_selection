import pandas as pd

class Condition:
	def __init__(self,rules):
		self.rules = rules 

	def getRule(self):
		return self.rules

	def extract(self,df):
		for key in self.rules:
			value = self.rules[key]
			if type(value) == list:
				df = df.ix[df[key].isin(value)]
			else:
				df = df.ix[df[key] == value]
		return df


if __name__ == '__main__':
	filename = '/home/linlt/code/probability/data.csv'
	rule =  {'Gender':'Male','Age':20,'Location':['California','Nebraska']}
	df = pd.read_csv(filename)
	#print df.describe()
	con = Condition(rule)
	print con.extract(df)

		
