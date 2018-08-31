import re
import os
import string
from read import *
import pandas as pd
from pandas import ExcelWriter, ExcelFile
import numpy as np
import matplotlib.pyplot as plt
import spacy
from nltk.corpus import stopwords
import nltk
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

class Reach(object):
	def __init__(self):
		pass

	def custom_sum(self, df_list):	#Custom sum function to calculate likes as some fields have video views as well in dataset
		summ = 0		#Initialising value to zero
		for val in df_list:		#Running through the entire column
			if(type(val)!=int):	#Checking if the value is a pure integer or not
				continue	#If not, then continue to next value
			summ += val		#Else add the val to summ
		return summ


	def custom_time_sum(self, df_list):	#Custom time sum function to calculate the sum of times in the dataset by removing " hours"
		summ = 0
		for val in df_list:	#Checking for every value in the column
			val = val.replace(u' hours',u'')	#Replacing " hours" with a null string
			summ += int(val)	#Adding the integral value of hours to summ
		return summ

	def custom_time_list(self, df_list):	#Custom time sum function to calculate the sum of times in the dataset by removing " hours"
		#print(df_list)
		for i in range(0,len(df_list)):	#Checking for every value in the column
			df_list[i] = df_list[i].replace(u' hours',u'')	#Replacing " hours" with a null string
			df_list[i] = int(df_list[i])	#Adding the integral value of hours to summ
		return df_list


	def caption_hashtag_generator(self, sentence):
		nlp = spacy.load("en_core_web_sm")
		stopw = stopwords.words("english")
		noun_list = []
		sentence = re.sub(r'https?:\/\/.*\/\w*','',sentence) # Remove hyperlinks
		sentence = re.sub(r'['+string.punctuation+']+', ' ',sentence) # Remove puncutations like 's
		sentence = sentence.replace("#","")
		emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF""]+", flags=re.UNICODE) #Removes emoji
		sentence = emoji_pattern.sub(r'', sentence) # no emoji
		doc = nlp(sentence)
		temp_list = []
		for sent in doc.sents:
			for token in sent:
				token_temp = str(token)
				#print(sent)
				print(token.text, token.pos_)
				if(token.pos_=="NOUN" and token.text not in stopw):
					#print(sent)
					#print(i, token.text)
					temp_list.append(token.text)
		noun_list.append(temp_list)
		temp_list = []
		#print(noun_list)
		return noun_list



	def model(self, frame_df, no_followers=400):
		custom_time_list(frame_df['Time since posted'])
		inp = frame_df[['Followers', 'Time since posted']]

		op = frame_df[['Likes']]
		train_x, test_x, train_y, test_y = train_test_split(inp, op, test_size = 0.2, random_state = 999)
		lr = LinearRegression().fit(train_x, train_y)	#Fitting and creating a model

		pred = lr.predict(test_x)		#Predicting the answers for valdiation data

		mse = mean_squared_error(pred, test_y)	#finding the mean squared error


		try:
			model = joblib.load("models/reach_model")
		except:
			os.system("mkdir models")
			joblib.dump(lr, "models/reach_model",compress=9)
			model = joblib.load("models/reach_model")

		reach_pred = model.predict([[no_followers,10]])
		#print(reach_pred, mse)
		expected_reach = "Expected Reach is " + str(int(reach_pred-round(mse**0.5))) + "-" + str(int(reach_pred+round(mse**0.5)))
		return expected_reach


	def combine(self, followers, caption):
		df = pd.read_csv("datasets/combined_hashtag.csv")		#Reading the new csv file
		frame_df = pd.DataFrame(df)
		hash_list = self.caption_hashtag_generator(caption)
		#data_science(df, frame_df)
		expected_reach = self.model(frame_df, followers)
		op = expected_reach + '\n\n' + str(hash_list)
		print(op)
		return op
