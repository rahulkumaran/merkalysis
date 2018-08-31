from django.shortcuts import render
from django.template.context import RequestContext
import string
import pandas as pd
from pandas import ExcelWriter, ExcelFile
from django.http import HttpResponse
import numpy as np
import matplotlib.pyplot as plt
import spacy
from nltk.corpus import stopwords
import nltk
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import re
from django.http import JsonResponse
import os


ml = pd.read_excel('datasets/hashtag.xls', sheet_name='#Machinelearning')		#Reading the #machinelearning sheet from the .xls file
bc = pd.read_excel('datasets/hashtag.xls', sheet_name='#blockchain')		#Reading the #blokchain sheet from the .xls file
ai = pd.read_excel('datasets/hashtag.xls', sheet_name='#artificialintelligence')	#Reading the #artificialintelligence sheet from the .xls file
su = pd.read_excel('datasets/hashtag.xls', sheet_name='#startup')			#Reading the #startup sheet from the .xls file
prod = pd.read_excel('datasets/hashtag.xls', sheet_name='#product')		#Reading the #product sheet from the .xls file
dev = pd.read_excel('datasets/hashtag.xls', sheet_name='#development')		#Reading the #development sheet from the .xls file


frame_ml = pd.DataFrame(ml)		#Converting the read ml sheets into dataframes
frame_bc = pd.DataFrame(bc)		#Converting the read bc sheets into dataframes
frame_ai = pd.DataFrame(ai)		#Converting the read ai sheets into dataframes
frame_su = pd.DataFrame(su)		#Converting the read su sheets into dataframes
frame_prod = pd.DataFrame(prod)		#Converting the read prod sheets into dataframes
frame_dev = pd.DataFrame(dev)		#Converting the read dev sheets into dataframes


combined_data = pd.concat([frame_ml,frame_bc,frame_ai,frame_su,frame_prod,frame_dev])	#Merging all the hashtag dataframes got from the separate sheets in the .xls file

combined_data.to_csv("datasets/combined_hashtag.csv")		#Converting the entire set of data into a new csv file with all hashtags merged

df = pd.read_csv("datasets/combined_hashtag.csv")
frame_df = pd.DataFrame(df)

hashtags = []		#Initialising hashtags list

for hs in df["Hashtags"]:	#Reading every hashtag that was used in posts
	hashtags += hs.split("#")




pd.options.mode.chained_assignment = None

'''
CODES AND THEIR MEANINGS:
ml -> Machine Learning
bc -> Blockchain
ai -> Artificial Intelligence
su -> StartUp
prod -> Product
dev -> Development
'''

######################################### CUSTOM FUNCTIONS TO PERFORM TASKS #############################################

def custom_sum(df_list):	#Custom sum function to calculate likes as some fields have video views as well in dataset
	summ = 0		#Initialising value to zero
	for val in df_list:		#Running through the entire column
		if(type(val)!=int):	#Checking if the value is a pure integer or not
			continue	#If not, then continue to next value
		summ += val		#Else add the val to summ
	return summ


def custom_time_sum(df_list):	#Custom time sum function to calculate the sum of times in the dataset by removing " hours"
	summ = 0
	for val in df_list:	#Checking for every value in the column
		val = val.replace(u' hours',u'')	#Replacing " hours" with a null string
		summ += int(val)	#Adding the integral value of hours to summ
	return summ

def custom_time_list(df_list):	#Custom time sum function to calculate the sum of times in the dataset by removing " hours"
	#print(df_list)
	for i in range(0,len(df_list)):	#Checking for every value in the column
		df_list[i] = df_list[i].replace(u' hours',u'')	#Replacing " hours" with a null string
		df_list[i] = int(df_list[i])	#Adding the integral value of hours to summ
	return df_list

def dataset_hashtag_generator(df_list):
	try:
		nlp = spacy.load("en_core_web_sm")
	except:
		os.system("python3 -m spacy download en")
		nlp = spacy.load("en_core_web_sm")

	try:
		stopw = stopwords.words("english")
	except:
		os.system("python3 -m nltk.downloader stopwords")
		stopw = stopwords.words("english")

	noun_list = []
	for i in range(0, len(df_list)-1):
		print(df_list[i])
		try:
			if(np.isnan(df_list[i])):
				continue
		except:
			df_list[i] = re.sub(r'https?:\/\/.*\/\w*','',df_list[i]) # Remove hyperlinks
			df_list[i] = re.sub(r'['+string.punctuation+']+', ' ',df_list[i]) # Remove puncutations like 's
			df_list[i] = df_list[i].replace("#","")
			emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF""]+", flags=re.UNICODE) #Removes emoji
			df_list[i] = emoji_pattern.sub(r'', df_list[i]) # no emoji
			doc = nlp(df_list[i])
			temp_list = []
			for sent in doc.sents:
				for token in sent:
					token_temp = str(token)
					if(token.pos_=="NOUN" and token.text not in stopw):
						#print(sent)
						#print(i, token.text)
						temp_list.append(token.text)
			noun_list.append(temp_list)
			temp_list = []
	print(noun_list)
	return noun_list

def caption_hashtag_generator(sentence):
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

def data_science(df, df_list):
	hashtags = []		#Initialising hashtags list

	for hs in df["Hashtags"]:	#Reading every hashtag that was used in posts
		hashtags += hs.split("#")	#Every field in Hashtags column contains more than one hashtag so need to identify all. That's why using the split at # thing

	#print(hashtags)

	for elem in range(0,len(hashtags)):	#If we print hashtags list before, it gives a non breaking space(\xa0) so need to replace it with null character or empty string
		hashtags[elem] = hashtags[elem].replace(u'\xa0',u'')	#Replacement happens here

	#print(hashtags)

	fdist = nltk.FreqDist(hashtags)		#freqdist function present in nltk

	fdist.plot(20)			#Finding top 20 hashtags

	frame_ml.plot(x="Followers", y="Likes", figsize=(50,100), style="o")

	frame_ai.plot(x="Followers", y="Likes", figsize=(50,100), style="o")
	frame_bc.plot(x="Followers", y="Likes", figsize=(50,100), style="o")
	frame_su.plot(x="Followers", y="Likes", figsize=(50,100), style="o")
	frame_prod.plot(x="Followers", y="Likes", figsize=(50,100), style="o")
	frame_dev.plot(x="Followers", y="Likes", figsize=(50,100), style="o")
	plt.show()

	mean_likes_ml = round(custom_sum(frame_ml['Likes'].tolist())/len(frame_ml))
	mean_likes_bc = round(custom_sum(frame_bc['Likes'].tolist())/len(frame_bc))
	mean_likes_ai = round(custom_sum(frame_ai['Likes'].tolist())/len(frame_ai))
	mean_likes_su = round(custom_sum(frame_su['Likes'].tolist())/len(frame_su))
	mean_likes_prod = round(custom_sum(frame_prod['Likes'].tolist())/len(frame_prod))
	mean_likes_dev = round(custom_sum(frame_dev['Likes'].tolist())/len(frame_dev))

	mean_time_ml = round(custom_time_sum(frame_ml['Time since posted'].tolist())/len(frame_ml))
	mean_time_bc = round(custom_time_sum(frame_bc['Time since posted'].tolist())/len(frame_bc))
	mean_time_ai = round(custom_time_sum(frame_ai['Time since posted'].tolist())/len(frame_ai))
	mean_time_su = round(custom_time_sum(frame_su['Time since posted'].tolist())/len(frame_su))
	mean_time_prod = round(custom_time_sum(frame_prod['Time since posted'].tolist())/len(frame_prod))
	mean_time_dev = round(custom_time_sum(frame_dev['Time since posted'].tolist())/len(frame_dev))

	mean_follow_ml = round(np.sum(frame_ml['Followers'])/len(frame_ml))
	mean_follow_bc = round(np.sum(frame_bc['Followers'])/len(frame_bc))
	mean_follow_ai = round(np.sum(frame_ai['Followers'])/len(frame_ai))
	mean_follow_su = round(np.sum(frame_su['Followers'])/len(frame_su))
	mean_follow_prod = round(np.sum(frame_prod['Followers'])/len(frame_prod))
	mean_follow_dev = round(np.sum(frame_dev['Followers'])/len(frame_dev))
	 
	like_rate_ml = round(mean_likes_ml/mean_time_ml)
	like_rate_bc = round(mean_likes_bc/mean_time_bc)
	like_rate_ai = round(mean_likes_ai/mean_time_ai)
	like_rate_su = round(mean_likes_su/mean_time_su)
	like_rate_prod = round(mean_likes_prod/mean_time_prod)
	like_rate_dev = round(mean_likes_dev/mean_time_dev)

	print("MEAN LIKES\tMEAN TIME\tRATE OF LIKES(PER HR)\tMEAN FOLLOWERS")
	print(str(mean_likes_ml) + "\t\t" + str(mean_time_ml) + "\t\t" + str(like_rate_ml) + "\t\t\t" + str(mean_follow_ml))
	print(str(mean_likes_bc) + "\t\t" + str(mean_time_bc) + "\t\t" + str(like_rate_bc) + "\t\t\t" + str(mean_follow_bc))
	print(str(mean_likes_ai) + "\t\t" + str(mean_time_ai) + "\t\t" + str(like_rate_ai) + "\t\t\t" + str(mean_follow_ai))
	print(str(mean_likes_su) + "\t\t" + str(mean_time_su) + "\t\t" + str(like_rate_su) + "\t\t\t" + str(mean_follow_su))
	print(str(mean_likes_prod) + "\t\t" + str(mean_time_prod) + "\t\t" + str(like_rate_prod) + "\t\t\t" + str(mean_follow_prod))
	print(str(mean_likes_dev) + "\t\t" + str(mean_time_dev) + "\t\t" + str(like_rate_dev) + "\t\t\t" + str(mean_follow_dev))

	print("\n\nAVERAGE LIKE RATE COMBINING ALL HASHTAGS:")
	print(round((like_rate_ml + like_rate_bc + like_rate_ai + like_rate_su + like_rate_prod + like_rate_dev)/6))

	print("Likes after 3 hours would be "+str(round((like_rate_ml + like_rate_bc + like_rate_ai + like_rate_su + like_rate_prod + like_rate_dev)/6)*3))

	'''
	It's very clear from the mean of likes that dev is a moving hashtag to get more likes.
	But this might be because of various factors:
		(1) The user posting with #development might already have more followers
		(2) The size of the dataset is too small to come to a conlusion (125-130 only)
		(3) There might be more videos so views have been ommitted giving a better mean
	'''

def model(frame_df, no_followers=400):
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


def home(request):
    	if request.method == "POST":
    			return render(request,'analysis/index.html',{"context":"Blahblah"})
    	return render(request,'analysis/index.html',{})
		

def get_resp(request,*args,**kwargs):    	
		model = joblib.load("static/models/reach_model")
		reach_pred = model.predict([[int(kwargs['pk1']),10]])
		reach_pred = reach_pred[0][0]
		hashtags = caption_hashtag_generator(kwargs['state'])	
		return JsonResponse({
			'caption':kwargs['state'],
			'followers':kwargs['pk1'],
			'reach_pred':"Expected Reach is " + str(int(reach_pred-round(mse**0.5))) + "-" + str(int(reach_pred+round(mse**0.5))),
			'hashtag_suggest':"#"+str("#".join(hashtags[0]))
			})

def formV(request,*args,**kwargs):
	return render(request,"analysis/form.html",{})	