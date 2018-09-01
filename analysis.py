import re
import math
from collections import Counter
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
import sendgrid
from sendgrid.helpers.mail import *


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
email_list = ["rahuldravid313@gmail.com","rahul160562@mechyd.ac.in", "gillarohith1@gmail.com"]
sg = sendgrid.SendGridAPIClient(apikey="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")
WORD = re.compile(r'\w+')

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


def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])
     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)
     return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)


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

def sendmail(email_id, caption):
	from_email = Email("rahulkumaran313@gmail.com", name="Rahul Arulkumaran")
	to_email = Email(email_id)
	subject = "Weekly Updates From Merkalysis"
	content = Content("text/html", "<html><body><p>A post is up on Instagram from _rahul_kumaran_'s account with a caption \"" + caption + "\"</p></body></html>")
	mail = Mail(from_email, subject, to_email, content)
	response = sg.client.mail.send.post(request_body=mail.get())
	return response

def Main():
	df = pd.read_csv("datasets/combined_hashtag.csv")		#Reading the new csv file
	frame_df = pd.DataFrame(df)
	caption = input("What's your caption?\n")
	no_followers = int(input("How many followers do you have on Instagram?\n"))
	hash_list = caption_hashtag_generator(caption)
	#data_science(df, frame_df)
	expected_reach = model(frame_df, no_followers)
	print(expected_reach + '\n\n' + str(hash_list))
	'''for email_id in email_list:
		response = sendmail(email_id, caption)
		print(response.status_code)'''

	'''t1 = text_to_vector("machine")
	t2 = text_to_vector("machine learning")

	cosine = get_cosine(t1,t2)
	print(cosine)'''




