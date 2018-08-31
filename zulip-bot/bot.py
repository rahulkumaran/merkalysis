import pprint
import zulip
import sys
import re
import json
import httplib2
import os
from reach import Reach
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

p = pprint.PrettyPrinter()
BOT_MAIL = "merkalysis-bot@merkalysis.zulipchat.com"

class ZulipBot(object):


	def __init__(self):
		self.client = zulip.Client(site="https://merkalysis.zulipchat.com/api/")
		self.subscribe_all()
		self.market = Reach()

		print("done init")
		self.subkeys = ["reach"]



	def subscribe_all(self):
		json = self.client.get_streams()["streams"]
		streams = [{"name": stream["name"]} for stream in json]
		self.client.add_subscriptions(streams)




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
				#print(token.text, token.pos_)
				if(token.pos_=="NOUN" and token.text not in stopw):
					#print(sent)
					#print(i, token.text)
					temp_list.append(token.text)
		noun_list.append(temp_list)
		temp_list = []
		#print(noun_list)
		return noun_list




	def model(self, frame_df, followers):
		self.custom_time_list(frame_df['Time since posted'])
		inp = frame_df[['Followers', 'Time since posted']]
		op = frame_df[['Likes']]
		train_x, test_x, train_y, test_y = train_test_split(inp, op, test_size = 0.2, random_state = 999)
		lr = LinearRegression().fit(train_x, train_y)	#Fitting and creating a model
		pred = lr.predict(test_x)		#Predicting the answers for valdiation data
		mse = mean_squared_error(pred, test_y)	#finding the mean squared error
		#print(mse)
		reach_model = joblib.load("models/reach_model")
		reach_pred = reach_model.predict([[followers,10]])
		#print(reach_pred, mse)
		expected_reach = "Expected Reach is " + str(int(reach_pred-round(mse**0.5))) + "-" + str(int(reach_pred+round(mse**0.5)))
		return expected_reach




	def combine(self, followers, caption):
		df = pd.read_csv("datasets/combined_hashtag.csv")		#Reading the new csv file
		frame_df = pd.DataFrame(df)
		hash_list = self.caption_hashtag_generator(caption)
		#print(hash_list)
		hash_list = "Hashtags that you could possibly use to widen your post's reach are:\n#" + " #".join(hash_list[0])
		#data_science(df, frame_df)
		expected_reach = self.model(frame_df, int(followers))
		op = expected_reach + "\n\n" + hash_list
		print(op)
		return op




	def process(self, msg):
		content = msg["content"].split()
		sender_email = msg["sender_email"]
		ttype = msg["type"]
		stream_name = msg['display_recipient']
		stream_topic = msg['subject']
		#print(content)
		if sender_email == BOT_MAIL:
			return 
		print("Sucessfully heard.")
		if(content[0].lower() == "merkalysis" or content[0] == "@**merkalysis**"):
			if content[1].lower() == "reach":
				followers = content[2]
				ip = content[3:]
				ip = " ".join(ip)
				#print(ip)
				message = self.combine(followers, ip)
				self.client.send_message({
					"type": "stream",
					"subject": msg["subject"],
					"to": msg["display_recipient"],
					"content": message
					})
			if(content[1].lower() not in self.subkeys):
				self.client.send_message({
					"type": "stream",
					"subject": msg["subject"],
					"to": msg["display_recipient"],
					"content": "Hey there! :blush:"
					})

def main():
	bot = ZulipBot()
	bot.client.call_on_each_message(bot.process)

if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("Thanks for using Merkalysis Bot. Bye!")
		sys.exit(0)






















