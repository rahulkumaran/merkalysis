import pandas as pd
import matplotlib.pyplot as plt
import nltk

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

frame_csv = pd.read_csv("datasets/combined_hashtag.csv")
df = pd.DataFrame(frame_csv)

hashtags = []		#Initialising hashtags list

for hs in df["Hashtags"]:	#Reading every hashtag that was used in posts
	hashtags += hs.split("#")	#Every field in Hashtags column contains more than one hashtag so need to identify all. That's why using the split at # thing

#print(hashtags)

''''for elem in range(0,len(hashtags)):	#If we print hashtags list before, it gives a non breaking space(\xa0) so need to replace it with null character or empty string
	hashtags[elem] = hashtags[elem].replace(u'\xa0',u'')	#Replacement happens here

fdist = nltk.FreqDist(hashtags)		#freqdist function present in nltk

fdist.plot(20)			#Finding top 20 hashtags

#frame_ml.plot(x="Followers", y="Likes", figsize=(5,10), style="o")

#frame_ai.plot(x="Followers", y="Likes", figsize=(5,10), style="o")
plt.show()'''


