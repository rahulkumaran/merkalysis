import json
import requests

class Merkalysis(object):
	def __init__(self, followers, caption):
		self.followers = followers
		self.caption = caption
		self.url = "http://localhost:8000/api/"


	def get_response(self, url):
		'''Returns the reponse got from the url'''
		return requests.get(url)

	def get_json(self,response):
		'''Pass in the requests reponse and returns the json data'''
		return response.json()

	def reach(self):
		response = self.get_response(self.url + "followers=" + str(self.followers))
		json_obj = self.get_json(response)
		return json_obj['reach_pred']

	def hashtags(self):
		response = self.get_response(self.url + "caption=" + self.caption)
		json_obj = self.get_json(response)
		return json_obj['hashtag_suggest']

	def reach_hashtags(self):
		response = self.get_response(self.url +  "followers=" + str(self.followers) + "/caption=" + self.caption)
		json_obj = self.get_json(response)
		return json_obj['reach_pred'] + '\n' + json_obj['hashtag_suggest']	

if(__name__=="__main__"):
	obj = Merkalysis(500, "I love machine learning")
	market_reach = obj.reach()
	hashtags = obj.hashtags()
	hybrid = obj.reach_hashtags()
	print(market_reach, hashtags, hybrid)
