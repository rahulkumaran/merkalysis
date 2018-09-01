import sendgrid
import os
from sendgrid.helpers.mail import *


email_list = ["rahuldravid313@gmail.com","rahul160562@mechyd.ac.in", "freeflow.ai@gmail.com"]

def main_func(email_id):
	from_email = Email("rahulkumaran313@gmail.com", name="Rahul Arulkumaran")
	to_email = Email(email_id)
	subject = "Testing Merkalysis"
	content = Content("text/html", "<html><body><p>Test Email from Instalysis to figure out whether the API is functioning proper or not!</p></body></html>")
	mail = Mail(from_email, subject, to_email, content)
	response = sg.client.mail.send.post(request_body=mail.get())
	return response



if(__name__=="__main__"):
	sg = sendgrid.SendGridAPIClient(apikey="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")
	for email_id in email_list:
		response = main_func(email_id)
		print(response.status_code)
