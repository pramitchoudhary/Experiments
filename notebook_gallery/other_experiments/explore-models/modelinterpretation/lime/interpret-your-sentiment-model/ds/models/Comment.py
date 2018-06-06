import os, datetime
import pandas as pd

class Thread(object):
	def __init__(self):
		self.history = []
		self.add_comment('Created Thread','SYSTEM')

	def add_comment(self, comment, name):
		ID = len(self.history)
		new_comment = {'ID':ID, 'ts':datetime.datetime.now(), 'comment':comment, 'name':name}
		
		self.history.append(new_comment)

	def delete_comment(self,ID):
		del self.history[ID]


	def display_history(self):
		return pd.DataFrame(self.history).sort('ID', ascending = True)