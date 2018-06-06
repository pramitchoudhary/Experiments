import numpy as np
import pandas as pd
import warnings
from lime.lime_text import LimeTextExplainer
warnings.filterwarnings('ignore')



def TextInterpret(text, predict):
	lte = LimeTextExplainer()
	explanation = lte.explain_instance(text, predict)
	explanation.show_in_notebook()
	return explanation

def Interpret(text, predict, preprocessing_function = None, classes = None):
	
	if preprocessing_function:
		processor = preprocessing_function()
		X = processor(x)
	else:
		X = x

	baseline = predict(X)[0]
	token_effects = {}

	for token in doc:
	    index = token.i
	    vectors = np.array([doc[j].vector for j in range(len(doc)) if j != index])
	    vector = vectors.mean(axis=0)
	    new_prediction = predict(vector)[0]
	    token_effects[token.orth_] =   baseline - new_prediction
	df = pd.DataFrame.from_dict(token_effects, orient = 'index')
	try:
		df.columns = classes
	except:
		pass

	return df
