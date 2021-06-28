import tensorflow
import torch
import os
import sys
import argparse
import numpy as np
import pandas as pd
import models
import glob
from sklearn.metrics import accuracy_score



def a(f):
	mm = []
	with open(f, "r") as q:
		for l in q:
			mm += [c for c in l]

	mm = ["<s>", "<s>"] + mm + ["<e>", "<e>"]
	return mm, list(set(mm))

def g(x, p):
	z = np.zeros(len(p))
	z[p.index(x)] = 1
	return z

def b(u, p):
	gt = []
	gr = []
	for v in range(len(u) - 4):
		if u[v+2] not in vowels:
			continue
		
		h2 = vowels.index(u[v+2])
		gt.append(h2)
		r = np.concatenate([g(x, p) for x in [u[v], u[v+1], u[v+3], u[v+4]]])
		gr.append(r)

	return np.array(gr), np.array(gt)



def predict(model, input_vector):

	pred = model(torch.Tensor(input_vector)).detach().numpy()
	print(pred)
	pred_vowels = np.argmax(np.abs(pred), axis=1)
	print(pred_vowels)
	return pred_vowels



def evaluate_accuracy():
	pass


def overwrite(input_file, output_file):
	pass







if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("input_data", type=str)
	parser.add_argument("pickle", type=str)
	parser.add_argument("overwriting_file", type=str)   

	args = parser.parse_args()


	#preproccessing funct A
	preprocessed_input = a(args.input_data)

	#create data, funct B
	index_vowels, one_vector = b(preprocessed_input[0], model.vocab)

	#load model and set evaluation mode for the model
	model = torch.load(args.pickle)
	model.eval()

	#test model / predict
	predicted_vowels = predict(model, one_vector)

	#evaluate / print accuracy  

	#use model to write!

