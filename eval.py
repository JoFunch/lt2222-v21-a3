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

vowels = sorted(['y', 'é', 'ö', 'a', 'i', 'å', 'u', 'ä', 'e', 'o'])

def a(f):
	mm = []
	with open(f, "r") as q:
		for l in q:
			mm += [c for c in l]

	mm = ["<s>", "<s>"] + mm + ["<e>", "<e>"]
	return mm, list(set(mm))

def g(x, p):
	z = np.zeros(len(p))
	if x in p:
		z[p.index(x)] = 1
	return z

def b(input_data, vocab):
	gt = []
	gr = []
	for v in range(len(input_data) - 4):
		if input_data[v+2] not in vocab:
			continue
		
		h2 = vocab.index(input_data[v+2])
		gt.append(h2)
		r = np.concatenate([g(x, vocab) for x in [input_data[v], input_data[v+1], input_data[v+3], input_data[v+4]]])
		gr.append(r)

	return np.array(gr), np.array(gt)



def predict(model, input_vector):

	pred = model(torch.Tensor(input_vector)).detach().numpy()
	# print(pred)
	pred_vowels = np.argmin(np.abs(pred), axis=1)
	# print(pred_vowels)
	return pred_vowels



def evaluate_accuracy(true_vector, predicted_vector):
	true_positives = sum(true_vector == predicted_vector)
	# point = 0
	# for i in range(len(true_vector)):
	# 	if true_vector[i] == predicted_vector[i]:
	# 		point += 1

	total = len(true_vector)
	accuracy = (true_positives / total)
	print(accuracy)
	print('Current Model Accuracy: ', accuracy*100, '%')


def overwrite(processed_files, pred_vowels):

	lst = []
	diff = len(processed_files)-len(pred_vowels)
	idx = 0
	for t in range(len(processed_files)-diff):
		if processed_files[t] in vowels:
			lst.append(vowels[pred_vowels[idx]])
			idx += 1
		else:
			lst.append(processed_files[t])
			idx += 1

	return lst



def perplexity(l):
	perpl  = torch.exp(l)
	print(perpl)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("input_data", type=str)
	parser.add_argument("pickle", type=str)
	parser.add_argument("overwriting_file", type=str) 
	# parser.add_argument("--perplexity", action='store_true')   

	args = parser.parse_args()

	model = torch.load(args.pickle)
	model.eval()


	#preproccessing funct A
	preprocessed_input = a(args.input_data) #Test-data!

	#create data, funct B
	index_vowels, one_vector = b(preprocessed_input[0], model.vocab) #index-vector from test and vocab from TRAIN! 

	#test model / predict
	pred = predict(model, index_vowels)

	#evaluate / print accuracy  
	accu = evaluate_accuracy(one_vector, pred)

	#use model to write!
	with open(args.overwriting_file, 'w') as f:
		f.write(''.join(overwrite(preprocessed_input[0], pred))) 


	# if args.perplexity:
	# 	perp = perplexity(model.loss)
	# 	# print(perp)




