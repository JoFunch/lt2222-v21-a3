# LT2222 V21 Assignment 3

Your name:

## Part 1
Function A: Simple pre-processing. 
	Returns a tuple with two lists: 
		1. is a list with a character-level tokenization/split of the entire document, splits everything on character level and includes meta descriptors like \n-ewline, etc.
		2. is a list containing the set of the characters used in 1. 

Function G:
	takes the two lists of fun. A as arguments and creates a dataframe
	The dataframe has the length of the set of A2 and maps the usages of characters by its index if any.


Function B:
	takes two lists as arguments, U, the list of characters in context, and P, list containing the set of characters used.

	the purpose of B is to map statistics of vowel-appearance and map it to a ML-trainable format.

	B iterates the characters in context by index+2 (the newspaper articles, preprocessed) and looks for vowels found in the vowel-list predefined.
	If it finds a vowel in the list, several things happen.
	to GT, the vowel-list-index is appended (h2)

	following that (variable "r"), the code considers several positions around its current iteration.
	r takes into consideration four characters by index: -2 of the vowel, -1 of the vowel, and +1 and +2 of the vowel of the current iteration (u[v+3/4])

	r (using function G to create set-character-lengthed arrays) creates numpy-arrays based on the appearence of these previous and ahead-appearing characters. 

Args:

	k = adjusted number of nodes in each of the hidden layers of the NN seen in Train.py's model.

	r = adjusted number of epochs (training iteration loops/cycles for optimization)

	m = input-text-file taken by A for preprocessing

	h = position of torch.save (path to model)


## Part 2




## Part 3

## Bonuses

## Other notes
