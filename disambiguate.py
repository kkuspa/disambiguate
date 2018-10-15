# Author: Kai Kuspa
# Copyright © 2018
import os
import sys
import csv
import time
import pydot
import string
import random
import itertools
import numpy as np
import pandas as pd
import operator as op
import cPickle as pickle
from datetime import datetime
from itertools import groupby
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import jaccard_similarity_score
from sklearn.model_selection import train_test_split

baseline_prediction = []


def remove_punctuation(x):
	exclude = set(string.punctuation)
	try:
		x = ''.join(ch for ch in x if ch not in exclude)
	except:
		pass
	return x

def update_progress(progress):
	barLength = 30 # Modify this to change the length of the progress bar
	status = ""
	if isinstance(progress, int):
		progress = float(progress)
	if not isinstance(progress, float):
		progress = 0
		status = "error: progress var must be float\r\n"
	if progress < 0:
		progress = 0
		status = "Halt...\r\n"
	if progress >= 1:
		progress = 1
		status = "Done...\r\n"
	block = int(round(barLength*progress))
	text = "\rPercent: [{0}] {1:.4f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
	sys.stdout.write(text)
	sys.stdout.flush()


def ncr(n, r):
	# helper function to provide ceiling for random sample
	# n choose r
	r = min(r, n-r)
	numer = reduce(op.mul, xrange(n, n-r, -1), 1)
	denom = reduce(op.mul, xrange(1, r+1), 1)
	return numer//denom

def read_challenge_file(path, subfolder, fname, numRows = None):
	""" Read a csv and place into a dataframe
	"""
	df = pd.read_csv(path + subfolder + '/' + fname, nrows=numRows)
	df.columns = [x[1:] if x[0]=='_' else x for x in df.columns] #remove leading underscore for itertuples later
	print fname + " read..."
	return df

#hashify names separated to optimize for speed
def hashify_names(df, value_word, progress=True):
	""" Create a hashtable from a dataframe using "namkeys"
		namekey = "first initial + lastname"
	"""
	c = 0
	l = len(df)
	hashmap = {}
	for row in df.itertuples(index=True, name='Rows'):
		c = c + 1
		progress = float(c)/float(l)
		update_progress(progress) #for the user's sanity

		name = str(getattr(row, "firstname"))[0] + str(getattr(row, "lastname")) #maybe this should be firstname lastname not first initial lastname
		if name in hashmap:
			hashmap[name].append(getattr(row, value_word))
		else:
			hashmap[name] = [getattr(row, value_word)]
	return hashmap

def hashify_lists(df, key_word, value_word_list, progress=True):
	""" Create hashtable of lists objs
	"""
	c = 0
	l = len(df)
	hashmap = {}
	for row in df.itertuples(index=True, name='Rows'):
		c = c + 1
		progress = float(c)/float(l)
		update_progress(progress) #for the user's sanity

		key = str(getattr(row, key_word))
		values = []
		for val in value_word_list:
			values.append(getattr(row, val))
		if key in hashmap:
			hashmap[key].append(values)
		else:
			hashmap[key] = [values]
	return hashmap

def hashify(df, key_word, value_word, progress=True):
	""" Create hashtable of one df column to another
	"""
	c = 0
	l = len(df)
	hashmap = {}
	for row in df.itertuples(index=True, name='Rows'):
		c = c + 1
		progress = float(c)/float(l)
		update_progress(progress) #for the user's sanity

		key = str(getattr(row, key_word))
		if key in hashmap:
			hashmap[key].append(getattr(row, value_word))
		else:
			hashmap[key] = [getattr(row, value_word)]
	return hashmap

def pair_generator(numbers):
	"""Return an iterator of random pairs from a list of numbers.
		Keeps track of already generated pairs
	"""
	used_pairs = set()

	while True:
		pair = random.sample(numbers, 2)
		# Avoid generating both (1, 2) and (2, 1)
		pair = tuple(sorted(pair))
		if pair not in used_pairs:
			used_pairs.add(pair)
			yield pair

def rnd_sample_pairs(assets, nSamples):
	"""Returns random pairs without replacement
	"""
	rnd_pairs = []
	n_assets = len(assets)
	if ncr(n_assets, 2) < nSamples:
		rnd_pairs = list(itertools.combinations(assets, 2))
	else:
		gen = pair_generator(assets)
		for i in xrange(nSamples):
			rnd_pairs.append(gen.next())
	return rnd_pairs

def find_name_identity(candidates, fname_t, lname_t):
	"""Return an identity of interest from an asset with multiple coauthors
	"""
	identity = None

	# The way we index the articles we must choose the correct target author to compare in asset a and b
	for coauthor in candidates:
		if coauthor[0] != coauthor[0] or coauthor[2] != coauthor[2]:
			#skip name if first or last name is blank
			continue
		#if perfect match, select name and return
		if coauthor[0] == fname_t and coauthor[2] == lname_t:
			return coauthor
		#if first initial matches and last name matches, set first name to initial, but keep searching for perfect match.
		elif coauthor[0][0] == fname_t[0] and coauthor[2] == lname_t:
			coauthor[0] = coauthor[0][0]
			identity = coauthor
		
	if identity is None:
		return -1

	return identity

def get_f_name_similarity(fname_a, fname_b):
	"""Return first name similarity.
	"""
	if fname_a != fname_b and len(fname_a) > 1 and len(fname_b) > 1:
		#this never happens because our definition of ambiguity demands that one coauth has the same "namekey" which includes first initial
		return 0
	elif fname_a != fname_b and (len(fname_a) == 1 or len(fname_b) == 1):
		return 1
	elif fname_a == fname_b and (len(fname_a) == 1 or len(fname_b)== 1):
		return 2
	else:
		#if fname_a == fname_b and len(fname_a) > 1 and len(fname_b) > 1. Perfect Match!
		return 3

def get_m_name_similarity(mname_a, mname_b):
	"""Return Middle name similarity.
	"""
	if mname_a == mname_b and mname_a != "":
		return 0
	elif mname_a != mname_a and mname_b != mname_b:
		return 1
	elif (mname_a != mname_a) != (mname_b != mname_b):
		return 2
	else:
		# perfect match
		return 3

def compare_name_similarity(identity_A, identity_B):
	"""Compare  name similarity of two identities.
	"""
	fname_a = identity_A[0]
	fname_b = identity_B[0]
	mname_a = identity_A[1]
	mname_b = identity_B[1]

	#Some initialed names have a ".". Clean this.
	if fname_a == fname_a: fname_a = fname_a.replace(".", "") 
	if fname_b == fname_b: fname_b = fname_b.replace(".", "") 
	if mname_a == mname_a: mname_a = mname_a.replace(".", "")
	if mname_b == mname_b: mname_b = mname_b.replace(".", "") 

	fname_similarity = get_f_name_similarity(fname_a, fname_b)
	mname_similarity = get_m_name_similarity(mname_a, mname_b)

	return fname_similarity, mname_similarity

def inverse_doc_frequency(fcount, target):
	"""return log(IDF(word)) where IDF = L/freq(word)
	"""
	if fcount[target] > 0:
		return np.log(sum(fcount)/fcount[target])
	else:
		return -1

def jaccard_distance(A, B):
	"""return Jaccardian Distance Similarity Metric
	"""
	if A != A or B != B:
		return 0

	#filter our common english words and tokenize
	s=set(stopwords.words('english'))
	A = filter(lambda w: not w in s, A.split())
	B = filter(lambda w: not w in s, B.split())

	#sklearn requires comparison vectors be the same length
	longer = max(len(A), len(B))
	shorter = min(len(A), len(B))
	diff = longer-shorter

	if len(A) is shorter:
		for i in xrange(diff):
			A.append("")
	else:
		for i in xrange(diff):
			B.append("")

	return jaccard_similarity_score(A, B)

def get_common_elements(asset_a, asset_b, asset_keywords):
	"""return list of matching elements from two asset keyword lists
	"""
	if asset_a not in asset_keywords or asset_b not in asset_keywords:
		return -1

	keywords_a = asset_keywords[asset_a]
	keywords_b = asset_keywords[asset_b]

	return list(set(keywords_a).intersection(keywords_b))
	
def get_keyw_idf(asset_a, asset_b, asset_keywords, keyword_hist):
	"""return keyword IDF metric
	"""
	keyw_idf = 0
	shared_keywords = get_common_elements(asset_a, asset_b, asset_keywords)
	if shared_keywords == -1:
		return -1
	for keyword in shared_keywords:
		keyw_idf = keyw_idf + inverse_doc_frequency(keyword_hist, keyword)
	return keyw_idf

def get_jour_idf(asset_a, asset_b, asset_info, journal_hist):
	"""return keyword IDF metric
	"""
	if asset_a not in asset_info or asset_b not in asset_info:
		return -1

	#asset_info is set up as a list of lists
	if asset_info[asset_a][0][2] == asset_info[asset_b][0][2]:
		return inverse_doc_frequency(journal_hist, remove_punctuation(asset_info[asset_a][0][2]).lower())
	else:
		return 0

def get_jour_date_diff(asset_a, asset_b, asset_info):
	"""return journal date difference in days
	"""
	if asset_a not in asset_info or asset_b not in asset_info:
		return -1

	#print asset_info[asset_a][0][2]
	publish_date_a = datetime.strptime(asset_info[asset_a][0][1], '%Y-%m-%d')
	publish_date_b = datetime.strptime(asset_info[asset_b][0][1], '%Y-%m-%d')

	diff = abs((publish_date_b - publish_date_a).days)
	
	return diff

def compare_emails(email_A, email_B):
	"""return email similarity metric
	"""
	if email_A != email_A or email_B != email_B:
		return 0
	elif email_A != email_B:
		return 1
	else:
		# perfect Match
		return 2


def generate_similarity_profile(pair, pid, pids_info, asset_coauths, asset_keywords, asset_info, lname_hist, keyword_hist, journal_hist):
	"""return a list of features computed between two assets and their given identity target
	"""
	sim_profile = []

	asset_a = str(pair[0])
	asset_b = str(pair[1])

	fname_t = pids_info[pid][0][0]
	lname_t = pids_info[pid][0][2]

	if asset_a not in asset_coauths or asset_b not in asset_coauths:
		# skip this pair because an asset isn't found in our dataset
		return -1

	identity_A = find_name_identity(asset_coauths[asset_a], fname_t, lname_t)
	identity_B = find_name_identity(asset_coauths[asset_b], fname_t, lname_t)

	if identity_A is -1 or identity_B is -1:
		return -1

	auth_fst, auth_mid = compare_name_similarity(identity_A, identity_B)
	auth_lname_idf = inverse_doc_frequency(lname_hist, lname_t)
	aff_jacc = jaccard_distance(identity_A[4], identity_B[4])

	# TODO: aff_tf_idf = affiliation_termFrequency_idf()

	keyw_share = get_common_elements(asset_a, asset_b, asset_keywords)
	if keyw_share != -1: keyw_share = len(keyw_share)

	keyw_idf = get_keyw_idf(asset_a, asset_b, asset_keywords, keyword_hist)

	jour_shared_idf = get_jour_idf(asset_a, asset_b, asset_info, journal_hist)
	jour_date_diff = get_jour_date_diff(asset_a, asset_b, asset_info)
	if jour_date_diff == -1:
		# use jour_date_diff to check if assets are in asset_info
		title_jacc = -1
	else:
		title_jacc = jaccard_distance(asset_info[asset_a][0][0], asset_info[asset_b][0][0])

	email_sim = compare_emails(identity_A[3], identity_B[3])

	sim_profile = [auth_fst, auth_mid, auth_lname_idf, aff_jacc, keyw_share, keyw_idf, jour_shared_idf, jour_date_diff, title_jacc, email_sim]

	return sim_profile

def make_features(ambiguous_assets, assets_pids, pids_assets, targets_pids, 
	asset_coauths, asset_keywords, asset_info, pids_info, lname_hist, keyword_hist, journal_hist):

	"""Use the training data to create a list of feature vectors and corresponding labels
	"""

	feature_vec = []
	label_vec = []

	n_targets = len(targets_pids)
	target_counter = 0

	for name in targets_pids:
		target_counter = target_counter + 1

		pid = targets_pids[name] #this returns a list bc hashifynames always gives lists, so just take the first (and only) element
		pid = str(pid[0])


		#Choose pairs from matches
		target_assets = pids_assets[pid]
		ambig_assets = ambiguous_assets[name]
		not_target_assets = [x for x in ambig_assets if x not in target_assets]

		rnd_pos_pairs = rnd_sample_pairs(target_assets, 100)

		#Choose pairs from not matches
		if len(target_assets) > 20 and len(not_target_assets) > 20:
			#want 15 because 20^2 = 400 around how many samples I wanted for [1:0] matches
			target_assets = np.random.choice(target_assets, 20)
			not_target_assets = np.random.choice(not_target_assets, 20)
		else:
			lowest = min(len(target_assets), len(not_target_assets))
			if lowest != 0:
				target_assets = np.random.choice(target_assets, lowest)
				not_target_assets = np.random.choice(not_target_assets, lowest)
			else:
				target_assets = []
				not_target_assets = []
		
		rnd_neg_pairs = list(itertools.product(target_assets, not_target_assets))

		c = 0
		l = len(rnd_pos_pairs) + len(rnd_neg_pairs)
		print "Making features for " + str(name) + ". " +str(target_counter) + " of " + str(n_targets)

		# Generate features from match pairs
		for pair in rnd_pos_pairs:
			c = c + 1
			progress = float(c)/float(l)
			update_progress(progress) #for the user's sanity

			features = generate_similarity_profile(pair, pid, pids_info, asset_coauths, asset_keywords, asset_info, lname_hist, keyword_hist, journal_hist)
			if features != -1:
				feature_vec.append(features)
				label_vec.append(1)
				#make baseline prediction based on name similarity
				if feature_vec[0] == 2 or feature_vec[0] == 3:
					baseline_prediction.append(1)
				else:
					baseline_prediction.append(0)
				
		# Generate Features from not match pairs
		for pair in rnd_neg_pairs:
			c = c + 1
			progress = float(c)/float(l)
			update_progress(progress) #for The user's sanity
			features = generate_similarity_profile(pair, pid, pids_info, asset_coauths, asset_keywords, asset_info, lname_hist, keyword_hist, journal_hist)
			if features != -1 and -1 not in features:
				feature_vec.append(features)
				label_vec.append(0)
				if feature_vec[0] == 2 or feature_vec[0] == 3:
					baseline_prediction.append(1)
				else:
					baseline_prediction.append(0)

	print "Number of features: " + str(len(feature_vec))

	return np.array(feature_vec), np.array(label_vec)


def make_predictions(rf_model, ambiguous_assets, assets_pids, pids_assets, targets_pids, 
	asset_coauths, asset_keywords, asset_info, pids_info, lname_hist, keyword_hist, journal_hist):
	"""Use the random forest model, test data, and manually curated test label list to create a list of feature vectors and corresponding predictions
	"""

	predict_sim_profiles = []
	unknown_assets = []
	unknown_pids = []

	for name in targets_pids:
		print "Predicting assets for " + str(name)
		pid = targets_pids[name] #this returns a list bc hashifynames always gives lists, so just take the first (and only) element
		pid = str(pid[0])

		target_assets = pids_assets[pid]
		ambig_assets = ambiguous_assets[name]

		known_asset = target_assets[0] #only need one, a more sophisticated approach would randomnly sample from the data we have if multiple assets found

		#if not certain about any articles, don't make predictions for this person_id
		if known_asset == -1:
			continue

		predict_pairs = []
		
		#Generate pairs
		for unknown_asset in ambig_assets:

			pair = (known_asset, unknown_asset)
			predict_pairs.append(pair)
			unknown_assets.append(unknown_asset)
			unknown_pids.append(pid)
		
		#Generate features from pairs
		for pair in predict_pairs:
			sim_profile = generate_similarity_profile(pair, pid, pids_info, asset_coauths, asset_keywords, asset_info, lname_hist, keyword_hist, journal_hist)
			if sim_profile != -1:
				predict_sim_profiles.append(sim_profile)

	# If no prediction made, remove it from the set
	valid_profiles = []
	valid_unknown_assets = []
	valid_unknown_pids = []
	for i in range(len(predict_sim_profiles)):
		if predict_sim_profiles != -1:
			valid_profiles.append(predict_sim_profiles[i])
			valid_unknown_assets.append(unknown_assets[i])
			valid_unknown_pids.append(unknown_pids[i])

	# sklearn loves numpy
	predict_sim_profiles = np.array(valid_profiles)

	# make predictions
	predictions = rf.predict(predict_sim_profiles)

	to_csv = []
	#if the model is 75% sure then label the asset with the corresponding person_id
	for i in range(len(predictions)):
		if predictions[i] > 0.75: #change this to 0.95 if you want surety
			to_csv.append([valid_unknown_pids[i],valid_unknown_assets[i], predictions[i]])

	print "Writing to CSV..."
	with open("predictions.csv", "wb") as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		for prediction in to_csv:
			writer.writerow(prediction)
	print "Done..."


if __name__ == "__main__":	
	challenge_input_path = os.getcwd()
	data_path = challenge_input_path + "/challenge_input/"


	''' GENERATE RF MODEL FROM KNOWN IDENTITIES '''
	
	asset_coauth_train= read_challenge_file(data_path, "train", "assets_contributors_train.csv")
	asset_info_train = read_challenge_file(data_path, "train", "assets_candidates_train.csv")
	asset_keywords_train = read_challenge_file(data_path, "train", "assets_keywords_train.csv")

	personId_asset_train = read_challenge_file(data_path, "train", "identity_assets_train.csv")
	personId_info_train = read_challenge_file(data_path, "train", "identity_train.csv")

	### This takes too long right now ###
	# tf-idf affliations
	# aff_corpus = [x for x in asset_coauth_train['raw_affiliation'] if x==x]
	# text = [" ".join(tokenize(txt.lower().decode('utf-8'))) for txt in aff_corpus]
	# vectorizer = TfidfVectorizer()
	# matrix = vectorizer.fit_transform(text).todense()
	# matrix = pd.DataFrame(matrix, columns=vectorizer.get_feature_names())
	# top_words = matrix.sum(axis=0).sort_values(ascending=False)
	# print top_words

	# assets linked to namekey
	unknown_idt_asset_hash = hashify_names(asset_coauth_train, "asset_id")
	pkl_fname = challenge_input_path + "/pkl/" + "unknown_idt_asset_hash"
	# fileObject = open(pkl_fname,'wb') 
	# pickle.dump(unknown_idt_asset_hash,fileObject) 
	# fileObject.close()
	print "Pickling..."
	fileObject = open(pkl_fname,'r') 
	unknown_idt_asset_hash = pickle.load(fileObject)  

	# pid linked to asset_id
	personId_asset_hash = hashify(personId_asset_train, "asset", "person_id")

	# asset_id linked to pid
	asset_personId_hash = hashify(personId_asset_train, "person_id", "asset")

	#pid linked to namekey
	targetName_personId_hash = hashify_names(personId_info_train, "person_id")

	#personal info of targets linked to pid
	#personId_info_hash = hashify_lists(personId_info_train, "person_id", ["firstname", "middlename", "lastname", "person_email", "department", "title"])
	pkl_fname = challenge_input_path + "/pkl/" + "personId_info_hash"
	# fileObject = open(pkl_fname,'wb') 
	# pickle.dump(personId_info_hash,fileObject) 
	# fileObject.close()
	print "Pickling..."
	fileObject = open(pkl_fname,'r') 
	personId_info_hash = pickle.load(fileObject) 


	# coauthor info linked to asset_id
	#asset_coauth_hash = hashify_lists(asset_coauth_train, "asset_id", ["firstname", "middlename", "lastname", "email", "raw_affiliation"])
	pkl_fname = challenge_input_path + "/pkl/" + "asset_coauth_hash"
	# fileObject = open(pkl_fname,'wb') 
	# pickle.dump(asset_coauth_hash,fileObject) 
	# fileObject.close()
	print "Pickling..."
	fileObject = open(pkl_fname,'r') 
	asset_coauth_hash = pickle.load(fileObject) 

	# asset info linked to asset_id
	#asset_info_hash = hashify_lists(asset_info_train, "asset_id", ["title", "date_published", "journal_name"]) #not hashing abstracts
	pkl_fname = challenge_input_path + "/pkl/" + "asset_info_hash"
	# fileObject = open(pkl_fname,'wb') 
	# pickle.dump(asset_info_hash,fileObject) 
	# fileObject.close()
	print "Pickling..."
	fileObject = open(pkl_fname,'r') 
	asset_info_hash = pickle.load(fileObject) 

	# asset keywords linked to asset_id
	#asset_keywords_hash = hashify(asset_keywords_train, "asset_id", "asset_keyword")
	pkl_fname = challenge_input_path + "/pkl/" + "asset_keywords_hash"
	# fileObject = open(pkl_fname,'wb') 
	# pickle.dump(asset_keywords_hash,fileObject) 
	# fileObject.close()
	print "Pickling..."
	fileObject = open(pkl_fname,'r') 
	asset_keywords_hash = pickle.load(fileObject) 

	# last name frequencies
	print "Making Lastname Histogram..."
	lname_hist = asset_coauth_train['lastname'].value_counts()

	# keyword frequencies
	print "Making Keyword Histogram..."
	# typically we would strip extra characters but some are important for biomed
	# don't need "tolowercase", data upon inspection looks pretty clean
	keyword_hist = asset_keywords_train['asset_keyword'].value_counts()

	print "Making Journal Histogram..."
	# here we will strip non-alphanumeric chars
	asset_info_train["journal_name"] = asset_info_train['journal_name'].apply(remove_punctuation)
	asset_info_train["journal_name"] = asset_info_train["journal_name"].str.lower()
	journal_hist = asset_info_train['journal_name'].value_counts()


	# Magic
	features, labels = make_features(unknown_idt_asset_hash, personId_asset_hash, asset_personId_hash, 
		targetName_personId_hash, asset_coauth_hash, asset_keywords_hash, asset_info_hash, personId_info_hash,
		lname_hist, keyword_hist, journal_hist)

	print "Generating Random Forest..."
	rf = RandomForestRegressor(n_estimators = 2000, max_features=3, oob_score=True)

	train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25)
	print('Training Features Shape:', train_features.shape)
	print('Training Labels Shape:', train_labels.shape)
	print('Testing Features Shape:', test_features.shape)
	print('Testing Labels Shape:', test_labels.shape)

	print "Fitting Training Data..."
	rf.fit(train_features, train_labels)

	fileObject = open("pkl/rf_model",'wb') 
	pickle.dump(rf,fileObject) 
	fileObject.close()

	predictions = rf.predict(test_features)
	errors = abs(predictions - test_labels)
	print 'Mean Absolute Error: ' + str(np.mean(errors))

	baseline_prediction = np.array(baseline_prediction)
	baseline_errors = abs(baseline_prediction - labels)
	print 'Mean Absolute Baseline Error: ' + str(np.mean(baseline_errors))

	feature_list = ['auth_fst', 'auth_mid', 'auth_lname_idf', 'aff_jacc', 'keyw_share', 'keyw_idf', 'jour_shared_idf', 'jour_date_diff', 'title_jacc', 'email_sim']

	tree = rf.estimators_[5]
	export_graphviz(tree, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
	(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
	graph.write_png('small_tree.png')

	importances = list(rf.feature_importances_)

	print "Gini Importance of Features"
	for i in range(len(feature_list)):
		print feature_list[i] + " \t\t: \t\t\t" + str(importances[i])




	''' PREDICT ASSETS FROM UNKNOWN IDENTITIES '''

	asset_coauth_test= read_challenge_file(data_path, "test", "assets_contributors_test.csv")
	asset_info_test = read_challenge_file(data_path, "test", "assets_candidates_test.csv")
	asset_keywords_test = read_challenge_file(data_path, "test", "assets_keywords_test.csv")

	personId_asset_test = read_challenge_file(data_path, "test", "identity_assets_test.csv")
	personId_info_test = read_challenge_file(data_path, "test", "identity_test.csv")

	# assets linked to namekey
	# unknown_idt_asset_hash = hashify_names(asset_coauth_test, "asset_id")
	pkl_fname = challenge_input_path + "/pkl/" + "unknown_idt_asset_hash_test"
	# fileObject = open(pkl_fname,'wb') 
	# pickle.dump(unknown_idt_asset_hash,fileObject) 
	# fileObject.close()
	print "Pickling..."
	fileObject = open(pkl_fname,'r') 
	unknown_idt_asset_hash = pickle.load(fileObject)  

	# pid linked to asset_id THIS IS THE MANUALLY CURATED SET
	personId_asset_hash = hashify(personId_asset_test, "asset_id", "person_id")

	# asset_id linked to pid
	asset_personId_hash = hashify(personId_asset_test, "person_id", "asset_id")

	#pid linked to namekey
	targetName_personId_hash = hashify_names(personId_info_test, "person_id")

	#personal info of targets linked to pid
	# personId_info_hash = hashify_lists(personId_info_test, "person_id", ["firstname", "middlename", "lastname", "person_email", "department", "title"])
	pkl_fname = challenge_input_path + "/pkl/" + "personId_info_hash_test"
	# fileObject = open(pkl_fname,'wb') 
	# pickle.dump(personId_info_hash,fileObject) 
	# fileObject.close()
	print "Pickling..."
	fileObject = open(pkl_fname,'r') 
	personId_info_hash = pickle.load(fileObject) 


	# coauthor info linked to asset_id
	# asset_coauth_hash = hashify_lists(asset_coauth_test, "asset_id", ["firstname", "middlename", "lastname", "email", "raw_affiliation"])
	pkl_fname = challenge_input_path + "/pkl/" + "asset_coauth_hash_test"
	# fileObject = open(pkl_fname,'wb') 
	# pickle.dump(asset_coauth_hash,fileObject) 
	# fileObject.close()
	print "Pickling..."
	fileObject = open(pkl_fname,'r') 
	asset_coauth_hash = pickle.load(fileObject) 

	# asset info linked to asset_id
	# asset_info_hash = hashify_lists(asset_info_test, "asset_id", ["title", "date_published", "journal_name"]) #not hashing abstracts
	pkl_fname = challenge_input_path + "/pkl/" + "asset_info_hash_test"
	# fileObject = open(pkl_fname,'wb') 
	# pickle.dump(asset_info_hash,fileObject) 
	# fileObject.close()
	print "Pickling..."
	fileObject = open(pkl_fname,'r') 
	asset_info_hash = pickle.load(fileObject) 

	# asset keywords linked to asset_id
	# asset_keywords_hash = hashify(asset_keywords_test, "asset_id", "asset_keyword")
	pkl_fname = challenge_input_path + "/pkl/" + "asset_keywords_hash_test"
	# fileObject = open(pkl_fname,'wb') 
	# pickle.dump(asset_keywords_hash,fileObject) 
	# fileObject.close()
	print "Pickling..."
	fileObject = open(pkl_fname,'r') 
	asset_keywords_hash = pickle.load(fileObject) 

	# last name frequencies
	print "Making Lastname Histogram..."
	lname_hist = asset_coauth_test['lastname'].value_counts()

	# keyword frequencies
	print "Making Keyword Histogram..."
	# typically we would strip extra characters but some are important for biomed
	# don't need "tolowercase", data upon inspection looks pretty clean
	keyword_hist = asset_keywords_test['asset_keyword'].value_counts()

	print "Making Journal Histogram..."
	# here we will strip non-alphanumeric chars
	asset_info_test["journal_name"] = asset_info_test['journal_name'].apply(remove_punctuation)
	asset_info_test["journal_name"] = asset_info_test["journal_name"].str.lower()
	journal_hist = asset_info_test['journal_name'].value_counts()

	print "Pickling Random Forest Model (1GB)"
	pkl_fname = challenge_input_path + "/pkl/" + "rf_model"
	fileObject = open(pkl_fname,'r') 
	rf = pickle.load(fileObject) 

	print "Making Predictions..."
	make_predictions(rf, unknown_idt_asset_hash, personId_asset_hash, asset_personId_hash, targetName_personId_hash, 
	asset_coauth_hash, asset_keywords_hash, asset_info_hash, personId_info_hash, lname_hist, keyword_hist, journal_hist)

	print "Done...."















