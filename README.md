# Disambiguation Challenge

A frequent and non trivial issue in the academic world is *Author Disambiguation*, which is the ability to find all publications belonging to a given author and distinguish them from publications of other authors who share the same name.

Despite being an intricate challenge, fortunately there are ways to overcome the problem with the help of data science (: .

The challenge provided to you is a small-scale version of this problem where you are given just a tiny sample of the data out there used to disambiguate.

Below you can find the files description, the objectives and some tips that might help you in navigating this little data journey.

### Files Description
#### Training Files
- **identity_train.csv**: people for whom full set of academic accomplishment (i.e. assets) is provided (see identity_assets_train.csv).
- **identity_assets_train.csv**: multimap between identity ids and assets ids representing the full set of academic accomplishments belonging to the people in *identity_train.csv*.
- **identity_institutions_train.csv**: multimap between identities in *identity_train.csv* and the institutions linked to them.
- **identity_keywords_train.csv**: multimap between identities in *identity_train.csv* and the keywords linked to them.
- **assets_candidates_train.csv**: academic accomplishments that potentially belong to the people in *identity_train.csv*.
- **assets_contributors_train.csv**: people associated with a given academic accomplishment (i.e. asset) in *assets_candidates_train.csv*.
- **assets_keywords_train.csv**: keywords associated with a academic accomplishments in *assets_candidates_train.csv*.


#### Test Files
- **identity_test.csv**: people for whom academic accomplishments need to be predicted.
- **identity_institutions_test.csv**: multimap between identities in *identity_test.csv* and the institutions linked to them.
- **identity_keywords_test.csv**: multimap between identities in *identity_test.csv* and the keywords linked to them.
- **assets_candidates_test.csv**: academic accomplishments that potentially belong to the people in *identity_test.csv*.
- **assets_contributors_test.csv**: people associated with a given academic accomplishment (i.e. asset) in *assets_candidates_test.csv*.
- **assets_keywords_test.csv**: keywords associated with a academic accomplishments in *assets_candidates_test.csv*.



### Objectives:

**1.** Build features and a model that are able to predict which assets of *assets_candidates_test.csv* belong to the identities in *identity_test.csv*

**2.** Provide the code implemented for point **1** above.

**3.** Provide in a write up, what you did, your reasonings, findings, what you would do if you had a lot more time to work on this and any assumptions you have made.                                                                                
                                                                                                                       
### Things to keep in mind: 
**1.** Just because an article has a person with the same name, doesn't necessarily make it a match! Multiple people can have the same name (just take a look at how many cases there are in *assets_candidates_train.csv* that are not in *identity_assets_train.csv*). The identities provided do not represent the full universe of people, thus even if you don't see in the identity table two people having the same in *identity_assets_train.csv* don't make the assumption that you match by name is enough for proper linkage.

**2.**  The email values in the files look funny. Don't worry, they are not corrupted: emails were obfuscated using hashing.

**3.** Names look funny too! But don't worry: even if you haven't ever seen them, you treat them as real names for purposes of the challenge. 

**4.** Wherever you see a group of words separated by commas, it means that the text has been cleaned.

**5.** If the data is too big for the computer you are using to do the challenge, you can use a subset of rows for each file (just make sure that the subsets are selected properly so that you can JOIN the tables without too much loss of information) and don't worry if you can't produce final results because of it.

**6.** It's OK if you are not able to complete the challenge! Do what you can/want and document what would you do if you were given more time.

**7.** If you have doubts, make plausible assumptions and document them.  

**8.** You can use any language, but Python or Scala are strongly preferred.

**9.** Clean code (e.g. use of functions, explanatory comments etc.) is very nice to have and will earn you "points".

**10.** A preferrable paradigm is quality over quantity: building a few advanced feautures that you think will have high predictive power is better than many naive ones.

**11.** If you have any additional files relative to the challenge that you want to share because will help with the evaluation(e.g. Jupyter Notebook with some EDA or findings), feel free to do it.

**12.** The challenge should be completed alone, but you can consult your friend the Internet and your notes.