import pandas as pd 
import numpy as np 
import os 

random_state = 3535 

linzen_gulordava_testset_dir = "./linzen_testset_files"
# See https://github.com/facebookresearch/colorlessgreenRNNs/tree/main/data/linzen_testset for file descriptions

sentences = pd.read_table(os.path.join(linzen_gulordava_testset_dir, "subj_agr_filtered.text"), header=None, names=["sentence"])
agr_information = pd.read_table(os.path.join(linzen_gulordava_testset_dir, "subj_agr_filtered.gold"), header=None, names=["target_idx", "correct_form", "wrong_form", "n_attractors"])
test_data_raw = pd.concat([agr_information, sentences], axis=1)

test_data_processed = test_data_raw.copy()

def insert_form(row, form):
    sentence = row['sentence']
    target_idx = row['target_idx']
    correct_form = row[f'{form}_form']
    sentence = sentence.split()
    sentence[target_idx] = correct_form
    return " ".join(sentence)

test_data_processed['sentence_grammatical'] = test_data_processed.apply(lambda row: insert_form(row, 'correct'), axis=1) # Safe to assume that sentences already have grammatical form, but harm running this just in case
test_data_processed['sentence_ungrammatical'] = test_data_processed.apply(lambda row: insert_form(row, 'wrong'), axis=1) # Replaces correct forms with incorrect forms in the sentences

# Next, clean the sentences slightly, removing extra spaces and <eos> tokens:
def clean_sentence(sentence):
    sentence_clean = sentence.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" 's", "'s").replace(" n't", "n't").replace(" 'm", "'m").replace(" 're", "'re").replace(" 've", "'ve").replace(" 'll", "'ll").replace(" 'd", "'d").replace("`` ", "\"").replace(" ''", "\"").replace("( ", "(").replace(" )", ")").replace(" ;", ";").replace(" :", ":").replace(" /", "/").replace(" '", "'").replace(" - ", "-").replace(" -", "-").replace("- ", "-")
    sentence_clean = sentence_clean.replace(" <eos>", "")
    return sentence_clean

test_data_processed['sentence_grammatical'] = test_data_processed['sentence_grammatical'].apply(lambda x: clean_sentence(x))
test_data_processed['sentence_grammatical'] = test_data_processed['sentence_grammatical'].apply(lambda x: x.capitalize()) # Capitalize the first letter of the sentence
test_data_processed['sentence_ungrammatical'] = test_data_processed['sentence_ungrammatical'].apply(lambda x: clean_sentence(x))
test_data_processed['sentence_ungrammatical'] = test_data_processed['sentence_ungrammatical'].apply(lambda x: x.capitalize()) 

# Add sentence length annotation: will likely be useful for analysis later:
test_data_processed['sentence_length'] = test_data_processed['sentence_grammatical'].apply(lambda x: len(x.split())) # Grammatical v ungrammatical sentences have same length, since it's only the verb form that changes

# Stratified sampling, by n_attractors, while controlling for sentence length:
base_df = pd.DataFrame()
for i in np.unique(test_data_processed['n_attractors']):
    n_attractors_indexed = test_data_processed[test_data_processed['n_attractors'] == i]
    length_controlled = n_attractors_indexed[n_attractors_indexed['sentence_length'].apply(lambda x: 15 <= x <= 25)] # Get sentences with length between 15 and 25 -- this is a comfortable range to get sentences from all `n_attractors` values
    sample = length_controlled.sample(30, random_state=random_state)
    base_df = pd.concat([base_df, sample], axis=0)

base_df = base_df.drop(columns=["sentence"])
base_df['source_idx'] = base_df.index
base_df = base_df.reset_index(drop=True)
base_df.to_csv("linzen_gulordava_testset_sample.csv", index=False)



