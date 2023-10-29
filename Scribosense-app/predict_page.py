# import pandas as pd
# import numpy as np
# from transformers import BertTokenizer
import joblib
import torch
import streamlit as st
import numpy as np
import pandas as pd
import inflect
import re
from nltk.stem import PorterStemmer
import string
from rouge import Rouge
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle
import sentencepiece

import nltk
import textstat
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# from transformers import GPT2Tokenizer, GPT2LMHeadModel



#----------------- Content prediction ----------------------------------------------------------------------------

# def tokenize_content(question,title,text,summary):
#     # Tokenizer for the bert model
#     tokenizer = BertTokenizer.from_pretrained('Models/Tokenizer')
#     tokenized_inputs = tokenizer(
#     question=question,
#     title=title,
#     text=text,
#     summary=summary,
#     padding=True,  # Specify padding here
#     truncation=True,
#     max_length=16,
#     return_tensors='pt'
#     )

#     input_ids = tokenized_inputs['input_ids']
#     attention_mask = tokenized_inputs['attention_mask']

#     return input_ids,attention_mask

# def predict_content(question,title,text,summary):

#     # predict function for content
#     model = joblib.load('Models/bertforcontent.pkl') # load the content model
#     model.eval()

#     input_id,attention_mask=tokenize_content(question,title,text,summary)

#     with torch.no_grad():
#         outputs = model(input_id,attention_mask=attention_mask)
#         predicted_score = outputs.logits.item()

#     return predicted_score

def predict_content(prompt_q,prompt_title,prompt_text,summary_in):
    #open the model
    with open('v2.pickle', 'rb') as f:
        model = pickle.load(f)

    # Regular expression pattern to find numbers in the text
    pattern = r'\d+(\.\d+)?'
    ps = PorterStemmer()
    # nltk.download('punkt',download_dir='../static/punkt')
    inflect_instance = inflect.engine()

    def remove_punctuations(text):
        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')
        return text
    def replace_numbers(match):
        num = match.group()
        num_word = inflect_instance.number_to_words(num)
        return num_word
    def rouge_features(generated_summary, reference_summary):
        rouge = Rouge()
        scores = rouge.get_scores(generated_summary, reference_summary)
        return scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f']
    def sentence_level_metrics(summary):
        # Compute sentence-level metrics
        sentences = summary.split('. ')  # Split into sentences
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        readability_scores = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
        return sentence_lengths, readability_scores
    def content_based_features(generated_summary, reference_summary):
        # Compute TF-IDF cosine similarity
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([generated_summary, reference_summary])
        cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
        return cosine_sim

    # Preprocessing
    def preprocessing(input):
        cols = input.columns
        for col in cols:
            input[col] = input[col].apply(lambda x: " ".join(x.lower() for x in x.split()))
            input[col] = input[col].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
            input[col] = input[col].apply(remove_punctuations)
            input[col] = input[col].apply(lambda x: re.sub(pattern, replace_numbers, x))
            input[col] = input[col].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
        return input

    # Generate summary
    def generate_summary(input):
        # Load pre-trained model and tokenizer
        model_name = "sshleifer/distilbart-cnn-12-6"
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

        generated_summaries = []

        # Loop through each row in the dataset
        for index, row in input.iterrows():
            prompt_text = row['prompt_text']
            prompt_title = row['prompt_title']

            # Combine prompts
            input_text = f"{prompt_text} {prompt_title}"

            # Tokenize input and generate summary
            input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(input_ids)
            generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Append the generated summary to the list
            generated_summaries.append(generated_summary)

        # Add the generated summaries to the dataset as a new column
        input['generated_summary'] = generated_summaries
        return input

    def calculate_features(input):
        # Calculate features for each summary

        # Initialize lists to store feature values
        rouge_1_score = []
        rouge_2_score = []
        sentence_lengths = []
        readability_scores = []
        # word_embeddings = []
        content_sim =[]
        cosine_similarities = []

        # Iterate through each row in the feature_dataset
        for index, row in input.iterrows():
            reference_summary = row['summary']
            generated_summary = row['generated_summary']
            prompt_txt = row['prompt_text']

            # Calculate features using your existing functions
            rouge_score_1,rouge_score_2 = rouge_features(reference_summary, generated_summary)
            sentence_length, readability_score = sentence_level_metrics(reference_summary)
            content_based_feature = content_based_features(reference_summary, generated_summary)

            # Append feature values to lists
            rouge_1_score.append(rouge_score_1)
            rouge_2_score.append(rouge_score_2)
            sentence_lengths.append(sentence_length[0])
            readability_scores.append(readability_score[0])
            content_sim.append(content_based_feature[0][0])

        # Create a new DataFrame with calculated features
        feature_columns = ['rouge_1_score','rouge_2_score', 'sentence_length', 'readability_score','content_sim']
        features_df = pd.DataFrame(zip(rouge_1_score,rouge_2_score, sentence_lengths, readability_scores, content_sim), columns=feature_columns)

        # Concatenate the original dataset and the calculated features
        input.reset_index(drop=True,inplace=True)
        features_df.reset_index(drop=True,inplace=True)
        final_dataset = pd.concat([input, features_df], axis=1)
        return final_dataset

    def get_content_score(input):
        content = model.predict(input)
        return content[0]

    prompt_question = prompt_q
    prompt_title = prompt_title
    prompt_txt = prompt_text
    summary = summary_in
    #summary = "Chapter thirteen discusses the importance of a perfect tragedy in constructing plots and the specific effect of tragedy. A perfect tragedy should be arranged on a complex plan, imitating actions that excite pity and fear. The change of fortune presented should not be the spectacle of a virtuous man brought from prosperity to adversity, as this does not move pity or fear. Instead, the plot should be based on the fortunes of a few houses, such as Alcmaeon, Oedipus, Orestes, Melager, Tyestes, and Telephus.  The best tragedies are founded on the story of a few houses, such as Alcmaeon, Oedipus, Orestes, Melager, Tyestes, and Telephus, who have done or suffered something terrible. The practice of the stage has led to the belief that the best tragedies are those founded on the stories of these houses, as well as those who have done or suffered something terrible.  Euripides, despite his general management of his subject, is considered the most tragic poet due to his well-crafted plays. However, there are other types of tragedies, such as odysseys, which have a double thread of plot and opposite catastrophe for the good and bad. These tragedies are deemed the best due to the weakness of the spectators, as the poet is guided by the wishes of his audience. The true tragic pleasure is proper for comedy, where the deadliest enemies, like Orestes and Aegisthus, leave the stage as friends at the end, and no one is slain or slain."

    # creating a data frame
    data = {'prompt_question': [prompt_question], 'prompt_title': [prompt_title], 'prompt_text': [prompt_txt], 'summary': [summary]}
    input = pd.DataFrame(data)
    preprocessed_input = preprocessing(input)
    generated_summ = pd.read_csv('generated_summary.csv')

    if prompt_title == 'on tragedy' and prompt_question== "summarize at least three elements of an ideal tragedy as described by aristotle":
        item_to_extract = generated_summ.at[0, 'generated_summary']
        preprocessed_input['generated_summary'] = item_to_extract
        input_with_sumary = preprocessed_input
    elif prompt_title == 'egyptian social structure' and prompt_question== "in complete sentences summarize the structure of the ancient egyptian system of government how were different social classes involved in this government cite evidence from the text":
        item_to_extract = generated_summ.at[1, 'generated_summary']
        preprocessed_input['generated_summary'] = item_to_extract
        input_with_sumary = preprocessed_input
    elif prompt_title == 'the third wave' and prompt_question== "summarize how the third wave developed over such a short period of time and why the experiment was ended":
        item_to_extract = generated_summ.at[2, 'generated_summary']
        preprocessed_input['generated_summary'] = item_to_extract
        input_with_sumary = preprocessed_input
    elif prompt_title == 'excerpt from the jungle' and prompt_question== "summarize the various ways the factory would use or cover up spoiled meat cite evidence in your answer":
        item_to_extract = generated_summ.at[3, 'generated_summary']
        preprocessed_input['generated_summary'] = item_to_extract
        input_with_sumary = preprocessed_input
    else:
        input_with_sumary = generate_summary(preprocessed_input)
    feature_vals = calculate_features(input_with_sumary)
    content = get_content_score(feature_vals.drop(columns=['prompt_question','prompt_title','prompt_text','summary','generated_summary']))
    return content



# ------------------------- Wording Prediction --------------------------------------------------------------------------------------

# def load_gpt_model():
#     with torch.no_grad():
#         model = GPT2LMHeadModel.from_pretrained('gpt2')
#         model.eval()
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     return model, tokenizer

# def score_sentence(sentence, model, tokenizer):
#     tokenize_input = tokenizer.encode(sentence)
#     tensor_input = torch.tensor([tokenize_input])
#     loss = model(tensor_input, labels=tensor_input)[0]
#     return np.exp(loss.detach().numpy())

# def predict_wording(summary):


#     # -------------------- load the models -------------------------------------

#     model_data = joblib.load("wording_model_2/wording_regressor_model.pkl")

#     regressor = model_data.get("regressionModel")

#     # Load the GPT model and tokenizer
#     gpt2_model, tokenizer = load_gpt_model()

#     #--------------------------------------------------------------------------------

#     # Get a sentence from user input 
#     user_input = summary

#     # Get the score for the input sentence
#     sentence_score = score_sentence(user_input, gpt2_model, tokenizer)

#     X = np.array([[sentence_score ]])

#     wording  = regressor.predict(X)

#     wording_score = float(wording[0])

#     return wording_score


def predict_wording(prompt_q,prompt_title,prompt_text,summary_in,content_score):
    #open the model
    # with open('wNN_2.pickle', 'rb') as f:
    #     model = pickle.load(f)
    
    

    model = load_model('wNN_3.h5')

    # Preprocessing
    # --------------------------------------------------
    # Regular expression pattern to find numbers in the text
    pattern = r'\d+(\.\d+)?'
    ps = PorterStemmer()
    inflect_instance = inflect.engine()
    def replace_numbers(match):
        num = match.group()
        num_word = inflect_instance.number_to_words(num)
        return num_word
    # ----------------------------------------------------

    def preprocessing(input):

        cols = input.columns
        for col in cols:
            input[col] = input[col].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
            input[col] = input[col].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
        return input

    # calculating metrics
    #----------------------------------------------
    def sentence_level_metrics(summary):
        # Compute sentence-level metrics
        sentences = summary.split('. ')  # Split into sentences
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        readability_scores = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
        return sentence_lengths, readability_scores

    def calculate_clarity_and_readability(summary):
        readability_score = textstat.flesch_reading_ease(summary)
        return readability_score
    def calculate_tense_and_voice_consistency(summary):
        # Define regular expressions for tense and voice patterns.
        past_tense_pattern = r'\b\w+ed\b'  
        present_tense_pattern = r'\b\w+e?s\b'  
        future_tense_pattern = r'will \b\w+\b'  
        passive_voice_pattern = r'\b\w+ been \w+'  

        # Count the occurrences of each pattern in the summary.
        past_tense_count = len(re.findall(past_tense_pattern, summary))
        present_tense_count = len(re.findall(present_tense_pattern, summary))
        future_tense_count = len(re.findall(future_tense_pattern, summary))
        passive_voice_count = len(re.findall(passive_voice_pattern, summary))

        tense_and_voice_score = 0

        tense_and_voice_score = present_tense_count + past_tense_count + future_tense_count + passive_voice_count
        return tense_and_voice_score
    def calculate_grammar_and_spelling(summary):
        blob = TextBlob(summary)
        corrected_text = blob.correct()
        
        # Compare the corrected text to the original to count the number of corrections made.
        error_count = len([1 for orig, corrected in zip(blob.words, corrected_text.words) if orig != corrected])
        return error_count


    #----------------------------------------------

    def calculate_features(input,content_score):
        # Calculate features for each summary

        # Initialize lists to store feature values
        clarity_and_readability = []
        tense_and_voice_consistency = []
        sentence_lengths = []
        readability_scores = []
        grammar_and_spelling =[]
        content = [content_score]

        # Iterate through each row in the feature_dataset
        for index, row in input.iterrows():
            reference_summary = row['summary']

            # Calculate features using your existing functions
            
            sentence_length, readability_score = sentence_level_metrics(reference_summary)
            tense_voice_score = calculate_tense_and_voice_consistency(reference_summary)
            readability = calculate_clarity_and_readability(reference_summary)
            grammar_spell_score = calculate_grammar_and_spelling(reference_summary)

            # Append feature values to lists
            sentence_lengths.append(sentence_length[0])
            readability_scores.append(readability_score[0])
            tense_and_voice_consistency.append(tense_voice_score)
            clarity_and_readability.append(readability)
            grammar_and_spelling.append(grammar_spell_score)

        # Create a new DataFrame with calculated features
        feature_columns = ['content','tense_and_voice_consistency','clarity_and_readability', 'sentence_length', 'grammar_and_spelling']
        features_df = pd.DataFrame(zip(content,tense_and_voice_consistency,clarity_and_readability, sentence_lengths, grammar_and_spelling), columns=feature_columns)

        # Concatenate the original dataset and the calculated features
        input.reset_index(drop=True,inplace=True)
        features_df.reset_index(drop=True,inplace=True)
        final_dataset = pd.concat([input, features_df], axis=1)
        # print(final_dataset.head())
        return final_dataset

    def get_content_score(input):
        content = model.predict(input)
        return content[0]

    prompt_question = prompt_q
    prompt_title = prompt_title
    prompt_txt = prompt_text
    summary = summary_in
    content = content_score
    #summary = "Chapter thirteen discusses the importance of a perfect tragedy in constructing plots and the specific effect of tragedy. A perfect tragedy should be arranged on a complex plan, imitating actions that excite pity and fear. The change of fortune presented should not be the spectacle of a virtuous man brought from prosperity to adversity, as this does not move pity or fear. Instead, the plot should be based on the fortunes of a few houses, such as Alcmaeon, Oedipus, Orestes, Melager, Tyestes, and Telephus.  The best tragedies are founded on the story of a few houses, such as Alcmaeon, Oedipus, Orestes, Melager, Tyestes, and Telephus, who have done or suffered something terrible. The practice of the stage has led to the belief that the best tragedies are those founded on the stories of these houses, as well as those who have done or suffered something terrible.  Euripides, despite his general management of his subject, is considered the most tragic poet due to his well-crafted plays. However, there are other types of tragedies, such as odysseys, which have a double thread of plot and opposite catastrophe for the good and bad. These tragedies are deemed the best due to the weakness of the spectators, as the poet is guided by the wishes of his audience. The true tragic pleasure is proper for comedy, where the deadliest enemies, like Orestes and Aegisthus, leave the stage as friends at the end, and no one is slain or slain."

    # creating a data frame
    data = {'prompt_question': [prompt_question], 'prompt_title': [prompt_title], 'prompt_text': [prompt_txt], 'summary': [summary]}
    input = pd.DataFrame(data)

    preprocessed_input = preprocessing(input)
    feature_vals = calculate_features(preprocessed_input,content)
    wording = get_content_score(feature_vals.drop(columns=['prompt_question','prompt_title','prompt_text','summary']))
    return wording[0]


#-----------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------- Feedback model ----------------------------------------------------------------------------

# def grammar_errors(summary):
#     os.environ["REPLICATE_API_TOKEN"] = "r8_KjNJz1w5dROUo3FRE1s6FJakHD3bNJS2kZ9nu"
#     pre_prompt1 = "List the grammatical errors in the following paragraph:"

#     prompt_input = "Me and him is going to the store to buys some apples. She don't likes apples, but we doesn't care. Them apples are so good, I seen it in a movie. She ain't never ate an apple in her life, ain't that strange? We goes to the store all the times, and it's always fun!"
#     # Generate LLM response
#     output1 = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',
#                             input={"prompt": f"{pre_prompt1}\n{prompt_input}\nAssistant:",
#                                 "temperature": 0.1, "top_p": 0.8, "max_length": 1000, "repetition_penalty": 1})

#     full_response = ""

#     for item in output1:
#         full_response += item


#     # Define a regular expression pattern to match lines containing errors
#     pattern = r'"([^"]+)" should be "([^"]+)"'

#     # Use re.findall to extract the error lines
#     error_lines = re.findall(pattern, full_response)

#     # Create an array to store the extracted errors
#     errors = []
#     for error in error_lines:
#         errors.append(f'"{error[0]}" should be "{error[1]}"')

#     st.write(f"There are {len(errors)} grammatical errors in your summary.")
#     # Print the list of errors with line numbers
#     for i, error in enumerate(errors):
#         st.write(f"{i + 1}. {error}")




#-----------------------------------------------------------------------------------------------------------------------------------


# st.write("Hi")

# summary = "the diffrent social class were like the diffrent part of the pyramid aka the govern if you were in the high class you are at the top of the pyramid lower class your the bottom of the pyramid or the base"

# wording_score = predict_wording(summary)
# st.write(wording_score)



