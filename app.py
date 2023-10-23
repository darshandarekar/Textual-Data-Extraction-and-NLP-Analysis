import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import logging

# Set up logging
logging.basicConfig(filename='log.txt', level=logging.ERROR)

# Set up NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Function to fetch webpage content
def fetch_webpage_content(url):
    header = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=header)
        return response.content
    except Exception as e:
        logging.error(f"Error fetching webpage content for URL: {url}")
        logging.error(str(e))
        return None

# Function to extract title from webpage
def extract_title(soup):
    try:
        title = soup.find('h1').get_text()
        return title
    except Exception as e:
        logging.error("Error extracting title")
        logging.error(str(e))
        return None

# Function to extract text from webpage
def extract_text(soup):
    try:
        article = ""
        for p in soup.find_all('p', class_=lambda value: value != 'tdm-descr'):
            article += p.get_text()+'\n'
        return article
    except Exception as e:
        logging.error("Error extracting text")
        logging.error(str(e))
        return None

# Function to write title and text to file
def write_to_file(file_name, title, article):
    try:
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(title + '\n' + article)
    except Exception as e:
        logging.error(f"Error writing to file: {file_name}")
        logging.error(str(e))

# Function to tokenize text and remove stop words
def tokenize_text(text):
    stop_words = set()
    for files in os.listdir(stopwords_dir):
        with open(os.path.join(stopwords_dir,files),'r',encoding='ISO-8859-1') as f:
            stop_words.update(set(f.read().splitlines()))
    words = word_tokenize(text)
    filtered_text = [word for word in words if word.lower() not in stop_words]
    return filtered_text

# Function to measure text properties
def measure_text(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            text = re.sub(r'[^\w\s.]', '', text)
            sentences = text.split('.')
            num_sentences = len(sentences)
            words = [word for word in text.split() if word.lower() not in stop_words]
            num_words = len(words)
            complex_words = [word for word in words if syllable_count(word) > 2]
            syllable_count_word = sum(syllable_count(word) for word in words)
            avg_sentence_len = num_words / num_sentences
            avg_syllable_word_count = syllable_count_word / len(words)
            percent_complex_words = len(complex_words) / num_words
            fog_index = 0.4 * (avg_sentence_len + percent_complex_words)
            return avg_sentence_len, percent_complex_words, fog_index, len(complex_words), avg_syllable_word_count
    except Exception as e:
        logging.error(f"Error measuring text properties for file: {file}")
        logging.error(str(e))
        return None

# Function to count personal pronouns
def count_personal_pronouns(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            personal_pronouns = ["I", "we", "my", "ours", "us"]
            count = 0
            for pronoun in personal_pronouns:
                count += len(re.findall(r"\b" + pronoun + r"\b", text))
            return count
    except Exception as e:
        logging.error(f"Error counting personal pronouns for file: {file}")
        logging.error(str(e))
        return None

# Function to count syllables in a word
def syllable_count(word):
    if word.endswith('es'):
        word = word[:-2]
    elif word.endswith('ed'):
        word = word[:-2]
    vowels = 'aeiou'
    syllable_count_word = sum(1 for letter in word if letter.lower() in vowels)
    return syllable_count_word

# Function to process a row in the dataframe
def process_row(row):
    url = row['URL']
    url_id = row['URL_ID']
    response_content = fetch_webpage_content(url)
    if response_content is None:
        return

    soup = BeautifulSoup(response_content, 'html.parser')
    title = extract_title(soup)
    if title is None:
        return

    article = extract_text(soup)
    if article is None:
        return

    file_name = f'TitleText/{url_id}.txt'
    write_to_file(file_name, title, article)

    filtered_text = tokenize_text(article)
    docs.append(filtered_text)

    pp_count.append(count_personal_pronouns(file_name))
    avg_sentence_length, percent_complex_words, fog_index, complex_word_count, avg_syllable_word_count = measure_text(file_name)

    avg_sentence_length_lst.append(avg_sentence_length)
    percent_complex_words_lst.append(percent_complex_words)
    fog_index_lst.append(fog_index)
    complex_word_count_lst.append(complex_word_count)
    avg_syllable_word_count_lst.append(avg_syllable_word_count)

# Set up directories
text_dir = "TitleText"
stopwords_dir = "StopWords"
sentiment_dir = "MasterDictionary"

# Load stop words from the stopwords directory
stop_words = set()
for file in os.listdir(stopwords_dir):
    with open(os.path.join(stopwords_dir, file), 'r', encoding='ISO-8859-1') as f:
        stop_words.update(set(f.read().splitlines()))

# Read the input DataFrame
df = pd.read_excel('Input.xlsx')

# Filter out rows with missing URL or URL_ID
df = df.dropna(subset=['URL', 'URL_ID'])

# Initialize lists for storing data
docs = []
pp_count = []
avg_sentence_length_lst = []
percent_complex_words_lst = []
fog_index_lst = []
complex_word_count_lst = []
avg_syllable_word_count_lst = []
average_word_length_lst =[]
word_count_lst =[]

# Process each row in the DataFrame
for index, row in df.iterrows():
    process_row(row)

# Create output DataFrame
output_df = pd.read_excel('Output Data Structure.xlsx')


# Calculate additional metrics for each document
for i, doc in enumerate(docs):
    word_count = len(doc)
    word_count_lst.append(word_count)
    total_word_length = sum(len(word) for word in doc)
    avg_word_length = total_word_length / word_count
    average_word_length_lst.append(avg_word_length)

# Calculate sentiment scores using MasterDictionary
# store positive and negative words from the directory
pos = set()
neg = set()

for file in os.listdir(sentiment_dir):
    if file == 'positive-words.txt':
        with open(os.path.join(sentiment_dir, file), 'r', encoding='ISO-8859-1') as f:
            pos.update(f.read().splitlines())
    else:
        with open(os.path.join(sentiment_dir, file), 'r', encoding='ISO-8859-1') as f:
            neg.update(f.read().splitlines())

# now collect the positive and negative words from each file
# calculate the scores from the positive and negative words
positive_words = []
negative_words = []
positive_score = []
negative_score = []
polarity_score = []
subjectivity_score = []

# iterate through the list of docs
for i in range(len(docs)):
    positive_words.append([word for word in docs[i] if word.lower() in pos])
    negative_words.append([word for word in docs[i] if word.lower() in neg])
    positive_score.append(len(positive_words[i]))
    negative_score.append(len(negative_words[i]))
    polarity_score.append((positive_score[i] - negative_score[i]) / ((positive_score[i] + negative_score[i]) + 0.000001))
    subjectivity_score.append((positive_score[i] + negative_score[i]) / (len(docs[i]) + 0.000001))

variables = [positive_score,
            negative_score,
            polarity_score,
            subjectivity_score,
            avg_sentence_length_lst,
            percent_complex_words_lst,
            fog_index_lst,
            avg_sentence_length_lst,
            complex_word_count_lst,
            word_count_lst,
            avg_syllable_word_count_lst,
            pp_count,
            average_word_length_lst]

output_df.drop([44-37,57-37,144-37], axis = 0, inplace=True)

# write the values to the dataframe
for i, variable in enumerate(variables):
    if len(variable) == len(output_df):
        output_df.iloc[:, i+2] = variable
    else:
        print(f"Length mismatch in variable at index {i}. Expected: {len(output_df)}, Actual: {len(variable)}")


# Save the output DataFrame to a CSV file
output_df.to_csv('Output.csv', index=False)

