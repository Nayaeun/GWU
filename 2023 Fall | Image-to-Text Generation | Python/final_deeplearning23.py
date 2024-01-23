import os
import csv
import pickle
from tqdm import tqdm
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16 , preprocess_input

# Update BASE_DIR and CODE_DIR
image_data_dir = "/home/ubuntu/NLP/home/ubuntu/DeepLearning/"
text_data_dir = "/home/ubuntu/NLP/home/ubuntu/DeepLearning/Flicker8k_text"
code_dir = "/home/ubuntu/NLP/home/ubuntu/DeepLearning/Code"

#%%
# Step 1. Create Image Features Dictionary

# 1-1. Load VGG Model & Restructure model
# base_model = ResNet50()
base_model = VGG16()
res_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

# 1-2. Extract features from images and create a feature dictionary
image_dict = {}
image_directory = os.path.join(image_data_dir, 'Flicker8k_Dataset')

for image_id in tqdm(os.listdir(image_directory)):
    image_path = os.path.join(image_directory, image_id)
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = res_model.predict(image, verbose=0)
    image_dict[image_id] = feature

# 1-3. Store features in pickle
pickle.dump(image_dict, open(os.path.join(code_dir, 'image_dict.pkl'), 'wb'))

# 1-4. Load features from pickle (if needed)
with open(os.path.join(code_dir, 'image_dict.pkl'), 'rb') as features_file:
    image_dict = pickle.load(features_file)


#%%
# Step 2. Create Text Dictionary

# 2-1. Set the path to text file
dataset_path = os.path.join(text_data_dir, 'Flickr8k.token.txt')
text_dict_path = "/home/ubuntu/NLP/home/ubuntu/DeepLearning/Code/text_dict.pkl"

# 2-2. Create a text dictionary
text_dict = {}

with open(dataset_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')

    for row in csv_reader:
        image_id, text = row[0].split('#')[0], row[1]
        image_id = image_id.strip()
        text = text.strip()
        if image_id not in text_dict:
            text_dict[image_id] = [text]
        else:
            text_dict[image_id].append(text)

# 2-3. Save the text dictionary to a file using pickle
with open(text_dict_path, 'wb') as text_dict_file:
    pickle.dump(text_dict, text_dict_file)

# 2-4. Check for missing image IDs in the dictionary
if len(image_dict) != len(text_dict):
    if len(text_dict) > len(image_dict):
        missing_ids = set(text_dict.keys()) - set(image_dict.keys())
        print(f"Removing {len(missing_ids)} image IDs not in image dict: {missing_ids}")
        text_dict = {image_id: text_dict[image_id] for image_id in set(text_dict.keys()) - missing_ids}
    else:
        missing_ids = set(image_dict.keys()) - set(text_dict.keys())
        print(f"Removing {len(missing_ids)} image IDs not in text dict: {missing_ids}")
        image_dict = {image_id: image_dict[image_id] for image_id in set(image_dict.keys()) - missing_ids}

print(f"Image feature dictionary size: {len(image_dict)}")
print(f"Text dictionary size: {len(text_dict)}")
print('Ready for data generation.')

# 2-5. Check the final format of text_dict
for i, (image_id, texts) in enumerate(text_dict.items()):
    print(f"Image ID: {image_id}")
    print(f"Texts: {texts}")
    print("-" * 20)
    if i == 9:
        break

#%%
# Step 3. Preprocess/Tokenize the text in text_dict

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer

# 3-1. Create a text cleaner for preprocessing and tokenization
def clean_text(texts):
    cleaned_texts = []
    # stop_words = set(stopwords.words('english'))
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        words = word_tokenize(text)
        # words = [word for word in words if word not in stop_words]
        cleaned_text = 'startseq ' + ' '.join(words) + ' endseq'
        cleaned_texts.append(cleaned_text)
    return cleaned_texts

# 3-2. Update the text_dict with preprocessed and tokenized text
for key, texts in text_dict.items():
    text_dict[key] = clean_text(texts)

# 3-3. Flatten the list of texts
list_texts = [text for texts in text_dict.values() for text in texts]

# 3-4. Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list_texts)

# 3-5. Vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")

# 3-6. Maximum length of texts
max_length = max(len(text.split()) for text in list_texts)
print(f"Maximum length of texts: {max_length}")

# 3-7. Check the final format of text_dict
for i, (image_id, texts) in enumerate(text_dict.items()):
    print(f"Image ID: {image_id}")
    print(f"Texts: {texts}")
    print("-" * 20)
    if i == 9:
        break


#%%
# Step 4. Split datasets
from sklearn.model_selection import train_test_split
def split_datasets(text_dict, image_dict):

    image_ids = list(text_dict.keys())
    train_image_ids, test_image_ids = train_test_split(image_ids, test_size=0.10, random_state=42)

    return train_image_ids, test_image_ids

train_image_ids, test_image_ids = split_datasets(text_dict, image_dict)

print(f'The number of train images: {len(train_image_ids)}')
print(f'The number of test images: {len(test_image_ids)}')

#%%
# Step 5. Data Generation
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
def data_generator(data_keys, text_dict, image_dict, tokenizer, max_length, vocab_size, batch_size, num_epochs):
    for epoch in range(num_epochs):
        X_images, X_texts, y_text = [], [], []

        for key in data_keys:
            captions = text_dict[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X_images.append(image_dict[key][0])
                    X_texts.append(in_seq)
                    y_text.append(out_seq)

                    if len(X_images) == batch_size:
                        yield [np.array(X_images), np.array(X_texts)], np.array(y_text)
                        X_images, X_texts, y_text = [], [], []

        print(f"Completed Epoch {epoch + 1}/{num_epochs}")

#%%
# Step 6. Model Generation
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.optimizers import Adam

# 6-1. Create a Model
from keras.layers import Concatenate, RepeatVector
def build_model(max_length, vocab_size, embedding_dim=256, lstm_units=256, dropout_rate=0.4):
    # Image feature input
    image_input = Input(shape=(4096,))
    image_dense = Dense(embedding_dim, activation='relu')(image_input)
    image_repeat = RepeatVector(max_length)(image_dense)

    # Text sequence input
    text_input = Input(shape=(max_length,))
    text_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(text_input)
    text_lstm = LSTM(lstm_units, return_sequences=True)(text_embedding)

    # Concatenate image features and text LSTM outputs
    concatenated = Concatenate(axis=-1)([image_repeat, text_lstm])

    # Decoder LSTM
    lstm_out = LSTM(lstm_units, return_sequences=False)(concatenated)
    lstm_out_dropout = Dropout(dropout_rate)(lstm_out)

    # Decoder Dense layers
    decoder_dense1 = Dense(embedding_dim, activation='relu')(lstm_out_dropout)
    outputs = Dense(vocab_size, activation='softmax')(decoder_dense1)

    # Model creation
    model = Model(inputs=[image_input, text_input], outputs=outputs)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # Print the model summary
    model.summary()

    return model


#%%
model = build_model(max_length, vocab_size)

#%%
# 6-2. Train a Model
def train_model(model, train_data, text_dict, image_dict, tokenizer, max_length, vocab_size, batch_size=32, num_epochs=20):
    steps = len(train_data) // batch_size
    for epoch in tqdm(range(num_epochs)):
        generator = data_generator(train_data, text_dict, image_dict, tokenizer, max_length, vocab_size, batch_size, num_epochs)
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

    return model

#%%
trained_model = train_model(model, train_image_ids, text_dict, image_dict, tokenizer, max_length, vocab_size)

#%%
# 6-3. Save a Model
def save_model(model, model_save_path = os.path.join(code_dir, 'final_model.h5')):
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

#%%
save_model(trained_model)

#%%
# Step 7. Text Generation
# Generate text for the Image

# 7-1. Convert the predicted index from the model into a word
def index_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# 7-2. Generate text for an image
def generate_text(model, image, tokenizer, max_length):
    gen_text = 'startseq'

    for _ in range(max_length):
        # Encode the input sequence
        sequence = tokenizer.texts_to_sequences([gen_text])[0]
        # Pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # Predict the next word
        yhat = model.predict([image, sequence], verbose=0)
        # Get the index with the highest probability
        yhat = np.argmax(yhat)
        # Convert the index to word
        word = index_to_word(yhat, tokenizer)

        if word is None:
            break
        gen_text += ' ' + word
        if word == 'endseq':
            break

    return gen_text

#%%
# # Step 8. Prediction
# from nltk.translate.bleu_score import corpus_bleu
#
# def evaluate_model(model, test_data, image_features, tokenizer, max_length):
#     actual, predicted = list(), list()
#
#     for key in tqdm(test_data):
#         captions = test_data[key]
#         image_feature = image_features[key]
#
#         # Actual captions
#         actual_captions = [caption.split() for caption in captions]
#         actual.extend(actual_captions)
#
#         # Predicted caption
#         predicted_caption = generate_text(model, image_feature, tokenizer, max_length).split()
#         predicted.append(predicted_caption)
#
#     # Calculate BLEU scores
#     bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
#     bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
#
#     return bleu1, bleu2
#
# bleu1, bleu2 = evaluate_model(model, text_dict, image_dict, tokenizer, max_length)
# print(f"BLEU-1: {bleu1}, BLEU-2: {bleu2}")

#%%
# Conclusion: Generate Result
from PIL import Image
import matplotlib.pyplot as plt
def generate_result(image_id):

    image_path = os.path.join(image_data_dir, "Flicker8k_Dataset", image_id)
    image = Image.open(image_path)
    texts = text_dict[image_id]
    print('Given Text:')
    for text in texts:
        print(text)
    # predict the caption
    y_pred = generate_text(model, image_dict[image_id], tokenizer, max_length)
    print('Generated Text:')
    print(y_pred)
    plt.imshow(image)
    plt.show()

#%%
generate_result("1001773457_577c3a7d70.jpg")
#%%
generate_result("1022454428_b6b660a67b.jpg")
#%%
generate_result("102351840_323e3de834.jpg")
#%%
generate_result("225699652_53f6fb33cd.jpg")
#%%
generate_result("3684562647_28dc325522.jpg")
