import streamlit as st

import cv2
import plotly.express as px
import tensorflow as tf
from urllib.request import urlretrieve
from pathlib import Path
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from transformers import TFBertModel
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy

model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def parse_line(line):
    data, intent_label = line.split(" <=> ")
    items = data.split()
    words = [item.rsplit(":", 1)[0]for item in items]
    word_labels = [item.rsplit(":", 1)[1]for item in items]
    return {
        "intent_label": intent_label, 
        "words": " ".join(words),
        "word_labels": " ".join(word_labels),
        "length": len(words),
    }

def encode_dataset(text_sequences):
    # Create token_ids array (initialized to all zeros), where 
    # rows are a sequence and columns are encoding ids
    # of each token in given sequence.
    token_ids = np.zeros(shape=(len(text_sequences), max_token_len),
                         dtype=np.int32)
    
    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence)
        token_ids[i, 0:len(encoded)] = encoded

    attention_masks = (token_ids != 0).astype(np.int32)
    return {"input_ids": token_ids, "attention_masks": attention_masks}

train_lines = Path("train.txt").read_text().strip().splitlines()
valid_lines = Path("valid.txt").read_text().strip().splitlines()
test_lines = Path("test.txt").read_text().strip().splitlines()

df_train = pd.DataFrame([parse_line(line) for line in train_lines])
df_valid = pd.DataFrame([parse_line(line) for line in valid_lines])
df_test = pd.DataFrame([parse_line(line) for line in test_lines])

max_token_len = 43

encoded_train = encode_dataset(df_train["words"])
encoded_valid = encode_dataset(df_valid["words"])
encoded_test = encode_dataset(df_test["words"])

intent_names = Path("vocab_intent.txt").read_text().split()
intent_map = dict((label, idx) for idx, label in enumerate(intent_names))
intent_train = df_train["intent_label"].map(intent_map).values
intent_valid = df_valid["intent_label"].map(intent_map).values
intent_test = df_test["intent_label"].map(intent_map).values

base_bert_model = TFBertModel.from_pretrained("bert-base-cased")


# Build a map from slot name to a unique id.
slot_names = ["[PAD]"] + Path("vocab_slot.txt").read_text().strip().splitlines()
slot_map = {}
for label in slot_names:
    slot_map[label] = len(slot_map)

# Uses the slot_map of slot name to unique id, defined above, as well
# as the BERT tokenizer, to create a np array with each row corresponding
# to a given sequence, and the columns as the id of the given token slot labels.
def encode_token_labels(text_sequences, true_word_labels):
    encoded = np.zeros(shape=(len(text_sequences), max_token_len), dtype=np.int32)
    
    # Loop through sequence
    for i, (text_sequence, word_labels) in enumerate( \
            zip(text_sequences, true_word_labels)):
        encoded_labels = []
        
        # Loop through word in sequence
        for word, word_label in zip(text_sequence.split(), word_labels.split()):
            tokens = tokenizer.tokenize(word)
            encoded_labels.append(slot_map[word_label])
            expand_label = word_label.replace("B-", "I-")
            if not expand_label in slot_map:
                expand_label = word_label
            encoded_labels.extend([slot_map[expand_label]] * (len(tokens) - 1))
        encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
    return encoded

#from mrcnn import config as config_std
# Define the class for the model that will create predictions
# for the overall intent of a sequence, as well as the NER token labels.
class JointIntentAndSlotFillingModel(tf.keras.Model):
    def __init__(self, intent_num_labels=None, slot_num_labels=None,
                dropout_prob=0.4):
        super().__init__(name="joint_intent_slot")

        self.bert = base_bert_model
        
        self.dropout = Dropout(dropout_prob)
        self.intent_classifier = Dense(intent_num_labels,
                                       name="intent_classifier")
        self.slot_classifier = Dense(slot_num_labels,
                                     name="slot_classifier")

    def call(self, inputs, **kwargs):
        tokens_output, pooled_output = self.bert(inputs, **kwargs, return_dict=False)

        tokens_output = self.dropout(tokens_output, training=kwargs.get("training", False))
        slot_logits = self.slot_classifier(tokens_output)

        pooled_output = self.dropout(pooled_output, training=kwargs.get("training", False))
        intent_logits = self.intent_classifier(pooled_output)

        return slot_logits, intent_logits

joint_model = JointIntentAndSlotFillingModel(7, 73)
joint_model.load_weights('weights')


def show_predictions(text, intent_names, slot_names):
    inputs = tf.constant(tokenizer.encode(text))[None, :]  # batch_size = 1
    outputs = joint_model(inputs)
    slot_logits, intent_logits = outputs
    slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]
    intent_id = intent_logits.numpy().argmax(axis=-1)[0]
    st.text("## Intent:" + str(intent_names[intent_id]))
    st.text("## Slots:")
    for token, slot_id in zip(tokenizer.tokenize(text), slot_ids):
        st.text("   " + f"{token:>10} : {slot_names[slot_id]}")


options = ["Demonstration", "Data", "Sequence Intents", "Slot Intents"]
selectbox = st.sidebar.selectbox(
    "What would you like to see?",
    (option for option in options)
)

if selectbox == options[0]:
  selectbox2 = None
  st.title("BERT Demonstration")
  st.header("Type your input below.")
  user_input = st.text_input("", "")
  if user_input != "":
    show_predictions(user_input, intent_names, slot_names)
  
  st.header("**What you can ask me:**")
  st.subheader("Add to a playlist")
  #st.text("Add ___ to the playlist ___.")
  st.subheader("Book a restaurant")
  #st.text("Book a table for tonight at ___.")
  st.subheader("Play music")
  #st.text("Play ___. ")
  st.subheader("Rate a book")
  #st.text("Rate ___ a zero out of ten.")
  st.subheader("Get weather")
  #st.text("What's the weather in ___.")
  st.subheader("Search a creative work")
  #st.text("Show me ___.")
  st.subheader("Search a screening event")
  #st.text("Find the closest showing of ___.")
elif selectbox == options[1]:
  options2 = ["Training", "Validation", "Testing"]
  selectbox2 = st.sidebar.selectbox(
    "Which data set would you like to see?",
    (option for option in options2)
  )
  if selectbox2 == options2[0]:
    st.title("**Training Data:**")
    st.dataframe(df_train)

  elif selectbox2 == options2[1]:
    st.title("**Validation Data:**")
    st.dataframe(df_valid)

  elif selectbox2 == options2[2]:
    st.title("**Testing Data:**")
    st.dataframe(df_test)
elif selectbox == options[2]:
  st.title("**Sequence Intents:**")
  st.dataframe(pd.DataFrame(intent_names))
elif selectbox == options[3]:
  st.title("**Slot Intents:**")
  st.dataframe(pd.DataFrame([intent for intent in slot_map]), width = 600)