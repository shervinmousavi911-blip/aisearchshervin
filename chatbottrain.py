# chatbot_train.py
import mysql.connector
import json
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================
# 1️⃣ اتصال به دیتابیس و گرفتن داده‌ها
# =========================================
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # اگر پسورد داری وارد کن
    database="people"
)
cursor = db.cursor(dictionary=True)
cursor.execute("SELECT id, namefull, years FROM persons")
data = cursor.fetchall()
print(f"[INFO] {len(data)} رکورد از دیتابیس دریافت شد.")

# =========================================
# 2️⃣ آماده‌سازی داده‌ها برای seq2seq
# =========================================
input_texts = []
target_texts = []

for row in data:
    # ورودی خام: رکورد دیتابیس
    input_text = f"{row['namefull']} {row['years']}"
    target_text = f"{row['namefull']} {row['years']} سال دارد."
    # اضافه کردن کاراکتر شروع و پایان برای seq2seq
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)

print(f"[INFO] {len(input_texts)} نمونه برای آموزش آماده شد.")

# =========================================
# 3️⃣ Tokenizer
# =========================================
input_tokenizer = Tokenizer(char_level=False, oov_token='<OOV>')
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
max_encoder_seq_length = max([len(seq) for seq in input_sequences])
num_encoder_tokens = len(input_tokenizer.word_index) + 1
encoder_input_data = pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')

target_tokenizer = Tokenizer(char_level=False, oov_token='<OOV>')
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
max_decoder_seq_length = max([len(seq) for seq in target_sequences])
num_decoder_tokens = len(target_tokenizer.word_index) + 1
decoder_input_data = pad_sequences(target_sequences, maxlen=max_decoder_seq_length, padding='post')

# decoder_output_data باید one-hot باشد
decoder_output_data = np.zeros((len(target_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
for i, seq in enumerate(target_sequences):
    for t, word_id in enumerate(seq[1:]):
        decoder_output_data[i, t, word_id] = 1.0

print("[INFO] داده‌ها آماده seq2seq شدند.")

# =========================================
# 4️⃣ ساخت مدل Seq2Seq
# =========================================
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(num_encoder_tokens, latent_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(num_decoder_tokens, latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# مدل seq2seq نهایی
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =========================================
# 5️⃣ آموزش مدل
# =========================================
epochs = 300
batch_size = 16

model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_output_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)

# =========================================
# 6️⃣ ذخیره مدل و Tokenizer
# =========================================
if not os.path.exists("models"):
    os.makedirs("models")

model.save("models/seq2seq_chatbot.h5")

with open("models/input_tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(input_tokenizer.to_json(), f, ensure_ascii=False, indent=4)

with open("models/target_tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(target_tokenizer.to_json(), f, ensure_ascii=False, indent=4)

print("[INFO] مدل Seq2Seq و Tokenizer ذخیره شدند. آماده runtime.")
