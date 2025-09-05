# chatbot_run_free.py
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# =========================================
# 1️⃣ بارگذاری مدل Seq2Seq
# =========================================
model = load_model("models/seq2seq_chatbot.h5")

with open("models/input_tokenizer.json", "r", encoding="utf-8") as f:
    input_tokenizer = tokenizer_from_json(json.load(f))

with open("models/target_tokenizer.json", "r", encoding="utf-8") as f:
    target_tokenizer = tokenizer_from_json(json.load(f))

max_encoder_seq_length = 20
max_decoder_seq_length = 25

# =========================================
# 2️⃣ تابع تولید پاسخ
# =========================================
def generate_response(user_input):
    input_seq = input_tokenizer.texts_to_sequences([user_input])
    encoder_input = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding='post')

    decoder_input = np.zeros((1, max_decoder_seq_length))
    start_token = target_tokenizer.word_index.get('\t', 1)
    decoder_input[0, 0] = start_token

    output_sentence = ''
    previous_words = set()
    for t in range(1, max_decoder_seq_length):
        preds = model.predict([encoder_input, decoder_input], verbose=0)
        next_word_id = np.argmax(preds[0, t-1, :])
        next_word = None
        for word, index in target_tokenizer.word_index.items():
            if index == next_word_id:
                next_word = word
                break
        if next_word in ('\n', None):
            break
        if next_word in previous_words:
            break
        output_sentence += next_word + ' '
        previous_words.add(next_word)
        decoder_input[0, t] = next_word_id

    if not output_sentence.strip():
        output_sentence = "متاسفانه نمی‌توانم پاسخ دهم."
    return output_sentence.strip()

# =========================================
# 3️⃣ محیط تعاملی
# =========================================
print("[INFO] Chatbot آماده است. برای خروج 'exit' تایپ کنید.")
while True:
    user_input = input(">> ").strip()
    if user_input.lower() == "exit":
        break

    response = generate_response(user_input)
    print("Bot:", response)
