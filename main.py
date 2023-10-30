import telebot
from io import BytesIO
import numpy as np
import pandas as pd
import sklearn
import warnings
import librosa
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from scipy.fftpack import fft


warnings.filterwarnings('ignore')
# Our API key for telegram bot
TOKEN = '5681427786:AAFnS-Fzgo5cuw5f5qUpEv2nK9Uf84mrTVw'
bot = telebot.TeleBot(TOKEN)

# Downloading models for gender and age detection
model_gender = joblib.load('model_gender.sav')
model_age = joblib.load('model_age.sav')

# Downloading scaler that were used during training of our models
scaler_age = joblib.load('scaler_age.joblib')
scaler_gender = joblib.load('scaler_gender.joblib')

# Tool to selecting best features (Anova analysis)
bestf = joblib.load('bestf_gender.joblib')

# Classes to predictions from model can be applied
class_age = ['fifties', 'fourties', 'seventies', 'sixties', 'teens','thirties', 'twenties']
class_gender = ['female', 'male']

# Starting bot with command /start
@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Welcome! This bot can detect gender and age group by voice. If you want to see it for yourself, just send me your voice message!")


# Help menu /help
@bot.message_handler(commands=['help'])
def help(message):
    bot.send_message(message.chat.id, """
    If the prediction of the bot is incorrect, it doesn't mean that your voice is different; it just means that the model sometimes can make mistakes.
    """)

# If anything is messaged outside of command above or voice message then this message will appear
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.send_message(message.chat.id, "Send me voice message if you want me to guess your gender and age group")

# Helper function to analyze audio and prepare features
def analyze_audio_age(audio_file):
    features = list()
    sampling_rate = 48000
    audio, _ = librosa.load(audio_file, sr=sampling_rate)

    # Fast Fourier Transform
    audio_fft = np.abs(fft(audio))

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sampling_rate))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate))
    features.append(5)
    features.append(spectral_centroid)
    features.append(spectral_bandwidth)
    features.append(spectral_rolloff)

    # Using FFT to extract MFCC features
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(audio_fft.reshape((-1, 1))), sr=sampling_rate)
    for el in mfcc:
        features.append(np.mean(el))

    return features

@bot.message_handler(content_types=['voice'])
def handle_audio_gender(message):
    # Getting voice data from voice message
    file_id = message.voice.file_id
    file_info = bot.get_file(file_id)
    file_bytes = bot.download_file(file_info.file_path)
    file_buffer = BytesIO(file_bytes)

    # Analyzing audio
    features = analyze_audio_age(file_buffer)
    for_age = features

    # Preparing features for gender model
    features = features[1:]
    features = scaler_gender.transform(np.array(features).reshape(1, -1))
    features = bestf.transform(features)

    # Predicting gender
    genderr = model_gender.predict(features)
    genderr = int(genderr[0])

    # Preparing feautres for age model
    for_age[0] = genderr
    for_age = scaler_age.transform(np.array(for_age).reshape(1, -1))

    # Predictiing age
    agee = model_age.predict(for_age)

    # Sending back predictions in message
    bot.send_message(message.chat.id, f"My guess is you are - {class_gender[genderr]} and your age group is "
                                      f"{class_age[agee[0]-1]}")

bot.polling()