import numpy as np
import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from snowballstemmer import TurkishStemmer
from wordcloud import WordCloud


try:
    data = pd.read_excel('Turkish Bulleying Tweet Dataset.xlsx')
    data['labels'] = pd.factorize(data.label)[0]
    data.drop([0], axis=0, inplace=True)

except Exception as e:
    print("An error occurred while loading the Excel file:", str(e))
stemmer = TurkishStemmer()
data = data.dropna()
# Metin önişleme fonksiyonu
stop_words = set(stopwords.words("turkish"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub("[^abcçdefgğhıijklmnoöprsştuüvyz]", " ", text)
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    stemmed_text = []
    for word in text:
        stemmed_word = stemmer.stemWord(word)
        stemmed_text.append(stemmed_word)
    return stemmed_text

data['preprocessed_text'] = data.text.apply(preprocess_text)


#%%veri görselleştirme
# Etiketlerin dağılımını gösteren çubuk grafik
label_counts = data['labels'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.xlabel('Etiketler')
plt.ylabel('Sayı')
plt.title('Etiketlerin Dağılımı')
plt.show()


word_freq = pd.Series(' '.join(data['preprocessed_text'].apply(' '.join)).split()).value_counts()

# En çok geçen 20 kelimeyi seçme
top_20_words = word_freq.head(20)

# Plot oluşturma
plt.figure(figsize=(10, 6))
sns.barplot(x=top_20_words.index, y=top_20_words.values)
plt.xlabel('Kelime')
plt.ylabel('Frekans')
plt.title('En Çok Geçen 20 Kelime')
plt.xticks(rotation=45)
plt.show()



# Veri setindeki metinleri birleştirme
text_concatenated = ' '.join(data['preprocessed_text'].apply(' '.join))

# WordCloud nesnesini oluşturma
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_concatenated)

# Kelime bulutunu görselleştirme
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Kelime Bulutu')
plt.show()

# 4 sınıf için ayrı ayrı histogramları çizme
text_lengths = data['preprocessed_text'].apply(len)
plt.figure(figsize=(10, 6))
plt.hist(text_lengths[data['labels'] == 0], bins=20, color='red', alpha=0.5, label='Irkçılık')
plt.hist(text_lengths[data['labels'] == 1], bins=20, color='blue', alpha=0.5, label='Nötr')
plt.hist(text_lengths[data['labels'] == 2], bins=20, color='green', alpha=0.5, label='Cinsiyetçilik')
plt.hist(text_lengths[data['labels'] == 3], bins=20, color='yellow', alpha=0.5, label='Kızdırma')
plt.xlabel('Metin Uzunluğu')
plt.ylabel('Frekans')
plt.title('Metin Uzunluğunun Dağılımı')
plt.legend()
plt.show()



#%% Frekans tabanlı metin vektörü oluşturma
text_list = data['preprocessed_text'].apply(' '.join).tolist()

count_vectorizer = CountVectorizer(max_features=1000)
count_matrix = count_vectorizer.fit_transform(text_list).toarray()

X_freq = count_matrix
y = data.labels.values

X_freq_train, X_freq_test, y_train, y_test = train_test_split(X_freq, y, test_size=0.3, random_state=42)

#%% Random Forest sınıflandırıcısı ile Frekans tabanlı
rf_freq_classifier = RandomForestClassifier(n_estimators=50, random_state=1)
rf_freq_classifier.fit(X_freq_train, y_train)
rf_freq_pred_freq = rf_freq_classifier.predict(X_freq_test)
rf_freq_accuracy = accuracy_score(y_test, rf_freq_pred_freq)
rf_freq_cm = confusion_matrix(y_test, rf_freq_pred_freq)
print("Frekans tabanli vektorlesme ile Random forest dogruluk degeri", rf_freq_accuracy)
print("Frekans tabanli vektorlesme ile Random forest icin confusion matrix ")
print(rf_freq_cm)

# Confusion matrix görselleştirme - Random Forest with Frequency-based Vectorization
plt.figure(figsize=(8, 6))
sns.heatmap(rf_freq_cm, annot=True, cmap="Blues", fmt="d", xticklabels=['Irkçılık', 'Nötr', 'Kızdırma', 'Cinsiyetçilik'], yticklabels=['Irkçılık', 'Nötr', 'Kızdırma', 'Cinsiyetçilik'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Frekans tabanli vektorlesme ile Random forest icin confusion matrix ')

#%% Naive Bayes sınıflandırıcısı ile Frekans tabanlı
nb_freq_classifier = MultinomialNB()
nb_freq_classifier.fit(X_freq_train, y_train)
nb_freq_pred_freq = nb_freq_classifier.predict(X_freq_test)
nb_freq_accuracy = accuracy_score(y_test, nb_freq_pred_freq)
nb_freq_cm = confusion_matrix(y_test, nb_freq_pred_freq)
print("Frekans tabanli vektorlesme ile Naive Bayes dogruluk degeri", nb_freq_accuracy)
print("Frekans tabanli vektorlesme ile Naive Bayes icin confusion matrix ")
print(nb_freq_cm)

# Confusion matrix görselleştirme - Naive Bayes with Frequency-based Vectorization
plt.figure(figsize=(8, 6))
sns.heatmap(nb_freq_cm, annot=True, cmap="Blues", fmt="d", xticklabels=['Irkçılık', 'Nötr', 'Kızdırma', 'Cinsiyetçilik'], yticklabels=['Irkçılık', 'Nötr', 'Kızdırma', 'Cinsiyetçilik'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Frekans tabanli vektorlesme ile Naive Bayes icin confusion matrix ')

#%% Decision Tree sınıflandırıcısı ile Frekans tabanlı
dt_freq_classifier = DecisionTreeClassifier(random_state=1)
dt_freq_classifier.fit(X_freq_train, y_train)
dt_freq_pred_freq = dt_freq_classifier.predict(X_freq_test)
dt_freq_accuracy = accuracy_score(y_test, dt_freq_pred_freq)
dt_freq_cm = confusion_matrix(y_test, dt_freq_pred_freq)
print("Frekans tabanli vektorlesme ile Decision Tree dogruluk degeri", dt_freq_accuracy)
print("Frekans tabanli vektorlesme ile Decision Tree icin confusion matrix ")
print(dt_freq_cm)

# Confusion matrix görselleştirme - Decision Tree with Frequency-based Vectorization
plt.figure(figsize=(8, 6))
sns.heatmap(dt_freq_cm, annot=True, cmap="Blues", fmt="d", xticklabels=['Irkçılık', 'Nötr', 'Kızdırma', 'Cinsiyetçilik'], yticklabels=['Irkçılık', 'Nötr', 'Kızdırma', 'Cinsiyetçilik'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Frekans tabanli vektorlesme ile Decision Tree icin confusion matrix ')
#%% Word2Vec yöntemi ile metin vektörü oluşturma
word_list = data['preprocessed_text'].tolist()

w2v_model = Word2Vec(sentences=word_list, vector_size=100, window=5, min_count=1, workers=4)

def get_text_vector(word_list):
    vector = np.zeros(100)
    count = 0
    for word in word_list:
        if word in w2v_model.wv.key_to_index:
            vector += w2v_model.wv.get_vector(word)
            count += 1
    if count != 0:
        vector /= count
    return vector

w2v_matrix = np.array([get_text_vector(words) for words in data['preprocessed_text']])

#%% Train-test bölmesi
X_w2v = w2v_matrix
y = data.labels.values

X_w2v_train, X_w2v_test, y_train, y_test = train_test_split(X_w2v, y, test_size=0.3, random_state=42)

#%% Random Forest sınıflandırıcı ile Word2Vec
rf_w2v_classifier = RandomForestClassifier(n_estimators=50, random_state=1)
rf_w2v_classifier.fit(X_w2v_train, y_train)
rf_w2v_pred_w2v = rf_w2v_classifier.predict(X_w2v_test)
rf_w2v_accuracy = accuracy_score(y_test, rf_w2v_pred_w2v)
rf_w2v_cm = confusion_matrix(y_test, rf_w2v_pred_w2v)
print("Random Forest with Word2Vec accuracy:", rf_w2v_accuracy)
print("Confusion Matrix for Random Forest with Word2Vec:")
print(rf_w2v_cm)

#%% Confusion matrix görselleştirme - Random Forest with Word2Vec
plt.figure(figsize=(8, 6))
sns.heatmap(rf_w2v_cm, annot=True, cmap="Blues", fmt="d", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Random Forest with Word2Vec')
plt.show()

#%% Kullanıcının tweetini tahmin etme
def predict_category(tweet):
    preprocessed_tweet = preprocess_text(tweet)
    tweet_vector = count_vectorizer.transform([' '.join(preprocessed_tweet)]).toarray()
    category_index = nb_freq_classifier.predict(tweet_vector)[0]
    categories = ['Irkçılık', 'Kızdırma', 'Nötr', 'Cinsiyetçilik']
    predicted_category = categories[category_index]
    return predicted_category

# Kullanıcıdan tweet girdisi alma ve sürekli sorma
while True:
    user_tweet = input("Lütfen tweetinizi girin (Çıkmak için 'q' girin): ")
    if user_tweet.lower() == 'q':
        break
    predicted_category = predict_category(user_tweet)
    print("Tweetin Kategorisi:", predicted_category)