from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split



tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('newmodel.pkl', 'rb'))

dataframe = pd.read_csv('cleaned_dataset.csv')
x = dataframe['statement']
y = dataframe['rating']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def predict_classes(news_data):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_data = [news_data]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction