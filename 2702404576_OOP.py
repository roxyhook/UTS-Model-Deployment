import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        
    def create_input_output(self, target_column):
        self.data = self.data.drop(columns=['Booking_ID'], inplace=True)
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)


class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)
        
    def fillingNAWithNumbers(self,columns,number):
        self.x_train[columns].fillna(number, inplace=True)
        self.x_test[columns].fillna(number, inplace=True)

    def createMedianFromColumn(self,col):
        return np.median(self.x_train[col])
    
    def createModeFromColumn(self,col):
        return self.x_train[col].mode()[0]
    
    def checkOutlierWithBox(self,col):
        boxplot = self.x_train.boxplot(column=[col]) 
        plt.show()

    def dataConvertToNumeric(self,columns):
        self.x_train[columns] = pd.to_numeric(self.x_train[columns], errors='coerce')
        self.x_test[columns] = pd.to_numeric(self.x_train[columns], errors='coerce')

    def encodeCat(self, categorical_cols):
        for col in categorical_cols:
            encoder = LabelEncoder()
            self.x_train[col] = encoder.fit_transform(self.x_train[col])
            self.x_test[col] = encoder.transform(self.x_test[col])  

    def encodeLabels(self):
        self.label_encoder = LabelEncoder()
        self.y_train = pd.Series(self.label_encoder.fit_transform(self.y_train))
        self.y_test = pd.Series(self.label_encoder.transform(self.y_test))

    def createModel(self):
        self.model = XGBClassifier() 
    
    def trainModel(self):
        self.model.fit(self.x_train, self.y_train)
    
    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test) 

    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['Cancelled', 'Not_Cancelled']))

    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:  
            pkl.dump(self.model, file)  

    def save_model_to_file(self, filename):
        self.encoder = LabelEncoder()
        with open(filename, 'wb') as file:  
            pkl.dump(self.encoder, file)  

#load data
file_path = 'Dataset_B_hotel.csv'  
data_handler = DataHandler(file_path)
data_handler.load_data()

#split data
data_handler.create_input_output('booking_status')
input_df = data_handler.input_df
output_df = data_handler.output_df
model_handler = ModelHandler(input_df, output_df)
model_handler.split_data()

#preprocess

meal_plan_replace_na = model_handler.createModeFromColumn('type_of_meal_plan')
model_handler.fillingNAWithNumbers('type_of_meal_plan',meal_plan_replace_na)

parking_space_replace_na = model_handler.createModeFromColumn('required_car_parking_space')
model_handler.fillingNAWithNumbers('required_car_parking_space',parking_space_replace_na)
model_handler.dataConvertToNumeric('required_car_parking_space')

model_handler.checkOutlierWithBox('avg_price_per_room')
price_replace_na = model_handler.createModeFromColumn('avg_price_per_room')
model_handler.fillingNAWithNumbers('avg_price_per_room',price_replace_na)

#encode
categorical_cols = ['type_of_meal_plan','room_type_reserved', 'market_segment_type']
model_handler.encodeCat(categorical_cols)

model_handler.encodeLabels()

#modelling
model_handler.trainModel()
model_handler.makePrediction()
model_handler.createReport()

#export model
model_handler.save_model_to_file('trained_model.pkl') 
model_handler.save_encoder_to_file('label_encoder.pkl')

    
    



    
        
    