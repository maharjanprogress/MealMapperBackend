import os

class Config:
    SQLALCHEMY_DATABASE_URI = "postgresql://postgres:1234567890@localhost/MealMapper"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'FoodImage' # Define the upload folder for food images
