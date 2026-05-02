import pandas  as pd
import matplotlib.pyplot as plt
import os


def  load_csv() :

    #get the current directory of the script
    script_dir = os.path.dirname(__file__)

    #construct the relative path
    train_x_file_path = os.path.join(script_dir,'resources/train','X_Train_Data_Input.csv')
    train_y_file_path = os.path.join(script_dir, 'resources/train', 'Y_Train_Data_Target.csv')

    train_x = pd.read_csv(train_x_file_path)
    train_y = pd.read_csv( train_y_file_path)
    return train_x,train_y
