from GST_Hackathon.gst_eda import load_csv

df_train_x =""
df_train_y =""

def main():
    print("analysis GST data")
    #load training dataset
    global df_train_x ,df_train_y
    df_train_x,df_train_y = load_csv()



if __name__ == "__main__":
    main()