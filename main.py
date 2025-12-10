import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib as job

# randomState seed 
SEED = 14

# extracting the data and adding a column engagement rate -> (likes + shares + comments)/views 
# also cleaning the data and rearranging to hot encodings for specific data -> category and upload_hour
def open_data() -> pd.DataFrame :
    FILENAME = "data/youtube_shorts_extended.csv"
    try : 
          df = pd.read_csv(
              FILENAME,
              sep= ",",
              header= 0,
              usecols=["duration_sec","hashtags_count","upload_hour","category","likes", "views", "shares", "comments" ]
              ) 
          try:
            df.insert(0, "engagement_rate", None)
            # our target value is the engagement rate so we need to encode it into 0 | 1 after the calculations of the engagement rate
            df["engagement_rate"] = (df["likes"] + df["comments"] + df["shares"])/df["views"]
            
            # set threshold for engagement rate -> if engagement rate < threshold => bad otherwise if greater then => good
            df.insert(0,"target",0)
            threshold = df["engagement_rate"].median()
            df.loc[df["engagement_rate"] >= threshold, "target"] = 1
            df.loc[df["engagement_rate"] < threshold, "target"] = 0
            
            # splitting upload_hour into 4 categories 
            df.insert(0,"morning",0)
            df.insert(0,"afternoon",0)
            df.insert(0,"evening",0)
            df.insert(0,"night",0)
            
            # spliting upload_hour into 4 one hot encodings -> 0 | 1
            # df.loc just lets us pick the specific rows or columns within the dataframe  
            df.loc[(df["upload_hour"] >= 6) & (df["upload_hour"] <= 12), "morning"] = 1
            df.loc[(df["upload_hour"] >= 13) & (df["upload_hour"] <= 18), "afternoon"] = 1
            df.loc[(df["upload_hour"] >= 19) & (df["upload_hour"] <= 24), "evening"] = 1
            df.loc[(df["upload_hour"] >= 0) & (df["upload_hour"] <= 5), "night"] = 1
            
            # splitting categories into one hot encodings ->  0 | 1
            categories = df["category"].unique()
            for i in categories:
                df.insert(0,f"{i}", 0)
            
            for i in categories:
                df.loc[df["category"] == i, i] = 1
            
            # dropping specific columns that are irretating the models results 
            USELESS = ["views", "likes", "category", "shares", "comments", "upload_hour"]
            df.drop(USELESS, axis= 1, inplace=True)
            
          except Exception as e:
              print(f"Could not insert clean data... Process Failed: {e}")
              raise(ValueError)
          
          print("File is loaded...\n")
    except Exception as f:
        print(f"File could not load...Process Failed: {f}")
        raise(FileNotFoundError) 
    return df, categories

# splitting the data into training, validation, test data => 0.8 | 0.1 | 0.1
def split_data(df: pd.DataFrame) -> pd.DataFrame:
    Y = df["target"]
    X = df.drop(["engagement_rate", "target"], axis=1)
    # X_test and y-test are the test data and X_train, y_train are the training datas and X_valid, y_valid are validation data
    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, train_size=0.8, test_size=0.2, stratify=Y, random_state=SEED)
    X_valid,X_test,y_valid,y_test = train_test_split(X_temp,y_temp,train_size=0.5,test_size=0.5, stratify=y_temp, random_state=SEED)
    
    return X_train,y_train,X_valid,y_valid,X_test,y_test

# main algorithm is the random forest 
def random_forest(X_train: pd.DataFrame, y_train:pd.DataFrame, X_valid:pd.DataFrame,Y_valid:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.DataFrame):
    # using n= 200 for 200 trees 
    model = RandomForestClassifier(n_estimators= 200, class_weight="balanced",random_state=SEED)
    # training the model with the training data
    model.fit(X_train,y_train)
    
    # using the validation data on the model
    y_pred = model.predict(X_valid)
    class_report = classification_report(Y_valid,y_pred)
    
    # using the test data on the model
    pred = model.predict(X_test)
    test_report = classification_report(y_test,pred)
    print("model is trained...")
    print(class_report)
    print(test_report)
    
    return model,y_pred, class_report, y_test, test_report

# rearrange users inputs into our dataframes shape 
def make_prediction(duration, hashtags, hour, category):
    data = job.load("model.pkl")
    model = data["model"]
    model_columns = data["columns"]
    input_data = {col: 0 for col in model_columns}
    input_data['duration_sec'] = duration
    input_data['hashtags_count'] = hashtags
    if hour == "morning":
        input_data['morning'] = 1
    elif hour == "afternoon":
        input_data['afternoon'] = 1
    elif hour == "evening":
        input_data['evening'] = 1
    else: 
        input_data['night'] = 1

    # If user selects Category we set input_data[Category] = 1
    if category in input_data:
        input_data[category] = 1

    df = pd.DataFrame([input_data])
    
    # We want the probability of Class 1 (High Engagement)
    probability = model.predict_proba(df)[0][1]
    print(df)
    
    return probability, df