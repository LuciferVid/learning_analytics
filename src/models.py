import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_preprocessing import load_and_preprocess_data, get_train_test_split

def train():
    print("loading...")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, 'data', 'StudentsPerformance.csv')
    
    df, encoders = load_and_preprocess_data(path)
    xtrain, xtest, ytrain, ytest, mapping, sc = get_train_test_split(df)
    
    print("training...")
    model = LogisticRegression(max_iter=1000)
    model.fit(xtrain, ytrain)
    
    print("Accuracy:", accuracy_score(ytest, model.predict(xtest)))
    
    # save all to models folder
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/target_mapping.pkl', 'wb') as f:
        pickle.dump(mapping, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(sc, f)
    with open('models/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
        
    print("All saved.")

if __name__ == "__main__":
    train()
