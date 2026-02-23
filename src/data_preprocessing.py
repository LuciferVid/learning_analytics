import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # calc total and avg
    df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
    df['percentage'] = df['total_score'] / 3
    
    # cat students based on score
    def get_grade(score):
        if score >= 80: return 'High-performing'
        if score >= 50: return 'Average'
        return 'At-risk'
    
    df['performance_category'] = df['percentage'].apply(get_grade)
    
    # encode categorical stuff
    encoders = {}
    cats = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    for col in cats:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    return df, encoders

def get_train_test_split(df):
    # drop scores and target for features
    X = df.drop(['performance_category', 'total_score', 'percentage', 'math score', 'reading score', 'writing score'], axis=1)
    y = df['performance_category']
    
    # simple mapping for target
    mapping = {'At-risk': 0, 'Average': 1, 'High-performing': 2}
    y = y.map(mapping)
    
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    
    sc = StandardScaler()
    xtrain = sc.fit_transform(xtrain)
    xtest = sc.transform(xtest)
    
    return xtrain, xtest, ytrain, ytest, mapping, sc
