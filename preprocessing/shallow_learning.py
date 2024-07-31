from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB 

def split(df, target, test_size=0.33, random_state=42, **kwargs):
    display(df)
    X_train, X_test, y_train, y_test = train_test_split(df.drop([target], axis=1), 
                                                        df[target], test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def training_model(X_train, X_test, y_train, y_test, algorithm='rf', verbose=0, **kwargs):
    
    if algorithm == 'rf':
        clf = RandomForestClassifier(verbose=verbose)
    
    elif algorithm == 'nb':
        clf = MultinomialNB()

    else:
        print("You do not have any access to the code")
        return None

    model = clf.fit(X_train, y_train.ravel())

    if(verbose>0):
        print(f"Accuracy of the test with {algorithm}:", model.score(X_test,y_test.ravel()), '\n')

    return clf