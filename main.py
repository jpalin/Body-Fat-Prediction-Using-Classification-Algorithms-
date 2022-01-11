import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle


# function for importing body data and selecting correct data
def import_dataset():
    body_data = pd.read_csv("data/bodyfat.csv")
    body_data = body_data.drop(["Density"], axis=1)
    body_data = body_data[body_data["Age"] < 60]

    return body_data


# function for editing body data
def remove_incorrect_data(body_data):
    # max and min values show incorrect height and body fat values of 29.5 inches and 0%
    v = body_data.describe().loc[['min', 'max']]
    body_data = body_data.loc[body_data['BodyFat'] != 0.0]
    body_data = body_data.loc[body_data['Height'] != 29.50]

    return body_data


# function for calculating BMI
def add_bmi_column(body_data):
    square_h = body_data["Weight"] / body_data["Height"]**2
    body_data["BMI"] = square_h * 703

    return body_data


# function for converting waist in cm to waist in inches
def convert_waist(body_data):
    # converting waist in cm to waist in inches
    body_data["waist"] = body_data["Abdomen"] / 2.54

    return body_data


'# function for calculating height to waist ratio due to research suggesting'
'# this ratio acts as a indicator for estimating body fat'


def calc_height_to_waist(body_data):

    body_data["heightToWaistRatio"] = round(body_data["waist"] / body_data["Height"], 2)

    return body_data


'#function to replace target value with binary yes or no, this value indicating if the individual is a healthy body fat'


def replace_body_fat_col(body_data):

    body_data["HealthyBodyFat?"] = np.where((body_data["BodyFat"] < 22), 'yes', 'no')

    return body_data


# function to select required columns, using input array

def select_cols(body_data, arr):

    body_data = body_data[arr]

    return body_data


# after analysis, 3 columns were determined the best variables for the model to indicate body fat


# function to define variables, with y given as input
def define_vars(body_data, y_var):

    # independent variable
    y = body_data.loc[:, y_var].values
    # dependent variable
    x = body_data.iloc[:, :-1].values

    return y, x


# function to train algorithms on data
def training_algorithms(y, x):

    # splitting using 20% of data as the test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # training using three classifier algorithms to allow for comparison
    clf1 = LogisticRegression()
    clf1.fit(x_train, y_train)
    clf2 = RandomForestClassifier()
    clf2.fit(x_train, y_train)
    clf3 = KNeighborsClassifier(n_neighbors=3)
    clf3.fit(x_train, y_train)

    return clf1, clf2, clf3, x_test, y_test


# function to evaluate performance of each algorithm and then selecting best based on the highest accuracy
def evaluate_algorithms(clf1, clf2, clf3, x_test, y_test):
    classifiers = [clf1, clf2, clf3]
    scores = [0, 0, 0]
    names = ["LogRegression:", "RandomForest:", "KNeighbors:"]
    algorithms_dict = dict([(names[0], classifiers[0]), (names[1], classifiers[1]), (names[2], classifiers[2])])
    scores_dict = dict([(names[0], scores[0]), (names[1], scores[0]), (names[2], scores[0])])
    for i in names:
        score = algorithms_dict[i].score(x_test, y_test)
        scores_dict.update({i: score})
        print(i, score)
    best_algo = max(scores_dict, key=scores_dict.get)
    print("Highest Accuracy - ", best_algo, scores_dict[best_algo])

    return algorithms_dict[best_algo]


def main():
    body_data = import_dataset()
    body_data = remove_incorrect_data(body_data)
    body_data = add_bmi_column(body_data)
    body_data = convert_waist(body_data)
    body_data = convert_waist(body_data)
    body_data = calc_height_to_waist(body_data)
    body_data = replace_body_fat_col(body_data)
    body_data = select_cols(body_data, ["Weight", "Height", "waist", "Ankle", "Wrist", "heightToWaistRatio", "BMI",
                                        "HealthyBodyFat?"])
    body_data = select_cols(body_data, ["waist", "Wrist", "heightToWaistRatio", "HealthyBodyFat?"])
    y, x = define_vars(body_data, "HealthyBodyFat?")
    clf1, clf2, clf3, x_test, y_test = training_algorithms(y, x)
    best_algo = evaluate_algorithms(clf1, clf2, clf3, x_test, y_test)
    return best_algo


if __name__ == "__main__":
    best_model = main()
    pickle.dump(best_model, open('model.pkl', 'wb'))
    model = pickle.load(open('model.pkl', 'rb'))
    print(model.predict([[30, 16, 47]]))
