import numpy as np
import pandas as pd

def load(filename, include_demographics=False):
    df = pd.read_csv(f"{filename}")
    if not include_demographics:
        df = df.drop(columns=["Demographic"])
    
    return df

def get_p_x_given_y(x_column, y_column, df):
    """
    Computes P(X = 1 | Y = 1) and P(X = 1 | Y = 0), where X is a single feature (column).
    x_column: name of the column containing the feature X.
    y_column: name of the class containing the class label.

    return: [P(X = 1 | Y = 1), P(X = 1 | Y = 0)]
    """
    
    y_1 = df[y_column].sum()
    y_0 = len(df) - y_1

    x_1_y_0 = df.loc[df[y_column] == 0, x_column].sum()
    x_1_y_1 = df.loc[df[y_column] == 1, x_column].sum()

    p_0 = (x_1_y_0 + 1) / (y_0 + 2)
    p_1 = (x_1_y_1 + 1) / (y_1 + 2)


    return [p_0, p_1]

def get_all_p_x_given_y(y_column, df):
    # We want to store P(X_i=1 | Y=y) in p_x_given_y[i][y]
    all_p_x_given_y = np.zeros((df.shape[1]-1, 2))


    for i, column in enumerate(df.columns):
        if column != y_column:
            col_index = df.columns.get_loc(column)
            p = get_p_x_given_y(column, y_column, df)
            all_p_x_given_y[col_index] = p
    return all_p_x_given_y


def get_p_y(y_column, df):
    """
    Compute P(Y = 1)
    """

    y = df[y_column].sum()
    total = len(df)
    p_y = y/total

    print("estimate for p_Y == 1", p_y)

    return p_y

    
def get_p_x(x_column, df):
    """
    Compute P(xi = 1)
    """
    x = df[x_column].sum()
    total = len(df)
    p_x = x/total
    return p_x

def p_x_table(df):
    #Creates a dictionary with P(xi) for every xi
    dictionary = {}
    for i, column in enumerate(df.columns):
        if column != "Label":
            col_index = df.columns.get_loc(column)
            p = get_p_x(column, df)
            dictionary[col_index] = p
    return dictionary

def compute_best_movies(get_all_p_x_given_y, df):
    ratios = {}
    #probs_of_x is a dictionary with every p_xi
    probs_of_x = p_x_table(df)
    #p_y as an int
    p_y = get_p_y("Label", df)
    i = 0
    for x_giveny in get_all_p_x_given_y:
        x_0_y_1 = x_giveny[0]
        bayes1 = (x_0_y_1 * p_y) / (probs_of_x[i])
        x_1_y_1 = x_giveny[1]
        bayes2 = (x_1_y_1 * p_y) / (1 - probs_of_x[i])
        ratios[i] = (bayes1/bayes2)
        i += 1
    
    sorted_ratios = sorted(ratios.items(), key=lambda x:x[1])

    return sorted_ratios




def joint_prob(xs, y, all_p_x_given_y, p_y):
    """
    Computes the joint probability of a single row and y
    """
    prob = p_y
    for i, x in enumerate(xs):
        if x == 1:
            prob *= all_p_x_given_y[i][y]
        else:
            prob *= (1 - all_p_x_given_y[i][y])

    return prob

def get_prob_y_given_x(y, xs, all_p_x_given_y, p_y):
    """
    Computes the probability of a single row given y.
    """

    n, _ = all_p_x_given_y.shape # n is the number of features/columns


    prob_y_1 = joint_prob(xs, y, all_p_x_given_y, p_y)
    prob_y_0 = joint_prob(xs, 0, all_p_x_given_y, 1 - p_y)
    prob_y_given_x = prob_y_1 / (prob_y_0 + prob_y_1)
    

    return prob_y_given_x




def compute_accuracy(all_p_x_given_y, p_y, df):
    # split the test set into X and y. The predictions should not be able to refer to the test y's.
    X_test = df.drop(columns="Label")
    y_test = df["Label"]

    num_correct = 0
    #Total number of columns
    tot = len(y_test)

    for i, xs in X_test.iterrows():
        prob_y_given_x_1 = get_prob_y_given_x(1, xs, all_p_x_given_y, p_y)
        prob_y_given_x_0 = get_prob_y_given_x(0, xs, all_p_x_given_y, p_y)

        y = 1 if prob_y_given_x_1 >= 0.5 else 0
        if y == y_test[i]:
            num_correct += 1
    #Finds probability by dividing the number of correct by the number of rows
    accuracy = num_correct / tot

    a = get_all_p_x_given_y("Label", df)
    print(compute_best_movies(a, df))

    return accuracy

def main():
    # load the training set
    df_train = load("netflix-train.csv", include_demographics=False)

    # compute model parameters (i.e. P(Y), P(X_i|Y))
    all_p_x_given_y = get_all_p_x_given_y("Label", df_train)
    p_y = get_p_y("Label", df_train)

    # load the test set
    df_test = load("netflix-test.csv", include_demographics=False)

    print(f"Training accuracy: {compute_accuracy(all_p_x_given_y, p_y, df_train)}")
    print(f"Test accuracy: {compute_accuracy(all_p_x_given_y, p_y, df_test)}")

if __name__ == "__main__":
    main()
