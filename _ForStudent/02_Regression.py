import sklearn.linear_model

def build_linear_model(current_obervations_values, current_decision_values):
    # it is a linear model
    current_regression = sklearn.linear_model.LinearRegression()
    # fit in the observation
    current_regression.fit(
        X = current_obervations_values,
        y = current_decision_values
    )
    # return it
    return current_regression

def print_model_details(current_regression, current_observations_variables, current_decision_variable):
    print("Printing model details for "+ str(current_decision_variable))
    print("Intercept = " + str(current_regression.intercept_))
    print("Variables and their coeficient")
    for i in range(len(current_observations_variables)):
        print(str(current_observations_variables[i]) + " = " + str(current_regression.coef_))

def make_prediction(current_regression, new_observations, current_decision_variable):
    print("Making prediction for " + str(current_decision_variable))
    new_prediction = current_regression.predict(new_observations)
    for i in range(len(new_observations)):
        print(str(new_observations[i]) + " = " + str(new_prediction[i]))

if __name__ == "__main__":
    # set to current working directory
    import os
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # the dataset is a concrete dataset
    # this can be read from file as well
    # simple example with 20 observations and only 2 variables, there is more
    current_observations_variables = ["cement", "water"]
    current_obervations_values = [
        [540,162],[540,162],[332.5,228],[332.5,228],[198.6,192],[266,228],[380,228],[380,228],
        [266,228],[475,228],[198.6,192],[198.6,192],[427.5,228],[190,228],[304,228],[380,228],
        [139.6,192],[342,228],[380,228]
    ]
    current_decision_variable = ["Concrete Strength"]
    current_decision_values = [
        79.99,61.89,40.27,41.05,44.3,47.03,43.7,36.45,45.85,39.29,
        38.07,28.02,43.01,42.33,47.81,52.91,39.36,56.14,40.56
    ]
    
    # build the linear regression model
    current_regression = build_linear_model(current_obervations_values, current_decision_values)
    # print model details
    print_model_details(current_regression, current_observations_variables, current_decision_variable)

    # use the linear regression model
    new_observations = [
        [500,162],
        [330,162],
        [200,162]
    ]
    # make prediction
    make_prediction(current_regression, new_observations, current_decision_variable)
    
