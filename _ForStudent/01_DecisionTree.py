import sklearn.tree
import sklearn.metrics
import sklearn.model_selection


def split_dataset(current_obervations_values, current_decisions_values):
    # 70% training and 30% test
    current_observations_train, current_observations_test, current_decisions_train, current_decisions_test = sklearn.model_selection.train_test_split(
        current_obervations_values,
        current_decisions_values,
        test_size = 0.3,
        # note: 70:30 split
        random_state = 1
    )
    # return the values as training and testing
    return current_observations_train, current_observations_test, current_decisions_train, current_decisions_test


def build_tree(observations, decisions):
    # build the decision tree
    current_tree = sklearn.tree.DecisionTreeClassifier(
        criterion = "entropy",
        max_depth = 7
        # note: we can control the depth of our tree
    )
    current_tree = current_tree.fit(observations,decisions)
    # return the tree
    return current_tree

def visualize_tree(current_tree, observations_variables, decisions_variables):
    # needed to visualize the tree
    # note: we skip this part because it is more troublesome to setup
    import pydotplus
    # we need to set this up properly on the surface as well
    # C:\ProgramData\Anaconda3\Library\bin
    # set path to graphviz
    current_tree_dot = sklearn.tree.export_graphviz(
        current_tree,
        out_file =  None,
        feature_names = observations_variables,
        class_names = decisions_variables
        # filled = True,
        # rounded = True
    )
    # draw graph and output it
    graph = pydotplus.graph_from_dot_data(current_tree_dot) 
    graph.write_pdf("tree_test.pdf")


def make_prediction(new_observations):
    # print(new_observations)
    current_prediction = current_tree.predict(new_observations)
    # print(current_prediction)
    return current_prediction

def print_prediction(new_observations, current_prediction, current_observations_variables, current_decisions_variables):
    print(str(current_observations_variables) + " = either classes " + str(current_decisions_variables))
    for i in range(len(new_observations)):
        print(str(new_observations[i]) + " = " + str(current_prediction[i]))

if __name__ == "__main__":
    # set to current working directory
    import os
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # the dataset
    # this can be read from file as well
    current_observations_variables = ["height", "hair_length"]
    current_obervations_values = [
        [165,19],[175,32],[136,35],[174,65],[141,28],[176,15],[131,32],[166,6],
        [128,32],[179,10],[136,34],[186,2],[126,25],[176,28],[112,38],[169,9],
        [171,36],[116,25],[196,25]
    ]
    current_decisions_variables = ['Man','Woman']
    current_decisions_values = [
        'Man','Woman','Woman','Man','Woman','Man','Woman','Man','Woman','Man',
        'Woman','Man','Woman','Woman','Woman','Man','Woman','Woman','Man'
    ]

    # separate into training and testing
    current_observations_train, current_observations_test, current_decisions_train, current_decisions_test = split_dataset(current_obervations_values, current_decisions_values)

    # build the tree
    current_tree = build_tree(current_observations_train, current_decisions_train)

    # visualize the tree
    # we skip this part as it would take some time to set it up
    # visualize_tree(current_tree, current_observations_variables, current_decisions_variables)
    
    # use it for decision making
    new_observations = [
        [133,37],
        [170,14],
        [131,13]
    ]
    # put the values into the tree to make prediction
    current_prediction = make_prediction(new_observations)
    print_prediction(new_observations, current_prediction, current_observations_variables, current_decisions_variables)

    # what is our tree accuracy?
    current_prediction = current_tree.predict(current_observations_test)
    current_accuracy = sklearn.metrics.accuracy_score(
        current_decisions_test,
        current_prediction
    )
    print("Accuracy of the tree is: " + str(current_accuracy))