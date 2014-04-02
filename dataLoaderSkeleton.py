import Backprop_skeleton as backprop
from pylab import *

#Class for holding your data - one object for each line in the dataset
class dataInstance:

    def __init__(self,qid,rating,features):
        self.qid = qid #ID of the query
        self.rating = rating #Rating of this site for this query
        self.features = features #The features of this query-site pair.

    def __str__(self):
        return "Datainstance - qid: "+ str(self.qid)+ ". rating: "+ str(self.rating)+ ". features: "+ str(self.features)


#A class that holds all the data in one of our sets (the training set or the testset)
class dataHolder:

    def __init__(self, dataset):
        self.dataset = self.loadData(dataset)

    def loadData(self,file):
        #Input: A file with the data.
        #Output: A dict mapping each query ID to the relevant documents, like this: dataset[queryID] = [dataInstance1, dataInstance2, ...]
        data = open(file)
        dataset = {}
        for line in data:
            #Extracting all the useful info from the line of data
            lineData = line.split()
            rating = int(lineData[0])
            qid = int(lineData[1].split(':')[1])
            features = []
            for elem in lineData[2:]:
                if '#docid' in elem: #We reached a comment. Line done.
                    break
                features.append(float(elem.split(':')[1]))
            #Creating a new data instance, inserting in the dict.
            di = dataInstance(qid,rating,features)
            if qid in dataset.keys():
                dataset[qid].append(di)
            else:
                dataset[qid]=[di]
        return dataset



def order_pairs(data_instance):
    #TODO: Store the training instances into the trainingPatterns array. Remember to store them as pairs, where the first item is rated higher than the second.
    #TODO: Hint: A good first step to get the pair ordering right, is to sort the instances based on their rating for this query. (sort by x.rating for each x in data_instance)

    features = []

    for i in data_instance:
        for j in data_instance:
            if i.rating != j.rating:
                if i.rating > j.rating:
                    features.append((i.features, j.features,))
                else:
                    features.append((j.features, i.features,))

    return features

def rank(training_set, test_set):

    dh_training = dataHolder(training_set)
    dh_testing = dataHolder(test_set)

    #TODO: The lists below should hold training patterns in this format: [(data1Features,data2Features), (data1Features,data3Features), ... , (dataNFeatures,dataMFeatures)]
    #TODO: The training set needs to have pairs ordered so the first item of the pair has a higher rating.
    training_patterns = []
    for qid in dh_training.dataset.keys():
        data_instance=dh_training.dataset[qid]
        pattern = order_pairs(data_instance)
        if pattern is not []:
            training_patterns.append(order_pairs(data_instance))

    test_patterns = []
    for qid in dh_testing.dataset.keys():
        data_instance=dh_testing.dataset[qid]
        pattern = order_pairs(data_instance)
        if pattern is not []:
            test_patterns.append(order_pairs(data_instance))

    #Creating an ANN instance - feel free to experiment with the learning rate (the third parameter).
    neural_network = backprop.NeuralNetwork(46,10,0.001)

    error_percent_test = []
    error_percent_training = []
    error_percent_test.append(neural_network.countMisorderedPairs(test_patterns))
    error_percent_training.append((neural_network.countMisorderedPairs(training_patterns)))
    iterations = 20
    for i in range(iterations):
        print("Iteration #", i)
        #Running 25 iterations, measuring testing performance after each round of training.
        #Training
        neural_network.train(training_patterns, iterations=1)
        #Check ANN performance after training.
        error_percent_test.append(neural_network.countMisorderedPairs(test_patterns))
        error_percent_training.append(neural_network.countMisorderedPairs(training_patterns))

    plot(range(1,iterations+2),error_percent_test, label="Test")
    plot(range(1,iterations+2),error_percent_training, label="Training")
    ylim([0,1])
    show()

    return error_percent_test, error_percent_training
    #TODO: Store the data returned by countMisorderedPairs and plot it, showing how training and testing errors develop.


print rank("train.txt", "test.txt")
