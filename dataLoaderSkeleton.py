import Backprop_skeleton as backprop

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


def extract_unique_instances(data_instance):
    unique_instances = []
    for instance in data_instance:
        ratings_seen = [x.rating for x in uniquely_rated_instances]
        if instance.rating not in ratings_seen:
            unique_instances.append(instance)
    return unique_instances


def order_pairs(data_instance):
    #TODO: Store the training instances into the trainingPatterns array. Remember to store them as pairs, where the first item is rated higher than the second.
    #TODO: Hint: A good first step to get the pair ordering right, is to sort the instances based on their rating for this query. (sort by x.rating for each x in data_instance)


    # Since we can safely ignore instances with the same rating
    # we filter out any obsolete list items and return a new list
    # of only useful elements.
    uniquely_rated_instances = extract_unique_instances(data_instance)

    # Adds all permutations of unique instance pairs where A > B holds true.
    # By sorting our list of instances we ensure that A will always have a higher rating
    # than B.
    sorted_instances = sorted(uniquely_rated_instances)
    features = []
    for i in sorted_instances:
        for j in sorted_instances:
            if i.rating > j.rating:
                features.append((i.features, j.features,))

    return features

def rank(training_set, test_set):

    dh_training = dataHolder(training_set)
    dh_testing = dataHolder(test_set)

    #TODO: The lists below should hold training patterns in this format: [(data1Features,data2Features), (data1Features,data3Features), ... , (dataNFeatures,dataMFeatures)]
    #TODO: The training set needs to have pairs ordered so the first item of the pair has a higher rating.
    training_patterns = []
    for qid in dh_training.dataset.keys():
        data_instance=dh_training.dataset[qid]
        training_patterns.append(order_pairs(data_instance))

    test_patterns = []
    for qid in dh_testing.dataset.keys():
        data_instance=dh_testing.dataset[qid]
        test_patterns.append(order_pairs(data_instance))

    #Creating an ANN instance - feel free to experiment with the learning rate (the third parameter).
    neural_network = backprop.NN(46,10,0.001)

    training_performance = []
    training_performance.append(neural_network.countMisorderedPairs(test_patterns))
    for i in range(25):
        #Running 25 iterations, measuring testing performance after each round of training.
        #Training
        neural_network.train(training_patterns, iterations=1)
        #Check ANN performance after training.
        training_performance.append(neural_network.countMisorderedPairs(test_patterns))

    return training_performance
    #TODO: Store the data returned by countMisorderedPairs and plot it, showing how training and testing errors develop.


print rank("train.txt", "test.txt")
