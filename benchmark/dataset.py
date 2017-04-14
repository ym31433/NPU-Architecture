from random import shuffle

class Dataset:
    '''individual data sets
    Data member:
        input_data: 2-dimensional input data
        golden_data: 2-dimensional corresponding output golden data
        num_touched: indicates the number of touched data
    '''
#    input_data
#    golden_data
    num_touched = 0

    def __init__(self, in_data, gold_data):
        self.input_data = in_data
        self.golden_data = gold_data

    def next_batch(self, batch_size):
        '''provides a list of untouched data

        Args:
            batch_size: the number of untouched data to be returned

        Return:
            in_frac: the untouched fraction of input data with size batch_size
            gold_frac: the untouched fraction of output golden data with size batch_size
        '''
        num_data = len(self.input_data)
        self.num_touched += batch_size
        if self.num_touched >= num_data:
            indices_shuffle = range(num_data)
            shuffle(indices_shuffle)
            self.input_data = [ self.input_data[i] for i in indices_shuffle ]
            self.golden_data = [ self.golden_data[i] for i in indices_shuffle ]
            self.num_touched = batch_size
        in_frac = self.input_data[self.num_touched-batch_size:self.num_touched]
        gold_frac = self.golden_data[self.num_touched-batch_size:self.num_touched]
        #debug
        #print self.num_touched
        return in_frac, gold_frac

    def max_steps(self, batch_size):
        '''computes the max steps and the corresponding number of data examples

        Args:
            batch_size: batch size. determines the max steps of training/validation/testing

        Returns:
            steps_per_epoch: max steps of training/validation/testing
            num_ex: number of examples can be used
        '''
        steps_per_epoch = len(self.input_data) // batch_size
        #debug
        #print "number of input data"
        #print len(self.input_data)
        #print "max steps"
        #print steps_per_epoch
        num_ex = steps_per_epoch * batch_size
        return num_ex, steps_per_epoch

    def reset_touched(self):
        self.num_touched = 0
'''
    def get_data_size(self):
        return len(self.input_data)
'''

class Datasets:
    '''whole data sets. gathering training, validation and testing data

    Data members:
        train: training data, type: dataset
        validate: validation data, type: dataset
        test: testing data, type: dataset
        num_in_neuron: number of input neurons
        num_out_neuron: number of output neurons
    '''
#    train
#    validate
#    test

    #def __init__(self, data_dir, separate, type_input, type_golden, tile_size, num_maps):
    def __init__(self, data_dir, separate, type_input, type_golden):
        '''initialize training, validation, and testing data
        Args:
            data_dir: directory of data
            separate: indicates whether the data are already separated into training, validation, and testing
            type_in: type of input data
            type_gold: type of golden data
            #for training hotspot (loading data file)
            tile_size: tile size
            num_maps: number of maps
        '''
        if type_input == 'float':
            type_in = float
        else:
            type_in = int
        if type_golden == 'float':
            type_gold = float
        else:
            type_gold = int

        #for training hotspot
        '''
        tile_size_str = str(tile_size)
        num_maps_str = str(num_maps)
        '''

        if not separate:
            # import input file
            inFile = open(data_dir+"input.txt")
            num_in = int(inFile.readline())
            self.input_data = [ [type_in(i) for i in inputs.strip().split(' ')]
                    for inputs in inFile.readlines() ]
            self.num_in_neuron = len(self.input_data[0])
            #debug
            print len(self.input_data)
            assert(num_in == len(self.input_data))
            #debug
            #print "num_in"
            #print num_in
            # import golden file
            goldFile = open(data_dir+"golden.txt")
            num_gold = int(goldFile.readline())
            golden_data = [ [type_gold(i) for i in goldens.strip().split(' ')]
                    for goldens in goldFile.readlines() ]
            self.num_out_neuron = len(golden_data[0])
            assert(num_gold == len(golden_data))
            #debug
            #print "num_gold"
            #print num_gold
            '''
            # shuffle
            assert(num_in == num_gold)
            indices_shuffle = range(num_in)
            shuffle(indices_shuffle)
            input_shuffle = [ input_data[i] for i in indices_shuffle ]
            golden_shuffle = [ golden_data[i] for i in indices_shuffle ]
            '''
            # separate the data into train, validate, and test
            train_size = int(float(num_in) * 0.7)
            validate_size = int(float(num_in) * 0.2)
            test_offset = train_size + validate_size
            self.train = Dataset(self.input_data[0:train_size],
                    golden_data[0:train_size])
            self.validate = Dataset(self.input_data[train_size:test_offset],
                    golden_data[train_size:test_offset])
            self.test = Dataset(self.input_data[test_offset:-1],
                    golden_data[test_offset:-1])
        else:
            ### training data ###
            # import input file
            train_inFile = open(data_dir + 'train_input.txt')
            num_train_in = int(train_inFile.readline())
            train_in_data = [ [type_in(i) for i in inputs.strip().split(' ') ]
                    for inputs in train_inFile.readlines() ]
            self.num_in_neuron = len(train_in_data[0])
            assert(num_train_in == len(train_in_data))
            # import golden file
            train_goldFile = open(data_dir + 'train_golden.txt')
            num_train_gold = int(train_goldFile.readline())
            train_gold_data = [ [type_gold(i) for i in goldens.strip().split(' ')]
                    for goldens in train_goldFile.readlines() ]
            self.num_out_neuron = len(train_gold_data[0])
            assert(num_train_gold == len(train_gold_data))
            # shuffle
            assert(num_train_in == num_train_gold)
            indices_shuffle = range(num_train_in)
            shuffle(indices_shuffle)
            train_in_shuffle = [ train_in_data[i] for i in indices_shuffle ]
            train_gold_shuffle = [ train_gold_data[i] for i in indices_shuffle ]
            # initialize training data
            self.train = Dataset(train_in_shuffle, train_gold_shuffle)

            ### validation data ###
            # import input file
            validate_inFile = open(data_dir + 'validate_input.txt')
            num_validate_in = int(validate_inFile.readline())
            validate_in_data = [ [type_in(i) for i in inputs.strip().split(' ') ]
                    for inputs in validate_inFile.readlines() ]
            assert(self.num_in_neuron == len(validate_in_data[0]))
            assert(num_validate_in == len(validate_in_data))
            # import golden file
            validate_goldFile = open(data_dir + 'validate_golden.txt')
            num_validate_gold = int(validate_goldFile.readline())
            validate_gold_data = [ [type_gold(i) for i in goldens.strip().split(' ')]
                    for goldens in validate_goldFile.readlines() ]
            assert(self.num_out_neuron == len(validate_gold_data[0]))
            assert(num_validate_gold == len(validate_gold_data))
            # initialize validation data
            assert(num_validate_in == num_validate_gold)
            self.validate = Dataset(validate_in_data, validate_gold_data)

            ### testing data ###
            # import input file
            test_inFile = open(data_dir + 'test_input.txt')
            num_test_in = int(test_inFile.readline())
            test_in_data = [ [type_in(i) for i in inputs.strip().split(' ') ]
                    for inputs in test_inFile.readlines() ]
            assert(self.num_in_neuron == len(test_in_data[0]))
            assert(num_test_in == len(test_in_data))
            # import golden file
            test_goldFile = open(data_dir + 'test_golden.txt')
            num_test_gold = int(test_goldFile.readline())
            test_gold_data = [ [type_gold(i) for i in goldens.strip().split(' ')]
                    for goldens in test_goldFile.readlines() ]
            assert(self.num_out_neuron == len(test_gold_data[0]))
            assert(num_test_gold == len(test_gold_data))
            # initialize testing data
            assert(num_test_in == num_test_gold)
            self.test = Dataset(test_in_data, test_gold_data)

