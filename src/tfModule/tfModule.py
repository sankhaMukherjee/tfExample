from logs import logDecorator as lD 
from tfModule import smallNN
import json
import tensorflow as tf

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.module1'


@lD.log(logBase + '.makeNN')
def makeNN(logger):

    NNconfig = json.load(open('../config/smallNN.json'))

    # tf.Constant(np.ones(NNconfig['layers'][0]['nUnits']))
    someModel = smallNN.smallNN(NNconfig)
    print(someModel)
    someModel.__simpleTests__()
    someModel.__trainable__()

    print('-'*30)

    return

@lD.log(logBase + '.main')
def main(logger):
    '''main function for tfModel
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger function
    '''

    makeNN()

    return

