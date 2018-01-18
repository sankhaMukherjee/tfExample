import json, logging
import numpy                  as np
import tensorflow             as tf
from logs import logDecorator as lD

config  = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.tfModule.smallNN'

class smallNN():

    @lD.log(logBase + '.smallNN.__init__')
    def __init__(logger, self, NNconfig, inpLayer=None):
        '''Initialize the NNmodel informamtion
        
        This is responsible for generating NN graphs that 
        will be used for generating the small NN. This is
        a system whose 'linearity' can be tuned using the
        alpha parameter. This function is used to produce
        a network that has a feedforward mechanism allowing 
        the network to be short-circuited and use this as
        an indexer ...
        
        Parameters
        ----------
        logger : {[type]}
            [description]
        self : {[type]}
            [description]
        NNconfig : {[type]}
            [description]
        '''

        self.NNdefined = False
        self.weights   = []

        try:
            logger.info('Starting a new NN')
            self.NNc     = NNconfig

            # The first and the last layers must be of the same size
            if self.NNc['layers'][0]['nUnits'] != self.NNc['layers'][-1]['nUnits']:
                logger.error('The first and the last units are not the same size')
                logger.error('This part will be skipped')
                return

            self.name    = self.NNc['name']
            self.nLayers = len(self.NNc['layers']) -1
            self.initOp  = tf.global_variables_initializer()
            self.linear  = tf.placeholder(dtype=tf.float32, shape=() )
            self.X       = tf.placeholder(dtype=tf.float32, shape=(self.NNc['layers'][0]['nUnits'], None))
            self.y       = inpLayer
            
            for d1, d2 in zip(self.NNc['layers'], self.NNc['layers'][1:]):
                print('-'*30)
                print(d1)
                print(d2)

                # Generate a weight matrix ... 
                # -----------------------------
                weights = np.array(np.random.random(( d2['nUnits'], d1['nUnits'] )) )
                weights -= .5
                weights *= self.NNc['initWeightFactor']
                weights = tf.convert_to_tensor( weights.astype(np.float32),
                    dtype = tf.float32,
                    name = '-'.join([d1['name'], d2['name']]) )
                self.weights.append( weights )

                # Generate the multiplications
                # -----------------------------
                if self.y is None:
                    self.y = tf.matmul( weights, self.X )
                else:
                    self.y = tf.matmul( weights, self.y )

                # Let us generate activation functions as necessary
                # -------------------------------------------------
                if d2['activation'] == 'relu':
                    self.y = tf.nn.relu( self.y )
                if d2['activation'] == 'relu6':
                    self.y = tf.nn.relu6( self.y )
                if d2['activation'] == 'crelu':
                    self.y = tf.nn.crelu( self.y )
                if d2['activation'] == 'elu':
                    self.y = tf.nn.elu( self.y )
                if d2['activation'] == 'softplus':
                    self.y = tf.nn.softplus( self.y )
                if d2['activation'] == 'softsign':
                    self.y = tf.nn.softsign( self.y )
                if d2['activation'] == 'sigmoid':
                    self.y = tf.nn.sigmoid( self.y )
                if d2['activation'] == 'tanh':
                    self.y = tf.nn.tanh( self.y )
                
            # Now we tune the non-linearity of the network
            self.y = ( 1 - self.linear) * self.y + self.linear * self.X
            
            # Here we say that the NN is properly initialized.
            self.NNdefined = True

        except Exception as e:
            logger.error('Unable to initialize the neural network: {}'.format(e))

        return

    @lD.log(logBase + '.smallNN.__repr__')
    def __repr__(logger, self):

        result = ''
        
        if self.NNdefined:
            result = '\n\nInformation about the model: {}:\n'.format(self.name)
            result += '#'*30 + '\n'
            result += '\nThe number of layers: {}'.format(self.nLayers) + '\n'
            
            result += '\n----------[{:30}]--------'.format('Weights') + '\n'
            with tf.Session() as sess:
                sess.run(self.initOp)
                for w in self.weights:
                    result += '-'*30 + '\n'
                    result += str(sess.run(w)) + '\n'
        else:
            result = 'This NN has not been defined properly'
            logger.error('Unable to display the network since there was an error during the creation')

        return result

    @lD.log(logBase + 'smallNN.__trainable__')
    def __trainable__(logger, self):

        for w in self.weights:
            print(w.trainable)

        return

    @lD.log(logBase + 'smallNN.__simpleTests__')
    def __simpleTests__(logger, self):

        if not self.NNdefined:
            logger.error('The network isnt generated properly.')
            logger.error('The test will be skipped ...')
            return

        print('Testing the model ... :')
        print('#'*30)

        testX = np.ones((self.NNc['layers'][0]['nUnits'], 2))
        print('\nInput matrix: ')
        print(testX)

        print('\nResulting matrix:')
        with tf.Session() as sess:
            sess.run(self.initOp)

            for linear in np.linspace(0, 1, 4):
                print('alpha = {}'.format(linear))
                result = sess.run([self.y], feed_dict={
                    self.X      : testX,
                    self.linear : linear
                })
                print(result)

        return


