/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/
 * deeplearning4j/examples/quickstart/modeling/feedforward/classification/MNISTDoubleLayer.java **/

package org.klay.examples.classification

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Nadam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import org.klay.nn.*


/** A slightly more involved multilayered (MLP) applied to digit classification for the MNIST dataset (http://yann.lecun.com/exdb/mnist/).
 *
 * This example uses two input layers and one hidden layer.
 *
 * The first input layer has input dimension of numRows*numColumns where these variables indicate the
 * number of vertical and horizontal pixels in the image. This layer uses a rectified linear unit
 * (relu) activation function. The weights for this layer are initialized by using Xavier initialization
 * (https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
 * to avoid having a steep learning curve. This layer sends 500 output signals to the second layer.
 *
 * The second input layer has input dimension of 500. This layer also uses a rectified linear unit
 * (relu) activation function. The weights for this layer are also initialized by using Xavier initialization
 * (https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
 * to avoid having a steep learning curve. This layer sends 100 output signals to the hidden layer.
 *
 * The hidden layer has input dimensions of 100. These are fed from the second input layer. The weights
 * for this layer is also initialized using Xavier initialization. The activation function for this
 * layer is a softmax, which normalizes all the 10 outputs such that the normalized sums
 * add up to 1. The highest of these normalized values is picked as the predicted class.
 *
 */
object MNISTDoubleLayer {
    private val log = LoggerFactory.getLogger(MNISTDoubleLayer::class.java)
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        //number of rows and columns in the input pictures
        val numRows = 28
        val numColumns = 28
        val outputNum = 10 // number of output classes
        val batchSize = 64 // batch size for each epoch
        val rngSeed = 123 // random number seed for reproducibility
        val numEpochs = 15 // number of epochs to perform
        val rate = 0.0015 // learning rate

        //Get the DataSetIterators:
        val mnistTrain: DataSetIterator = MnistDataSetIterator(batchSize, true, rngSeed)
        val mnistTest: DataSetIterator = MnistDataSetIterator(batchSize, false, rngSeed)
        log.info("Build model....")
        val conf = sequential {
            seed(rngSeed.toLong()) //include a random seed for reproducibility
            // use stochastic gradient descent as an optimization algorithm
            activation(Activation.RELU)
            weightInit(WeightInit.XAVIER)
            updater(Nadam())
            l2(rate * 0.005) // regularize learning model
            layers {
                dense {
                    nIn(numRows * numColumns)
                    nOut(500)
                }
                dense {
                    nIn(500)
                    nOut(100)
                }
                output {
                    lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    activation(Activation.SOFTMAX)
                    nIn(100)
                    nOut(outputNum)
                }
            }
        }
        val model = MultiLayerNetwork(conf)
        model.init()
        model.setListeners(ScoreIterationListener(5)) //print the score with every iteration
        log.info("Train model....")
        model.fit(mnistTrain, numEpochs)
        log.info("Evaluate model....")
        val eval = model.evaluate<Evaluation>(mnistTest)
        log.info(eval.stats())
        log.info("****************Example finished********************")
    }
}