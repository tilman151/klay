package org.klay

import junit.framework.TestCase.assertEquals
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.distribution.UniformDistribution
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.junit.Test
import org.klay.nn.dense
import org.klay.nn.layers
import org.klay.nn.output
import org.klay.nn.sequential
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nadam
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction


class ExampleTests {
    @Test
    fun mnistSingleLayerExample() {
        val numRows = 28
        val numColumns = 28
        val outputNum = 10
        val rngSeed = 123

        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(rngSeed.toLong()) //include a random seed for reproducibility
            // use stochastic gradient descent as an optimization algorithm
            .updater(Nesterovs(0.006, 0.9))
            .l2(1e-4)
            .list()
            .layer(
                DenseLayer.Builder() //create the first, input layer with xavier initialization
                    .nIn(numRows * numColumns)
                    .nOut(1000)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .build()
            )
            .layer(
                OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                    .nIn(1000)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build()
            )
            .build()
        val dl4jString = dl4jNet.toString()

        val klayNet = sequential {
            seed(rngSeed.toLong()) //include a random seed for reproducibility
            // use stochastic gradient descent as an optimization algorithm
            updater(Nesterovs(0.006, 0.9))
            l2(1e-4)
            layers {
                dense {
                    nIn(numRows * numColumns)
                    nOut(1000)
                    activation(Activation.RELU)
                    weightInit(WeightInit.XAVIER)
                }
                output {
                    lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                    nIn(1000)
                    nOut(outputNum)
                    activation(Activation.SOFTMAX)
                    weightInit(WeightInit.XAVIER)
                }
            }
        }
        val klayString = klayNet.toString()

        assertEquals(dl4jString, klayString)
    }

    @Test
    fun mnistDoubleLayerExample() {
        val numRows = 28
        val numColumns = 28
        val outputNum = 10 // number of output classes
        val rngSeed = 123 // random number seed for reproducibility
        val rate = 0.0015 // learning rate

        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(rngSeed.toLong()) //include a random seed for reproducibility
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .updater(Nadam())
            .l2(rate * 0.005) // regularize learning model
            .list()
            .layer(
                DenseLayer.Builder() //create the first input layer.
                    .nIn(numRows * numColumns)
                    .nOut(500)
                    .build()
            )
            .layer(
                DenseLayer.Builder() //create the second input layer
                    .nIn(500)
                    .nOut(100)
                    .build()
            )
            .layer(
                OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                    .activation(Activation.SOFTMAX)
                    .nOut(outputNum)
                    .build()
            )
            .build()
        val dl4jString = dl4jNet.toString()

        val klayNet = sequential {
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
                    lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                    activation(Activation.SOFTMAX)
                    nIn(100)
                    nOut(outputNum)
                }
            }
        }
        val klayString = klayNet.toString()

        assertEquals(dl4jString, klayString)
    }

    @Test
    fun irisClassifierExample() {
        val numInputs = 4
        val outputNum = 3
        val seed: Long = 6

        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(seed)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .updater(Sgd(0.1))
            .l2(1e-4)
            .list()
            .layer(
                DenseLayer.Builder().nIn(numInputs).nOut(3)
                    .build()
            )
            .layer(
                DenseLayer.Builder().nIn(3).nOut(3)
                    .build()
            )
            .layer(
                OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                    .activation(Activation.SOFTMAX) //Override the global TANH activation with softmax for this layer
                    .nIn(3).nOut(outputNum).build()
            )
            .build()
        val dl4jString = dl4jNet.toString()

        val klayNet = sequential {
            seed(seed)
            activation(Activation.TANH)
            weightInit(WeightInit.XAVIER)
            updater(Sgd(0.1))
            l2(1e-4)
            layers {
                dense {
                    nIn(numInputs)
                    nOut(3)
                }
                dense {
                    nIn(3)
                    nOut(3)
                }
                output {
                    lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                    activation(Activation.SOFTMAX)
                    nIn(3)
                    nOut(outputNum)
                }
            }
        }
        val klayString = klayNet.toString()

        assertEquals(dl4jString, klayString)
    }

    @Test
    fun linearDataClassifierExample() {
        val seed = 123
        val learningRate = 0.01
        val numInputs = 2
        val numOutputs = 2
        val numHiddenNodes = 20

        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .weightInit(WeightInit.XAVIER)
            .updater(Nesterovs(learningRate, 0.9))
            .list()
            .layer(
                DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                    .activation(Activation.RELU)
                    .build()
            )
            .layer(
                OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                    .activation(Activation.SOFTMAX)
                    .nIn(numHiddenNodes).nOut(numOutputs).build()
            )
            .build()
        val dl4jString = dl4jNet.toString()

        val klayNet = sequential {
            seed(seed.toLong())
            weightInit(WeightInit.XAVIER)
            updater(Nesterovs(learningRate, 0.9))
            layers {
                dense {
                    activation(Activation.RELU)
                    nIn(numInputs)
                    nOut(numHiddenNodes)
                }
                output {
                    lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                    activation(Activation.SOFTMAX)
                    nIn(numHiddenNodes)
                    nOut(numOutputs)
                }
            }
        }
        val klayString = klayNet.toString()

        assertEquals(dl4jString, klayString)
    }

    @Test
    fun modelXORExample() {
        val seed = 1234

        val dl4jNet = NeuralNetConfiguration.Builder()
            .updater(Sgd(0.1))
            .seed(seed.toLong())
            .biasInit(0.0) // init the bias with 0 - empirical value, too
            // The networks can process the input more quickly and more accurately by ingesting
            // minibatches 5-10 elements at a time in parallel.
            // This example runs better without, because the dataset is smaller than the mini batch size
            .miniBatch(false)
            .list()
            .layer(
                DenseLayer.Builder()
                    .nIn(2)
                    .nOut(4)
                    .activation(Activation.SIGMOID) // random initialize weights with values between 0 and 1
                    .weightInit(UniformDistribution(0.0, 1.0))
                    .build()
            )
            .layer(
                OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(2)
                    .activation(Activation.SOFTMAX)
                    .weightInit(UniformDistribution(0.0, 1.0))
                    .build()
            )
            .build()
        val dl4jString = dl4jNet.toString()

        val klayNet = sequential {
            updater(Sgd(0.1))
            seed(seed.toLong())
            biasInit(0.0) // init the bias with 0 - empirical value, too
            // The networks can process the input more quickly and more accurately by ingesting
            // minibatches 5-10 elements at a time in parallel.
            // This example runs better without, because the dataset is smaller than the mini batch size
            miniBatch(false)
            layers {
                dense {
                    nIn(2)
                    nOut(4)
                    activation(Activation.SIGMOID) // random initialize weights with values between 0 and 1
                    weightInit(UniformDistribution(0.0, 1.0))
                }
                output {
                    lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                    nOut(2)
                    activation(Activation.SOFTMAX)
                    weightInit(UniformDistribution(0.0, 1.0))
                }
            }
        }
        val klayString = klayNet.toString()

        assertEquals(dl4jString, klayString)
    }
}