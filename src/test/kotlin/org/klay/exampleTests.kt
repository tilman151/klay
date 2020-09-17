package org.klay

import junit.framework.TestCase.assertEquals
import org.datavec.image.loader.CifarLoader
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.distribution.UniformDistribution
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.weights.WeightInit
import org.junit.Test
import org.klay.examples.convolution.CIFARClassifier
import org.klay.nn.*
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.*
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

        assertNetsEquals(dl4jNet, klayNet)
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

        assertNetsEquals(dl4jNet, klayNet)
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

        assertNetsEquals(dl4jNet, klayNet)
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

        assertNetsEquals(dl4jNet, klayNet)
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

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun moonClassifierExample() {
        val seed = 123
        val learningRate = 0.005
        val numInputs = 2
        val numOutputs = 2
        val numHiddenNodes = 50

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
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.SOFTMAX)
                    .nIn(numHiddenNodes).nOut(numOutputs).build()
            )
            .build()

        val klayNet = sequential {
            seed(seed.toLong())
            weightInit(WeightInit.XAVIER)
            updater(Nesterovs(learningRate, 0.9))
            layers {
                dense {
                    nIn(numInputs)
                    nOut(numHiddenNodes)
                    activation(Activation.RELU)
                }
                output {
                    lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                    weightInit(WeightInit.XAVIER)
                    activation(Activation.SOFTMAX)
                    nIn(numHiddenNodes)
                    nOut(numOutputs)
                }
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun csvDataModelExample() {
        val seed = 12345
        val learningRate = 0.00001
        val numInputs = 1
        val numOutputs = 1

        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .weightInit(WeightInit.XAVIER)
            .updater(Nesterovs(learningRate, 0.9))
            .list()
            .layer(
                DenseLayer.Builder().nIn(numInputs).nOut(numOutputs)
                    .activation(Activation.IDENTITY)
                    .build()
            )
            .layer(
                OutputLayer.Builder(LossFunction.MSE)
                    .activation(Activation.IDENTITY)
                    .nIn(numOutputs).nOut(numOutputs).build()
            )
            .build()

        val klayNet = sequential {
            seed(seed.toLong())
            weightInit(WeightInit.XAVIER)
            updater(Nesterovs(learningRate, 0.9))
            layers {
                dense {
                    nIn(numInputs)
                    nOut(numOutputs)
                    activation(Activation.IDENTITY)
                }
                output {
                    lossFunction(LossFunction.MSE)
                    activation(Activation.IDENTITY)
                    nIn(numOutputs)
                    nOut(numOutputs)
                }
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun mathFunctionsModelExample() {
        val learningRate = 0.01
        val seed = 12345
        val numInputs = 1
        val numOutputs = 1
        val numHiddenNodes = 50

        val dl4jNet = NeuralNetConfiguration.Builder()
                .seed(seed.toLong())
                .weightInit(WeightInit.XAVIER)
                .updater(Nesterovs(learningRate, 0.9))
                .list()
                .layer(DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.TANH).build())
                .layer(DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.TANH).build())
                .layer(OutputLayer.Builder(LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build()

        val klayNet = sequential {
            seed(seed.toLong())
            weightInit(WeightInit.XAVIER)
            activation(Activation.TANH)
            updater(Nesterovs(learningRate, 0.9))
            layers {
                dense {
                    nIn(numInputs)
                    nOut(numHiddenNodes)
                }
                dense {
                    nIn(numHiddenNodes)
                    nOut(numHiddenNodes)
                }
                output {
                    lossFunction(LossFunction.MSE)
                    activation(Activation.IDENTITY)
                    nIn(numHiddenNodes)
                    nOut(numOutputs)
                }
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun sumModelExample() {
        val seed = 12345
        val numInput = 2
        val numOutputs = 1
        val nHidden = 10
        val learningRate = 0.01

        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .weightInit(WeightInit.XAVIER)
            .updater(Nesterovs(learningRate, 0.9))
            .list()
            .layer(0, DenseLayer.Builder().nIn(numInput).nOut(nHidden)
                    .activation(Activation.TANH) //Change this to RELU and you will see the net learns very well very quickly
                    .build())
            .layer(1, OutputLayer.Builder(LossFunction.MSE)
                    .activation(Activation.IDENTITY)
                    .nIn(nHidden).nOut(numOutputs).build())
            .build()

        val klayNet = sequential {
            seed(seed.toLong())
            weightInit(WeightInit.XAVIER)
            updater(Nesterovs(learningRate, 0.9))
            layers {
                dense {
                    nIn(numInput)
                    nOut(nHidden)
                    activation(Activation.TANH) //Change this to RELU and you will see the net learns very well very quickly
                }
                output {
                    lossFunction(LossFunction.MSE)
                    activation(Activation.IDENTITY)
                    nIn(nHidden)
                    nOut(numOutputs)
                }
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun mnistAutoencoderExample() {
        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(12345)
            .weightInit(WeightInit.XAVIER)
            .updater(AdaGrad(0.05))
            .activation(Activation.RELU)
            .l2(0.0001)
            .list()
            .layer(DenseLayer.Builder().nIn(784).nOut(250)
                    .build())
            .layer(DenseLayer.Builder().nIn(250).nOut(10)
                    .build())
            .layer(DenseLayer.Builder().nIn(10).nOut(250)
                    .build())
            .layer(OutputLayer.Builder().nIn(250).nOut(784)
                    .lossFunction(LossFunction.MSE)
                    .build())
            .build()

        val hiddenUnits = listOf(784, 250, 10, 250)
        val outputUnits = 784
        val klayNet = sequential {
            seed(12345)
            weightInit(WeightInit.XAVIER)
            updater(AdaGrad(0.05))
            activation(Activation.RELU)
            l2(0.0001)
            layers {
                for (u in hiddenUnits.zipWithNext()) { // Dynamically generate layers from units list
                    dense {
                        nIn(u.first)
                        nOut(u.second)
                    }
                }
                output {
                    lossFunction(LossFunction.MSE)
                    nIn(hiddenUnits.last())
                    nOut(outputUnits)
                }
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun cifarClassifierExample() {
        val height = 32
        val width = 32
        val channels = 3
        val numLabels = CifarLoader.NUM_LABELS
        val seed = 123L

        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(seed)
            .updater(AdaDelta())
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(
                ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                    .activation(Activation.LEAKYRELU)
                    .nIn(channels).nOut(32).build()
            )
            .layer(BatchNormalization())
            .layer(
                SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2)
                    .poolingType(SubsamplingLayer.PoolingType.MAX).build()
            )
            .layer(
                ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(1, 1)
                    .activation(Activation.LEAKYRELU)
                    .nOut(16).build()
            )
            .layer(BatchNormalization())
            .layer(
                ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                    .activation(Activation.LEAKYRELU)
                    .nOut(64).build()
            )
            .layer(BatchNormalization())
            .layer(
                SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2)
                    .poolingType(SubsamplingLayer.PoolingType.MAX).build()
            )
            .layer(
                ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(1, 1)
                    .activation(Activation.LEAKYRELU)
                    .nOut(32).build()
            )
            .layer(BatchNormalization())
            .layer(
                ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                    .activation(Activation.LEAKYRELU)
                    .nOut(128).build()
            )
            .layer(BatchNormalization())
            .layer(
                ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(1, 1)
                    .activation(Activation.LEAKYRELU)
                    .nOut(64).build()
            )
            .layer(BatchNormalization())
            .layer(
                ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(1, 1)
                    .activation(Activation.LEAKYRELU)
                    .nOut(numLabels).build()
            )
            .layer(BatchNormalization())
            .layer(
                SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2)
                    .poolingType(SubsamplingLayer.PoolingType.AVG).build()
            )
            .layer(
                OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                    .name("output")
                    .nOut(numLabels)
                    .dropOut(0.8)
                    .activation(Activation.SOFTMAX)
                    .build()
            )
            .setInputType(InputType.convolutional(height.toLong(), width.toLong(), channels.toLong()))
            .build()

        val klayNet = sequential {
            seed(seed)
            updater(AdaDelta())
            optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            weightInit(WeightInit.XAVIER)
            activation(Activation.LEAKYRELU)
            layers {
                conv2d {
                    kernelSize(3, 3)
                    stride(1, 1)
                    padding(1, 1)
                    nIn(channels)
                    nOut(32)
                }
                batchNorm {
                    activation(Activation.SIGMOID)
                }
                subsampling {
                    kernelSize(2, 2)
                    stride(2, 2)
                    poolingType(SubsamplingLayer.PoolingType.MAX)
                }
                conv2d {
                    kernelSize(1, 1)
                    stride(1, 1)
                    padding(1, 1)
                    nOut(16)
                }
                batchNorm {
                    activation(Activation.SIGMOID)
                }
                conv2d {
                    kernelSize(3, 3)
                    stride(1, 1)
                    padding(1, 1)
                    nOut(64)
                }
                batchNorm {
                    activation(Activation.SIGMOID)
                }
                subsampling {
                    kernelSize(2, 2)
                    stride(2, 2)
                    poolingType(SubsamplingLayer.PoolingType.MAX)
                }
                conv2d {
                    kernelSize(1, 1)
                    stride(1, 1)
                    padding(1, 1)
                    nOut(32)
                }
                batchNorm {
                    activation(Activation.SIGMOID)
                }
                conv2d {
                    kernelSize(3, 3)
                    stride(1, 1)
                    padding(1, 1)
                    nOut(128)
                }
                batchNorm {
                    activation(Activation.SIGMOID)
                }
                conv2d {
                    kernelSize(1, 1)
                    stride(1, 1)
                    padding(1, 1)
                    nOut(64)
                }
                batchNorm {
                    activation(Activation.SIGMOID)
                }
                conv2d {
                    kernelSize(1, 1)
                    stride(1, 1)
                    padding(1, 1)
                    nOut(numLabels)
                }
                batchNorm {
                    activation(Activation.SIGMOID)
                }
                subsampling {
                    kernelSize(2, 2)
                    stride(2, 2)
                    poolingType(SubsamplingLayer.PoolingType.AVG)
                }
                output {
                    name("output")
                    nOut(numLabels)
                    dropOut(0.8)
                    activation(Activation.SOFTMAX)
                    lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                }
                inputType = InputType.convolutional(height.toLong(), width.toLong(), channels.toLong())
            }
        }
        assertNetsEquals(dl4jNet, klayNet)
    }

    private fun assertNetsEquals(dl4jNet: MultiLayerConfiguration, klayNet: MultiLayerConfiguration) {
        val dl4jString = dl4jNet.toString()
        val klayString = klayNet.toString()
        assertEquals(dl4jString, klayString)
    }
}