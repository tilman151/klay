package org.klay

import junit.framework.TestCase.assertEquals
import org.datavec.image.loader.CifarLoader
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.distribution.UniformDistribution
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor
import org.deeplearning4j.nn.weights.WeightInit
import org.junit.Test
import org.klay.examples.recurrent.VideoFrameClassifier
import org.klay.nn.*
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.*
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.schedule.MapSchedule
import org.nd4j.linalg.schedule.ScheduleType
import java.util.ArrayList


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

    @Test
    fun centerLossLeNetMNISTExample() {
        val outputNum = 10
        val seed = 123
        val lambda = 1.0
        val alpha = 0.1

        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .l2(0.0005)
            .activation(Activation.LEAKYRELU)
            .weightInit(WeightInit.RELU)
            .updater(Adam(0.01))
            .list()
            .layer(ConvolutionLayer.Builder(5, 5).stride(1, 1).nOut(32).activation(Activation.LEAKYRELU).build())
            .layer(SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())
            .layer(ConvolutionLayer.Builder(5, 5).stride(1, 1).nOut(64).build())
            .layer(SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())
            .layer(
                DenseLayer.Builder().nOut(256).build()
            ) //Layer 5 is our embedding layer: 2 dimensions, just so we can plot it on X/Y grid. Usually use more in practice
            .layer(
                DenseLayer.Builder().activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER)
                    .nOut(2) //Larger L2 value on the embedding layer: can help to stop the embedding layer weights
                    // (and hence activations) from getting too large. This is especially problematic with small values of
                    // lambda such as 0.0
                    .l2(0.1).build()
            )
            .layer(
                CenterLossOutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(2).nOut(outputNum)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.SOFTMAX) //Alpha and lambda hyperparameters are specific to center loss model: see comments above and paper
                    .alpha(alpha).lambda(lambda)
                    .build()
            )
            .setInputType(InputType.convolutionalFlat(28, 28, 1))
            .build()

        val klayNet = sequential {
            seed(seed.toLong())
            l2(0.0005)
            activation(Activation.LEAKYRELU)
            weightInit(WeightInit.RELU)
            updater(Adam(0.01))
            layers {
                conv2d {
                    kernelSize(5, 5)
                    stride(1, 1)
                    nOut(32)
                }
                subsampling {
                    poolingType(SubsamplingLayer.PoolingType.MAX)
                    kernelSize(2, 2)
                    stride(2, 2)
                }
                conv2d {
                    kernelSize(5, 5)
                    stride(1, 1)
                    nOut(64)
                }
                subsampling {
                    poolingType(SubsamplingLayer.PoolingType.MAX)
                    kernelSize(2, 2)
                    stride(2, 2)
                }
                dense {
                    nOut(256)
                }
                dense {
                    activation(Activation.IDENTITY)
                    weightInit(WeightInit.XAVIER)
                    nOut(2)
                    l2(0.1)
                }
                centerLossOutput {
                    lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                    nOut(outputNum)
                    weightInit(WeightInit.XAVIER)
                    activation(Activation.SOFTMAX) //Alpha and lambda hyperparameters are specific to center loss model: see comments above and paper
                    alpha(alpha)
                    lambda(lambda)
                }
                inputType = InputType.convolutionalFlat(28, 28, 1)
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun leNetMNISTExample() {
        val nChannels = 1
        val outputNum = 10
        val seed = 123

        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .l2(0.0005)
            .weightInit(WeightInit.XAVIER)
            .updater(Adam(1e-3))
            .list()
            .layer(
                ConvolutionLayer.Builder(
                    5,
                    5
                ) //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                    .nIn(nChannels)
                    .stride(1, 1)
                    .nOut(20)
                    .activation(Activation.IDENTITY)
                    .build()
            )
            .layer(
                SubsamplingLayer.Builder(PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
            )
            .layer(
                ConvolutionLayer.Builder(5, 5) //Note that nIn need not be specified in later layers
                    .stride(1, 1)
                    .nOut(50)
                    .activation(Activation.IDENTITY)
                    .build()
            )
            .layer(
                SubsamplingLayer.Builder(PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
            )
            .layer(
                DenseLayer.Builder().activation(Activation.RELU)
                    .nOut(500).build()
            )
            .layer(
                OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .build()
            )
            .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
            .build()

        val klayNet = sequential {
            seed(seed.toLong())
            l2(0.0005)
            weightInit(WeightInit.XAVIER)
            updater(Adam(1e-3))
            layers {
                conv2d {
                    activation(Activation.IDENTITY)
                    kernelSize(5, 5)
                    stride(1, 1)
                    nIn(nChannels)
                    nOut(20)
                }
                subsampling {
                    poolingType(SubsamplingLayer.PoolingType.MAX)
                    kernelSize(2, 2)
                    stride(2, 2)
                }
                conv2d {
                    activation(Activation.IDENTITY)
                    kernelSize(5, 5)
                    stride(1, 1)
                    nOut(50)
                }
                subsampling {
                    poolingType(SubsamplingLayer.PoolingType.MAX)
                    kernelSize(2, 2)
                    stride(2, 2)
                }
                dense {
                    activation(Activation.RELU)
                    nOut(500)
                }
                output {
                    lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                    nOut(outputNum)
                    activation(Activation.SOFTMAX)
                }
                inputType = InputType.convolutionalFlat(28, 28, 1)
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun leNetMNISTReLuExample() {
        val height = 28L
        val width = 28L
        val channels = 1L
        val outputNum = 10
        val seed = 123
        val learningRateSchedule: MutableMap<Int, Double> = HashMap()
        learningRateSchedule[0] = 0.06
        learningRateSchedule[200] = 0.05
        learningRateSchedule[600] = 0.028
        learningRateSchedule[800] = 0.0060
        learningRateSchedule[1000] = 0.001

        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .l2(0.0005) // ridge regression value
            .updater(Nesterovs(MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(
                ConvolutionLayer.Builder(5, 5)
                    .nIn(channels)
                    .stride(1, 1)
                    .nOut(20)
                    .activation(Activation.IDENTITY)
                    .build()
            )
            .layer(
                SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
            )
            .layer(
                ConvolutionLayer.Builder(5, 5)
                    .stride(1, 1) // nIn need not specified in later layers
                    .nOut(50)
                    .activation(Activation.IDENTITY)
                    .build()
            )
            .layer(
                SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
            )
            .layer(
                DenseLayer.Builder().activation(Activation.RELU)
                    .nOut(500)
                    .build()
            )
            .layer(
                OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .build()
            )
            .setInputType(
                InputType.convolutionalFlat(
                    height,
                    width,
                    channels
                )
            ) // InputType.convolutional for normal image
            .build()

        val klayNet = sequential {
            seed(seed.toLong())
            l2(0.0005)
            weightInit(WeightInit.XAVIER)
            updater(Nesterovs(MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
            layers {
                conv2d {
                    activation(Activation.IDENTITY)
                    kernelSize(5, 5)
                    stride(1, 1)
                    nIn(channels)
                    nOut(20)
                }
                subsampling {
                    poolingType(SubsamplingLayer.PoolingType.MAX)
                    kernelSize(2, 2)
                    stride(2, 2)
                }
                conv2d {
                    activation(Activation.IDENTITY)
                    kernelSize(5, 5)
                    stride(1, 1)
                    nOut(50)
                }
                subsampling {
                    poolingType(SubsamplingLayer.PoolingType.MAX)
                    kernelSize(2, 2)
                    stride(2, 2)
                }
                dense {
                    activation(Activation.RELU)
                    nOut(500)
                }
                output {
                    lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                    nOut(outputNum)
                    activation(Activation.SOFTMAX)
                }
                inputType = InputType.convolutionalFlat(28, 28, 1)
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun memorizeSequenceExample() {
        val LEARNSTRING = "*Der Cottbuser Postkutscher putzt den Cottbuser Postkutschkasten.".toCharArray()
        val LEARNSTRING_CHARS_LIST: MutableList<Char> = ArrayList()
        val LEARNSTRING_CHARS = LinkedHashSet<Char>()
        for (c in LEARNSTRING) LEARNSTRING_CHARS.add(c)
        LEARNSTRING_CHARS_LIST.addAll(LEARNSTRING_CHARS)

        val HIDDEN_LAYER_WIDTH = 50
        val HIDDEN_LAYER_CONT = 2

        val builder = NeuralNetConfiguration.Builder()
        builder.seed(123)
        builder.biasInit(0.0)
        builder.miniBatch(false)
        builder.updater(RmsProp(0.001))
        builder.weightInit(WeightInit.XAVIER)
        val listBuilder = builder.list()

        // first difference, for rnns we need to use LSTM.Builder
        for (i in 0 until HIDDEN_LAYER_CONT) {
            val hiddenLayerBuilder = LSTM.Builder()
            hiddenLayerBuilder.nIn(if (i == 0) LEARNSTRING_CHARS.size else HIDDEN_LAYER_WIDTH)
            hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH)
            // adopted activation function from LSTMCharModellingExample
            // seems to work well with RNNs
            hiddenLayerBuilder.activation(Activation.TANH)
            listBuilder.layer(i, hiddenLayerBuilder.build())
        }

        // we need to use RnnOutputLayer for our RNN
        val outputLayerBuilder = RnnOutputLayer.Builder(LossFunction.MCXENT)
        // softmax normalizes the output neurons, the sum of all outputs is 1
        // this is required for our sampleFromDistribution-function
        outputLayerBuilder.activation(Activation.SOFTMAX)
        outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH)
        outputLayerBuilder.nOut(LEARNSTRING_CHARS.size)
        listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build())

        // create network
        val dl4jNet = listBuilder.build()

        val klayNet = sequential {
            seed(123)
            biasInit(0.0)
            miniBatch(false)
            updater(RmsProp(0.001))
            weightInit(WeightInit.XAVIER)
            layers {
                for (i in 0 until HIDDEN_LAYER_CONT) {
                    lstm {
                        nIn(if (i == 0) LEARNSTRING_CHARS.size else HIDDEN_LAYER_WIDTH)
                        nOut(HIDDEN_LAYER_WIDTH)
                        activation(Activation.TANH)
                    }
                }
                rnnOutput {
                    lossFunction(LossFunction.MCXENT)
                    activation(Activation.SOFTMAX)
                    nIn(HIDDEN_LAYER_WIDTH)
                    nOut(LEARNSTRING_CHARS.size)
                }
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun rnnEmbeddingExample() {
        val nClassesIn = 10

        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(123)
            .activation(Activation.RELU)
            .list()
            .layer(EmbeddingLayer.Builder().nIn(nClassesIn).nOut(5).build())
            .layer(LSTM.Builder().nIn(5).nOut(7).activation(Activation.TANH).build())
            .layer(
                RnnOutputLayer.Builder(LossFunction.MCXENT).nIn(7).nOut(4).activation(Activation.SOFTMAX)
                    .build()
            )
            .inputPreProcessor(0, RnnToFeedForwardPreProcessor())
            .inputPreProcessor(1, FeedForwardToRnnPreProcessor())
            .build()

        val klayNet = sequential {
            seed(123)
            activation(Activation.RELU)
            layers {
                embedding {
                    nIn(nClassesIn)
                    nOut(5)
                }
                lstm {
                    nIn(5)
                    nOut(7)
                    activation(Activation.TANH)
                }
                rnnOutput {
                    lossFunction(LossFunction.MCXENT)
                    nIn(7)
                    nOut(4)
                    activation(Activation.SOFTMAX)
                }
                inputPreProcessor(0, RnnToFeedForwardPreProcessor())
                inputPreProcessor(1, FeedForwardToRnnPreProcessor())
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun uciSequenceClassificationExample() {
        val numLabelClasses = 7

        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(123) //Random number generator seed for improved repeatability. Optional.
            .weightInit(WeightInit.XAVIER)
            .updater(Nadam())
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue) //Not always required, but helps with this data set
            .gradientNormalizationThreshold(0.5)
            .list()
            .layer(LSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
            .layer(
                RnnOutputLayer.Builder(LossFunction.MCXENT)
                    .activation(Activation.SOFTMAX).nIn(10).nOut(numLabelClasses).build()
            )
            .build()

        val klayNet = sequential {
            seed(123) //Random number generator seed for improved repeatability. Optional.
            weightInit(WeightInit.XAVIER)
            updater(Nadam())
            gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            gradientNormalizationThreshold(0.5)
            layers {
                lstm {
                    activation(Activation.TANH)
                    nIn(1)
                    nOut(10)
                }
                rnnOutput {
                    lossFunction(LossFunction.MCXENT)
                    activation(Activation.SOFTMAX)
                    nIn(10)
                    nOut(numLabelClasses)
                }
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun videoFrameClassifier() {
        val V_WIDTH = 130
        val V_HEIGHT = 130
        val V_NFRAMES = 150

        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(12345)
            .l2(0.001) //l2 regularization on all layers
            .updater(AdaGrad(0.04))
            .list()
            .layer(
                ConvolutionLayer.Builder(10, 10)
                    .nIn(3) //3 channels: RGB
                    .nOut(30)
                    .stride(4, 4)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.RELU)
                    .build()
            ) //Output: (130-10+0)/4+1 = 31 -> 31*31*30
            .layer(
                SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(3, 3)
                    .stride(2, 2).build()
            ) //(31-3+0)/2+1 = 15
            .layer(
                ConvolutionLayer.Builder(3, 3)
                    .nIn(30)
                    .nOut(10)
                    .stride(2, 2)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.RELU)
                    .build()
            ) //Output: (15-3+0)/2+1 = 7 -> 7*7*10 = 490
            .layer(
                DenseLayer.Builder()
                    .activation(Activation.RELU)
                    .nIn(490)
                    .nOut(50)
                    .weightInit(WeightInit.RELU)
                    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                    .gradientNormalizationThreshold(10.0)
                    .updater(AdaGrad(0.01))
                    .build()
            )
            .layer(
                LSTM.Builder()
                    .activation(Activation.TANH)
                    .nIn(50)
                    .nOut(50)
                    .weightInit(WeightInit.XAVIER)
                    .updater(AdaGrad(0.008))
                    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                    .gradientNormalizationThreshold(10.0)
                    .build()
            )
            .layer(
                RnnOutputLayer.Builder(LossFunction.MCXENT)
                    .activation(Activation.SOFTMAX)
                    .nIn(50)
                    .nOut(4) //4 possible shapes: circle, square, arc, line
                    .weightInit(WeightInit.XAVIER)
                    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                    .gradientNormalizationThreshold(10.0)
                    .build()
            )
            .inputPreProcessor(0, RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, 3))
            .inputPreProcessor(3, CnnToFeedForwardPreProcessor(7, 7, 10))
            .inputPreProcessor(4, FeedForwardToRnnPreProcessor())
            .backpropType(BackpropType.TruncatedBPTT)
            .tBPTTForwardLength(V_NFRAMES / 5)
            .tBPTTBackwardLength(V_NFRAMES / 5)
            .build()

        val klayNet = sequential {
            seed(12345)
            l2(0.001) //l2 regularization on all layers
            updater(AdaGrad(0.04))
            layers {
                conv2d {
                    kernelSize(10, 10)
                    nIn(3)
                    nOut(30)
                    stride(4, 4)
                    activation(Activation.RELU)
                    weightInit(WeightInit.RELU)
                }
                subsampling {
                    poolingType(SubsamplingLayer.PoolingType.MAX)
                    kernelSize(3, 3)
                    stride(2, 2)
                }
                conv2d {
                    kernelSize(3, 3)
                    nIn(30)
                    nOut(10)
                    stride(2, 2)
                    activation(Activation.RELU)
                    weightInit(WeightInit.RELU)
                }
                dense {
                    activation(Activation.RELU)
                    nIn(490)
                    nOut(50)
                    weightInit(WeightInit.RELU)
                    gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                    gradientNormalizationThreshold(10.0)
                    updater(AdaGrad(0.01))
                }
                lstm {
                    activation(Activation.TANH)
                    nIn(50)
                    nOut(50)
                    weightInit(WeightInit.XAVIER)
                    updater(AdaGrad(0.008))
                    gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                    gradientNormalizationThreshold(10.0)
                }
                rnnOutput {
                    lossFunction(LossFunction.MCXENT)
                    activation(Activation.SOFTMAX)
                    nIn(50)
                    nOut(4) //4 possible shapes: circle, square, arc, line
                    weightInit(WeightInit.XAVIER)
                    gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                    gradientNormalizationThreshold(10.0)
                }
                inputPreProcessor(0, RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, 3))
                inputPreProcessor(3, CnnToFeedForwardPreProcessor(7, 7, 10))
                inputPreProcessor(4, FeedForwardToRnnPreProcessor())
                backpropType(BackpropType.TruncatedBPTT)
                tBPTTForwardLength(V_NFRAMES / 5)
                tBPTTBackwardLength(V_NFRAMES / 5)
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