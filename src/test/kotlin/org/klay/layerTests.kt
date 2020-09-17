package org.klay

import junit.framework.TestCase.*
import org.junit.*
import org.deeplearning4j.nn.conf.*
import org.deeplearning4j.nn.conf.layers.*
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.*
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import org.klay.nn.*

class LayerTests {
    @Test
    fun testDenseLayer() {
        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(42)
            .updater(Adam())
            .list()
            .layer(DenseLayer.Builder()
                .nIn(10)
                .nOut(100)
                .activation(Activation.RELU)
                .build())
            .layer(OutputLayer.Builder()
                .nOut(2)
                .activation(Activation.SOFTMAX)
                .build())
            .build()

        val klayNet = sequential {
            seed(42)
            updater(Adam())
            layers {
                dense {
                    nIn(10)
                    nOut(100)
                    activation(Activation.RELU)
                }
                output {
                    nOut(2)
                    activation(Activation.SOFTMAX)
                }
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun testCenterLossOutputLayer() {
        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(42)
            .updater(Adam())
            .list()
            .layer(DenseLayer.Builder()
                .nIn(10)
                .nOut(100)
                .activation(Activation.RELU)
                .build())
            .layer(CenterLossOutputLayer.Builder()
                .nOut(2)
                .activation(Activation.SOFTMAX)
                .build())
            .build()

        val klayNet = sequential {
            seed(42)
            updater(Adam())
            layers {
                dense {
                    nIn(10)
                    nOut(100)
                    activation(Activation.RELU)
                }
                centerLossOutput {
                    nOut(2)
                    activation(Activation.SOFTMAX)
                }
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun testLstmLayer() {
        val builder = NeuralNetConfiguration.Builder()
        builder.seed(123)
        builder.biasInit(0.0)
        builder.miniBatch(false)
        builder.updater(RmsProp(0.001))
        builder.weightInit(WeightInit.XAVIER)
        val listBuilder = builder.list()

        // first difference, for rnns we need to use LSTM.Builder
        val hiddenLayerBuilder = LSTM.Builder()
        hiddenLayerBuilder.nIn(16)
        hiddenLayerBuilder.nOut(32)
        // adopted activation function from LSTMCharModellingExample
        // seems to work well with RNNs
        hiddenLayerBuilder.activation(Activation.TANH)
        listBuilder.layer(0, hiddenLayerBuilder.build())

        // we need to use RnnOutputLayer for our RNN
        val outputLayerBuilder = RnnOutputLayer.Builder(LossFunction.MCXENT)
        // softmax normalizes the output neurons, the sum of all outputs is 1
        // this is required for our sampleFromDistribution-function
        outputLayerBuilder.activation(Activation.SOFTMAX)
        outputLayerBuilder.nIn(32)
        outputLayerBuilder.nOut(128)
        listBuilder.layer(1, outputLayerBuilder.build())

        // create network
        val dl4jNet = listBuilder.build()

        val klayNet = sequential {
            seed(123)
            biasInit(0.0)
            miniBatch(false)
            updater(RmsProp(0.001))
            weightInit(WeightInit.XAVIER)
            layers {
                lstm {
                    nIn(16)
                    nOut(32)
                    activation(Activation.TANH)
                }
                rnnOutput {
                    lossFunction(LossFunction.MCXENT)
                    activation(Activation.SOFTMAX)
                    nIn(32)
                    nOut(128)
                }
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun testConv2dLayer() {
        val dl4jNet = NeuralNetConfiguration.Builder()
                .seed(42)
                .updater(Adam())
                .list()
                .layer(ConvolutionLayer.Builder()
                        .nIn(10)
                        .nOut(100)
                        .kernelSize(3, 3)
                        .activation(Activation.RELU)
                        .build())
                .layer(OutputLayer.Builder()
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build()

        val klayNet = sequential {
            seed(42)
            updater(Adam())
            layers {
                conv2d {
                    nIn(10)
                    nOut(100)
                    kernelSize(3, 3)
                    activation(Activation.RELU)
                }
                output {
                    nOut(2)
                    activation(Activation.SOFTMAX)
                }
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun testSubsamplingLayer() {
        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(42)
            .updater(Adam())
            .list()
            .layer(ConvolutionLayer.Builder()
                .nIn(10)
                .nOut(100)
                .kernelSize(3, 3)
                .activation(Activation.RELU)
                .build())
            .layer(SubsamplingLayer.Builder()
                .kernelSize(2, 2)
                .stride(2, 2)
                .poolingType(SubsamplingLayer.PoolingType.MAX)
                .build()
            )
            .layer(OutputLayer.Builder()
                .nOut(2)
                .activation(Activation.SOFTMAX)
                .build())
            .build()

        val klayNet = sequential {
            seed(42)
            updater(Adam())
            layers {
                conv2d {
                    nIn(10)
                    nOut(100)
                    kernelSize(3, 3)
                    activation(Activation.RELU)
                }
                subsampling {
                    kernelSize(2, 2)
                    stride(2, 2)
                    poolingType(SubsamplingLayer.PoolingType.MAX)
                }
                output {
                    nOut(2)
                    activation(Activation.SOFTMAX)
                }
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun testBatchNormLayer() {
        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(42)
            .updater(Adam())
            .list()
            .layer(ConvolutionLayer.Builder()
                .nIn(10)
                .nOut(100)
                .kernelSize(3, 3)
                .activation(Activation.RELU)
                .build())
            .layer(BatchNormalization.Builder().build())
            .layer(OutputLayer.Builder()
                .nOut(2)
                .activation(Activation.SOFTMAX)
                .build())
            .build()

        val klayNet = sequential {
            seed(42)
            updater(Adam())
            layers {
                conv2d {
                    nIn(10)
                    nOut(100)
                    kernelSize(3, 3)
                    activation(Activation.RELU)
                }
                batchNorm {  }
                output {
                    nOut(2)
                    activation(Activation.SOFTMAX)
                }
            }
        }

        assertNetsEquals(dl4jNet, klayNet)
    }

    @Test
    fun testLoopInConfig() {
        val numRows = 28
        val numColumns = 28
        val outputNum = 10 // number of output classes
        val rngSeed = 123 // random number seed for reproducibility
        val rate = 0.0015 // learning rate

        val units = listOf(200, 200, 300, 400)

        val unfinished = NeuralNetConfiguration.Builder()
                .seed(rngSeed.toLong())
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Nesterovs(rate, 0.98))
                .l2(rate * 0.005)
                .list()
                .layer(DenseLayer.Builder()
                        .nIn(numRows * numColumns)
                        .nOut(units[0])
                        .build())

        for (i in 1 until units.size) {
            unfinished.layer(DenseLayer.Builder()
                    .nIn(units[i-1])
                    .nOut(units[i])
                    .build())
        }

        val dl4jNet = unfinished.layer(OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(units.last())
                .nOut(outputNum)
                .build())
                .build()

        val klayNet = sequential {
            seed(rngSeed.toLong())
            activation(Activation.RELU)
            weightInit(WeightInit.XAVIER)
            updater(Nesterovs(rate, 0.98))
            l2(rate * 0.005)
            layers {
                dense {
                    nIn(numRows * numColumns)
                    nOut(units[0])
                }
                for (i in 1 until units.size) {
                    dense {
                        nIn(units[i-1])
                        nOut(units[i])
                    }
                }
                output {
                    lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                    activation(Activation.SOFTMAX)
                    nIn(units.last())
                    nOut(outputNum)
                }
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