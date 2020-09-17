/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/
 * deeplearning4j/examples/quickstart/modeling/convolution/CenterLossLeNetMNIST.java **/

package org.klay.examples.convolution

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.klay.examples.utils.VAEPlotUtil
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.common.primitives.Pair
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import java.util.*
import org.klay.nn.*


/**
 * Example: training an embedding using the center loss model, on MNIST
 * The motivation is to use the class labels to learn embeddings that have the following properties:
 * (a) Intra-class similarity (i.e., similar vectors for same numbers)
 * (b) Inter-class dissimilarity (i.e., different vectors for different numbers)
 *
 * Refer to the paper "A Discriminative Feature Learning Approach for Deep Face Recognition", Wen et al. (2016)
 * http://ydwen.github.io/papers/WenECCV16.pdf
 *
 * This
 *
 * @author Alex Black
 */
object CenterLossLeNetMNIST {
    private val log = LoggerFactory.getLogger(CenterLossLeNetMNIST::class.java)
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val outputNum = 10 // The number of possible outcomes
        val batchSize = 64 // Test batch size
        val nEpochs = 10 // Number of training epochs
        val seed = 123

        //Lambda defines the relative strength of the center loss component.
        //lambda = 0.0 is equivalent to training with standard softmax only
        val lambda = 1.0

        //Alpha can be thought of as the learning rate for the centers for each class
        val alpha = 0.1
        log.info("Load data....")
        val mnistTrain: DataSetIterator = MnistDataSetIterator(batchSize, true, 12345)
        val mnistTest: DataSetIterator = MnistDataSetIterator(10000, false, 12345)
        log.info("Build model....")
        val conf = sequential {
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
                    lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    nIn(2).nOut(outputNum)
                    weightInit(WeightInit.XAVIER)
                    activation(Activation.SOFTMAX) //Alpha and lambda hyperparameters are specific to center loss model: see comments above and paper
                    alpha(alpha)
                    lambda(lambda)
                }
                inputType = InputType.convolutionalFlat(28, 28, 1)
            }
        }
        val model = MultiLayerNetwork(conf)
        model.init()
        log.info("Train model....")
        model.setListeners(ScoreIterationListener(100))
        val embeddingByEpoch: MutableList<Pair<INDArray, INDArray>> = ArrayList()
        val epochNum: MutableList<Int> = ArrayList()
        val testData = mnistTest.next()
        for (i in 0 until nEpochs) {
            model.fit(mnistTrain)
            log.info("*** Completed epoch {} ***", i)

            //Feed forward to the embedding layer (layer 5) to get the 2d embedding to plot later
            val embedding = model.feedForwardToLayer(5, testData.features)[6]
            embeddingByEpoch.add(Pair(embedding, testData.labels))
            epochNum.add(i)
        }

        //Create a scatterplot: slider allows embeddings to be view at the end of each epoch
        VAEPlotUtil.scatterPlot(embeddingByEpoch, epochNum, "MNIST Center Loss Embedding: l = $lambda, a = $alpha")
    }
}