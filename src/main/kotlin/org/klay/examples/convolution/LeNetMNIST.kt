/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/
 * deeplearning4j/examples/quickstart/modeling/convolution/LeNetMNIST.java **/

package org.klay.examples.convolution

import org.apache.commons.io.FilenameUtils
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.InvocationType
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import java.io.File
import org.klay.nn.*


/**
 * Created by agibsonccc on 9/16/15.
 */
object LeNetMNIST {
    private val log = LoggerFactory.getLogger(LeNetMNIST::class.java)
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val nChannels = 1 // Number of input channels
        val outputNum = 10 // The number of possible outcomes
        val batchSize = 64 // Test batch size
        val nEpochs = 1 // Number of training epochs
        val seed = 123 //

        /*
            Create an iterator using the batch size for one iteration
         */log.info("Load data....")
        val mnistTrain: DataSetIterator = MnistDataSetIterator(batchSize, true, 12345)
        val mnistTest: DataSetIterator = MnistDataSetIterator(batchSize, false, 12345)

        /*
            Construct the neural network
         */log.info("Build model....")
        val conf = sequential {
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
                    lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    nOut(outputNum)
                    activation(Activation.SOFTMAX)
                }
                inputType = InputType.convolutionalFlat(28, 28, 1)
            }
        }

        /*
        Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
        (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
            and the dense layer
        (b) Does some additional configuration validation
        (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
            layer based on the size of the previous layer (but it won't override values manually set by the user)
        InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
        For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
        MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
        row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
        */
        val model = MultiLayerNetwork(conf)
        model.init()
        log.info("Train model...")
        model.setListeners(
            ScoreIterationListener(10),
            EvaluativeListener(mnistTest, 1, InvocationType.EPOCH_END)
        ) //Print score every 10 iterations and evaluate on test set every epoch
        model.fit(mnistTrain, nEpochs)
        val path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "lenetmnist.zip")
        log.info("Saving model to tmp folder: $path")
        model.save(File(path), true)
        log.info("****************Example finished********************")
    }
}