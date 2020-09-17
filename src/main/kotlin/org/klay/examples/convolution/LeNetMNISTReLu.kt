/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/
 * deeplearning4j/examples/quickstart/modeling/convolution/LeNetMNISTReLu.java **/

package org.klay.examples.convolution

import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.klay.examples.utils.DataUtilities
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.schedule.MapSchedule
import org.nd4j.linalg.schedule.ScheduleType
import org.slf4j.LoggerFactory
import java.io.File
import java.util.*
import kotlin.collections.HashMap
import org.klay.nn.*


/**
 * Implementation of LeNet-5 for handwritten digits image classification on MNIST dataset (99% accuracy)
 * [[LeCun et al., 1998. Gradient based learning applied to document recognition]](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
 * Some minor changes are made to the architecture like using ReLU and identity activation instead of
 * sigmoid/tanh, max pooling instead of avg pooling and softmax output layer.
 *
 *
 * This example will download 15 Mb of data on the first run.
 *
 * @author hanlon
 * @author agibsonccc
 * @author fvaleri
 * @author dariuszzbyrad
 */
object LeNetMNISTReLu {
    private val LOGGER = LoggerFactory.getLogger(LeNetMNISTReLu::class.java)
    private val BASE_PATH = System.getProperty("java.io.tmpdir") + "/mnist"
    private const val DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val height = 28L // height of the picture in px
        val width = 28L // width of the picture in px
        val channels = 1L // single channel for grayscale images
        val outputNum = 10 // 10 digits classification
        val batchSize = 54 // number of samples that will be propagated through the network in each iteration
        val nEpochs = 1 // number of training epochs
        val seed = 1234 // number used to initialize a pseudorandom number generator.
        val randNumGen = Random(seed.toLong())
        LOGGER.info("Data load...")
        if (!File("$BASE_PATH/mnist_png").exists()) {
            LOGGER.debug("Data downloaded from {}", DATA_URL)
            val localFilePath = "$BASE_PATH/mnist_png.tar.gz"
            if (DataUtilities.downloadFile(DATA_URL, localFilePath)) {
                DataUtilities.extractTarGz(localFilePath, BASE_PATH)
            }
        }
        LOGGER.info("Data vectorization...")
        // vectorization of train data
        val trainData = File("$BASE_PATH/mnist_png/training")
        val trainSplit = FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
        val labelMaker = ParentPathLabelGenerator() // use parent directory name as the image label
        val trainRR = ImageRecordReader(height, width, channels, labelMaker)
        trainRR.initialize(trainSplit)
        val trainIter: DataSetIterator = RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum)

        // pixel values from 0-255 to 0-1 (min-max scaling)
        val imageScaler: DataNormalization = ImagePreProcessingScaler()
        imageScaler.fit(trainIter)
        trainIter.preProcessor = imageScaler

        // vectorization of test data
        val testData = File("$BASE_PATH/mnist_png/testing")
        val testSplit = FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
        val testRR = ImageRecordReader(height, width, channels, labelMaker)
        testRR.initialize(testSplit)
        val testIter: DataSetIterator = RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum)
        testIter.preProcessor = imageScaler // same normalization for better results
        LOGGER.info("Network configuration and training...")
        // reduce the learning rate as the number of training epochs increases
        // iteration #, learning rate
        val learningRateSchedule: MutableMap<Int, Double> = HashMap()
        learningRateSchedule[0] = 0.06
        learningRateSchedule[200] = 0.05
        learningRateSchedule[600] = 0.028
        learningRateSchedule[800] = 0.0060
        learningRateSchedule[1000] = 0.001
        val conf = sequential {
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
        val net = MultiLayerNetwork(conf)
        net.init()
        net.setListeners(ScoreIterationListener(10))
        LOGGER.info("Total num of params: {}", net.numParams())

        // evaluation while training (the score should go down)
        for (i in 0 until nEpochs) {
            net.fit(trainIter)
            LOGGER.info("Completed epoch {}", i)
            val eval = net.evaluate<Evaluation>(testIter)
            LOGGER.info(eval.stats())
            trainIter.reset()
            testIter.reset()
        }
        val ministModelPath = File("$BASE_PATH/minist-model.zip")
        ModelSerializer.writeModel(net, ministModelPath, true)
        LOGGER.info("The MINIST model has been saved in {}", ministModelPath.path)
    }
}