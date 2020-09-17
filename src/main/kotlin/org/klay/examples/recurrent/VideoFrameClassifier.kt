/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/
 * deeplearning4j/examples/quickstart/modeling/recurrent/VideoFrameClassifier.java **/

package org.klay.examples.recurrent

import org.apache.commons.io.FileUtils
import org.datavec.api.conf.Configuration
import org.datavec.api.records.reader.SequenceRecordReader
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.InputSplit
import org.datavec.api.split.NumberedFileInputSplit
import org.datavec.codec.reader.NativeCodecRecordReader
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.klay.examples.utils.DownloaderUtility
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.AsyncDataSetIterator
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.AdaGrad
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import java.io.IOException
import java.nio.charset.Charset
import org.klay.nn.*


/**
 * Example: Combine convolutional, max pooling, dense (feed forward) and recurrent (LSTM) layers to classify each
 * frame of a video (using a generated/synthetic video data set)
 * Specifically, each video contains a shape (randomly selected: circles, squares, lines, arcs) which persist for
 * multiple frames (though move between frames) and may leave the frame. Each video contains multiple shapes which
 * are shown for some random number of frames.
 * The network needs to classify these shapes, even when the shape has left the frame.
 *
 * This example is somewhat contrived, but shows data import and network configuration for classifying video frames.
 * The data for this example is automatically downloaded to:
 * "~/dl4j-examples-data/dl4j-examples/video/videoshapesexample"
 * *******************************************************
 * @author Alex Black
 */
object VideoFrameClassifier {
    private const val N_VIDEOS = 500
    private const val V_WIDTH = 130
    private const val V_HEIGHT = 130
    private const val V_NFRAMES = 150
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {

        // Note that you will need to run with at least 7G off heap memory
        // if you want to keep this batchsize and train the nn config specified
        val miniBatchSize = 16
        val dataDirectory = DownloaderUtility.VIDEOEXAMPLE.Download() + "/videoshapesexample/"

        //Set up network architecture:
        val conf = sequential {
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
                    lossFunction(LossFunctions.LossFunction.MCXENT)
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
        val net = MultiLayerNetwork(conf)
        net.init()

        // summary of layer and parameters
        println(net.summary())
        val testStartIdx = (0.9 * N_VIDEOS).toInt() //90% in train, 10% in test
        val nTest = N_VIDEOS - testStartIdx

        //Conduct learning
        println("Starting training...")
        net.setListeners(ScoreIterationListener(1))
        val nTrainEpochs = 15
        for (i in 0 until nTrainEpochs) {
            val trainData = getDataSetIterator(dataDirectory, 0, testStartIdx - 1, miniBatchSize)
            while (trainData.hasNext()) net.fit(trainData.next())
            Nd4j.saveBinary(net.params(), File("videomodel.bin"))
            FileUtils.writeStringToFile(File("videoconf.json"), conf.toJson(), null as Charset?)
            println("Epoch $i complete")

            //Evaluate classification performance:
            evaluatePerformance(net, testStartIdx, nTest, dataDirectory)
        }
    }

    @Throws(Exception::class)
    private fun evaluatePerformance(
        net: MultiLayerNetwork,
        testStartIdx: Int,
        nExamples: Int,
        outputDirectory: String
    ) {
        //Assuming here that the full test data set doesn't fit in memory -> load 10 examples at a time
        val labelMap: MutableMap<Int, String> = HashMap()
        labelMap[0] = "circle"
        labelMap[1] = "square"
        labelMap[2] = "arc"
        labelMap[3] = "line"
        val evaluation = Evaluation(labelMap)
        val testData = getDataSetIterator(outputDirectory, testStartIdx, nExamples, 10)
        while (testData.hasNext()) {
            val dsTest = testData.next()
            val predicted = net.output(dsTest.features, false)
            val actual = dsTest.labels
            evaluation.eval(actual, predicted)
        }
        println(evaluation.stats())
    }

    @Throws(Exception::class)
    private fun getDataSetIterator(
        dataDirectory: String,
        startIdx: Int,
        nExamples: Int,
        miniBatchSize: Int
    ): DataSetIterator {
        //Here, our data and labels are in separate files
        //videos: shapes_0.mp4, shapes_1.mp4, etc
        //labels: shapes_0.txt, shapes_1.txt, etc. One time step per line
        val featuresTrain = getFeaturesReader(dataDirectory, startIdx, nExamples)
        val labelsTrain = getLabelsReader(dataDirectory, startIdx, nExamples)
        val sequenceIter = SequenceRecordReaderDataSetIterator(featuresTrain, labelsTrain, miniBatchSize, 4, false)
        sequenceIter.preProcessor = VideoPreProcessor()

        //AsyncDataSetIterator: Used to (pre-load) load data in a separate thread
        return AsyncDataSetIterator(sequenceIter, 1)
    }

    @Throws(IOException::class, InterruptedException::class)
    private fun getFeaturesReader(path: String, startIdx: Int, num: Int): SequenceRecordReader {
        //InputSplit is used here to define what the file paths look like
        val `is`: InputSplit = NumberedFileInputSplit(path + "shapes_%d.mp4", startIdx, startIdx + num - 1)
        val conf = Configuration()
        conf[NativeCodecRecordReader.RAVEL] = "true"
        conf[NativeCodecRecordReader.START_FRAME] = "0"
        conf[NativeCodecRecordReader.TOTAL_FRAMES] = V_NFRAMES.toString()
        conf[NativeCodecRecordReader.ROWS] = V_WIDTH.toString()
        conf[NativeCodecRecordReader.COLUMNS] = V_HEIGHT.toString()
        val crr = NativeCodecRecordReader()
        crr.initialize(conf, `is`)
        return crr
    }

    @Throws(Exception::class)
    private fun getLabelsReader(path: String, startIdx: Int, num: Int): SequenceRecordReader {
        val isLabels: InputSplit = NumberedFileInputSplit(path + "shapes_%d.txt", startIdx, startIdx + num - 1)
        val csvSeq = CSVSequenceRecordReader()
        csvSeq.initialize(isLabels)
        return csvSeq
    }

    private class VideoPreProcessor : DataSetPreProcessor {
        override fun preProcess(toPreProcess: DataSet) {
            toPreProcess.features.divi(255) //[0,255] -> [0,1] for input pixel values
        }
    }
}