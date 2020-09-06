package org.klay.examples

import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.klay.examples.utils.DownloaderUtility
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.klay.examples.utils.PlotUtil.generatePointsOnGraph
import org.klay.examples.utils.PlotUtil.plotTestData
import org.klay.examples.utils.PlotUtil.plotTrainingData
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import java.io.File
import java.util.concurrent.TimeUnit
import org.klay.nn.*


/**
 * "Moon" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 */
object MoonClassifier {
    var visualize = true
    var dataLocalPath: String? = null
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val seed = 123
        val learningRate = 0.005
        val batchSize = 50
        val nEpochs = 100
        val numInputs = 2
        val numOutputs = 2
        val numHiddenNodes = 50
        dataLocalPath = DownloaderUtility.CLASSIFICATIONDATA.Download()

        //Load the training data:
        val rr: RecordReader = CSVRecordReader()
        rr.initialize(FileSplit(File(dataLocalPath, "moon_data_train.csv")))
        val trainIter: DataSetIterator = RecordReaderDataSetIterator(rr, batchSize, 0, 2)

        //Load the test/evaluation data:
        val rrTest: RecordReader = CSVRecordReader()
        rrTest.initialize(FileSplit(File(dataLocalPath, "moon_data_eval.csv")))
        val testIter: DataSetIterator = RecordReaderDataSetIterator(rrTest, batchSize, 0, 2)

        //log.info("Build model....");
        val conf = sequential {
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
        val model = MultiLayerNetwork(conf)
        model.init()
        model.setListeners(ScoreIterationListener(100)) //Print score every 100 parameter updates
        model.fit(trainIter, nEpochs)
        println("Evaluate model....")
        val eval = model.evaluate<Evaluation>(testIter)

        //Print the evaluation statistics
        println(eval.stats())
        println("\n****************Example finished********************")

        //Training is complete. Code that follows is for plotting the data & predictions only
        generateVisuals(model, trainIter, testIter)
    }

    @Throws(Exception::class)
    fun generateVisuals(model: MultiLayerNetwork?, trainIter: DataSetIterator?, testIter: DataSetIterator?) {
        if (visualize) {
            val xMin = -1.5
            val xMax = 2.5
            val yMin = -1.0
            val yMax = 1.5

            //Let's evaluate the predictions at every point in the x/y input space, and plot this in the background
            val nPointsPerAxis = 100

            //Generate x,y points that span the whole range of features
            val allXYPoints = generatePointsOnGraph(xMin, xMax, yMin, yMax, nPointsPerAxis)
            //Get train data and plot with predictions
            plotTrainingData(model!!, trainIter!!, allXYPoints, nPointsPerAxis)
            TimeUnit.SECONDS.sleep(3)
            //Get test data, run the test data through the network to generate predictions, and plot those predictions:
            plotTestData(model, testIter!!, allXYPoints, nPointsPerAxis)
        }
    }
}