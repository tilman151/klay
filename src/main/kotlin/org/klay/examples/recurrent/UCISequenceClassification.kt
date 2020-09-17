/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/
 * deeplearning4j/examples/quickstart/modeling/recurrent/UCISequenceClassification.java **/

package org.klay.examples.recurrent

import org.apache.commons.io.FileUtils
import org.apache.commons.io.IOUtils
import org.datavec.api.records.reader.SequenceRecordReader
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.InvocationType
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.common.primitives.Pair
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.learning.config.Nadam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import java.io.File
import java.net.URL
import java.nio.charset.Charset
import java.util.*
import org.klay.nn.*


/**
 * Sequence Classification Example Using a LSTM Recurrent Neural Network
 *
 * This example learns how to classify univariate time series as belonging to one of six categories.
 * Categories are: Normal, Cyclic, Increasing trend, Decreasing trend, Upward shift, Downward shift
 *
 * Data is the UCI Synthetic Control Chart Time Series Data Set
 * Details:     https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series
 * Data:        https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data
 * Image:       https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/data.jpeg
 *
 * This example proceeds as follows:
 * 1. Download and prepare the data (in downloadUCIData() method)
 * (a) Split the 600 sequences into train set of size 450, and test set of size 150
 * (b) Write the data into a format suitable for loading using the CSVSequenceRecordReader for sequence classification
 * This format: one time series per file, and a separate file for the labels.
 * For example, train/features/0.csv is the features using with the labels file train/labels/0.csv
 * Because the data is a univariate time series, we only have one column in the CSV files. Normally, each column
 * would contain multiple values - one time step per row.
 * Furthermore, because we have only one label for each time series, the labels CSV files contain only a single value
 *
 * 2. Load the training data using CSVSequenceRecordReader (to load/parse the CSV files) and SequenceRecordReaderDataSetIterator
 * (to convert it to DataSet objects, ready to train)
 * For more details on this step, see: https://deeplearning4j.konduit.ai/models/recurrent#data-for-rnns
 *
 * 3. Normalize the data. The raw data contain values that are too large for effective training, and need to be normalized.
 * Normalization is conducted using NormalizerStandardize, based on statistics (mean, st.dev) collected on the training
 * data only. Note that both the training data and test data are normalized in the same way.
 *
 * 4. Configure the network
 * The data set here is very small, so we can't afford to use a large network with many parameters.
 * We are using one small LSTM layer and one RNN output layer
 *
 * 5. Train the network for 40 epochs
 * At each epoch, evaluate and print the accuracy and f1 on the test set
 *
 * @author Alex Black
 */
object UCISequenceClassification {
    private val log = LoggerFactory.getLogger(UCISequenceClassification::class.java)

    //'baseDir': Base directory for the data. Change this if you want to save the data somewhere else
    private val baseDir = File("src/main/resources/uci/")
    private val baseTrainDir = File(baseDir, "train")
    private val featuresDirTrain = File(baseTrainDir, "features")
    private val labelsDirTrain = File(baseTrainDir, "labels")
    private val baseTestDir = File(baseDir, "test")
    private val featuresDirTest = File(baseTestDir, "features")
    private val labelsDirTest = File(baseTestDir, "labels")
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        downloadUCIData()

        // ----- Load the training data -----
        //Note that we have 450 training files for features: train/features/0.csv through train/features/449.csv
        val trainFeatures: SequenceRecordReader = CSVSequenceRecordReader()
        trainFeatures.initialize(NumberedFileInputSplit(featuresDirTrain.absolutePath + "/%d.csv", 0, 449))
        val trainLabels: SequenceRecordReader = CSVSequenceRecordReader()
        trainLabels.initialize(NumberedFileInputSplit(labelsDirTrain.absolutePath + "/%d.csv", 0, 449))
        val miniBatchSize = 10
        val numLabelClasses = 7
        val trainData: DataSetIterator = SequenceRecordReaderDataSetIterator(
            trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END
        )

        //Normalize the training data
        val normalizer: DataNormalization = NormalizerStandardize()
        normalizer.fit(trainData) //Collect training data statistics
        trainData.reset()

        //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
        trainData.preProcessor = normalizer


        // ----- Load the test data -----
        //Same process as for the training data.
        val testFeatures: SequenceRecordReader = CSVSequenceRecordReader()
        testFeatures.initialize(NumberedFileInputSplit(featuresDirTest.absolutePath + "/%d.csv", 0, 149))
        val testLabels: SequenceRecordReader = CSVSequenceRecordReader()
        testLabels.initialize(NumberedFileInputSplit(labelsDirTest.absolutePath + "/%d.csv", 0, 149))
        val testData: DataSetIterator = SequenceRecordReaderDataSetIterator(
            testFeatures, testLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END
        )
        testData.preProcessor =
            normalizer //Note that we are using the exact same normalization process as the training data


        // ----- Configure the network -----
        val conf = sequential {
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
                    lossFunction(LossFunctions.LossFunction.MCXENT)
                    activation(Activation.SOFTMAX)
                    nIn(10)
                    nOut(numLabelClasses)
                }
            }
        }
        val net = MultiLayerNetwork(conf)
        net.init()
        log.info("Starting training...")
        net.setListeners(
            ScoreIterationListener(20),
            EvaluativeListener(testData, 1, InvocationType.EPOCH_END)
        ) //Print the score (loss function value) every 20 iterations
        val nEpochs = 40
        net.fit(trainData, nEpochs)
        log.info("Evaluating...")
        val eval = net.evaluate<Evaluation>(testData)
        log.info(eval.stats())
        log.info("----- Example Complete -----")
    }

    //This method downloads the data, and converts the "one time series per line" format into a suitable
    //CSV sequence format that DataVec (CsvSequenceRecordReader) and DL4J can read.
    @Throws(Exception::class)
    private fun downloadUCIData() {
        if (baseDir.exists()) return  //Data already exists, don't download it again
        val url =
            "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data"
        val data = IOUtils.toString(URL(url), null as Charset?)
        val lines = data.split("\n").toTypedArray()

        //Create directories
        baseDir.mkdir()
        baseTrainDir.mkdir()
        featuresDirTrain.mkdir()
        labelsDirTrain.mkdir()
        baseTestDir.mkdir()
        featuresDirTest.mkdir()
        labelsDirTest.mkdir()
        val contentAndLabels: MutableList<Pair<String, Int>?> = ArrayList()
        for ((lineCount, line) in lines.withIndex()) {
            val transposed = line.replace(" +".toRegex(), "\n")

            //Labels: first 100 quickstartexamples (lines) are label 0, second 100 quickstartexamples are label 1, and so on
            contentAndLabels.add(Pair(transposed, lineCount / 100))
        }

        //Randomize and do a train/test split:
        contentAndLabels.shuffle(Random(12345))
        val nTrain = 450 //75% train, 25% test
        var trainCount = 0
        var testCount = 0
        for (p in contentAndLabels) {
            //Write output in a format we can read, in the appropriate locations
            var outPathFeatures: File
            var outPathLabels: File
            if (trainCount < nTrain) {
                outPathFeatures = File(featuresDirTrain, "$trainCount.csv")
                outPathLabels = File(labelsDirTrain, "$trainCount.csv")
                trainCount++
            } else {
                outPathFeatures = File(featuresDirTest, "$testCount.csv")
                outPathLabels = File(labelsDirTest, "$testCount.csv")
                testCount++
            }
            FileUtils.writeStringToFile(outPathFeatures, p!!.first, null as Charset?)
            FileUtils.writeStringToFile(outPathLabels, p.second.toString(), null as Charset?)
        }
    }
}