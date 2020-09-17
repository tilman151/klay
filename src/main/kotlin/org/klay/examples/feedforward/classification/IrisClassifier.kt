package org.klay.examples.feedforward.classification

import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.klay.examples.utils.DownloaderUtility
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.LoggerFactory
import java.io.File
import org.klay.nn.*


/**
 * @author Adam Gibson
 */
object IrisClassifier {
    private val log = LoggerFactory.getLogger(IrisClassifier::class.java)
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        val numLinesToSkip = 0
        val delimiter = ','
        val recordReader: RecordReader = CSVRecordReader(numLinesToSkip, delimiter)
        recordReader.initialize(FileSplit(File(DownloaderUtility.IRISDATA.Download(), "iris.txt")))

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        val labelIndex =
            4 //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        val numClasses =
            3 //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        val batchSize =
            150 //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
        val iterator: DataSetIterator = RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses)
        val allData = iterator.next()
        allData.shuffle()
        val testAndTrain = allData.splitTestAndTrain(0.65) //Use 65% of data for training
        val trainingData = testAndTrain.train
        val testData = testAndTrain.test

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        val normalizer: DataNormalization = NormalizerStandardize()
        normalizer.fit(trainingData) //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData) //Apply normalization to the training data
        normalizer.transform(testData) //Apply normalization to the test data. This is using statistics calculated from the *training* set
        val numInputs = 4
        val outputNum = 3
        val seed: Long = 6
        log.info("Build model....")
        val conf = sequential {
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

        //run the model
        val model = MultiLayerNetwork(conf)
        model.init()
        //record score once every 100 iterations
        model.setListeners(ScoreIterationListener(100))
        for (i in 0..999) {
            model.fit(trainingData)
        }

        //evaluate the model on the test set
        val eval = Evaluation(3)
        val output = model.output(testData.features)
        eval.eval(testData.labels, output)
        log.info(eval.stats())
    }
}