/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/
 * deeplearning4j/examples/quickstart/modeling/convolution/CIFARClassifier.java **/

package org.klay.examples.convolution

import org.datavec.image.loader.CifarLoader
import org.deeplearning4j.core.storage.StatsStorage
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.InvocationType
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.model.stats.StatsListener
import org.deeplearning4j.ui.model.storage.FileStatsStorage
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.AdaDelta
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.io.File
import kotlin.system.exitProcess
import org.klay.nn.*


/**
 * train model by cifar
 * identification unknown file
 *
 * @author wangfeng
 * @since June 7,2017
 */
//@Slf4j
open class CIFARClassifier {
    val model: MultiLayerNetwork
        get() {
            log.info("Building simple convolutional network...")
            val conf = sequential {
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
                        lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    }
                    inputType = InputType.convolutional(height.toLong(), width.toLong(), channels.toLong())
                }
            }
            val model = MultiLayerNetwork(conf)
            model.init()
            return model
        }

    companion object {
        protected val log: Logger = LoggerFactory.getLogger(CIFARClassifier::class.java)
        private const val height = 32
        private const val width = 32
        private const val channels = 3
        private const val numLabels = CifarLoader.NUM_LABELS
        private const val batchSize = 96
        private const val seed = 123L
        private const val epochs = 4
        @Throws(Exception::class)
        @JvmStatic
        fun main(args: Array<String>) {
            val cf = CIFARClassifier()
            val cifar = Cifar10DataSetIterator(batchSize, intArrayOf(height, width), DataSetType.TRAIN, null, seed)
            val cifarEval = Cifar10DataSetIterator(batchSize, intArrayOf(height, width), DataSetType.TEST, null, seed)

            //train model and eval model
            val model = cf.model
            val uiServer: UIServer = UIServer.getInstance()
            val statsStorage: StatsStorage =
                FileStatsStorage(File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"))
            uiServer.attach(statsStorage)
            model.setListeners(
                StatsListener(statsStorage),
                ScoreIterationListener(50),
                EvaluativeListener(cifarEval, 1, InvocationType.EPOCH_END)
            )
            model.fit(cifar, epochs)
            log.info("Saving model...")
            model.save(File(System.getProperty("java.io.tmpdir"), "cifarmodel.dl4j.zip"), true)
            exitProcess(0)
        }
    }
}