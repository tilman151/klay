/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/
 * org/deeplearning4j/examples/utils/DownloaderUtility.java **/

package org.klay.examples.utils

import org.apache.commons.io.FilenameUtils
import org.nd4j.common.resources.Downloader
import java.io.File
import java.net.URL


/**
 * Given a base url and a zipped file name downloads contents to a specified directory under ~/dl4j-examples-data
 * Will check md5 sum of downloaded file
 *
 *
 *
 * Sample Usage with an instantiation DATAEXAMPLE(baseurl,"DataExamples.zip","data-dir",md5,size):
 *
 * DATAEXAMPLE.Download() & DATAEXAMPLE.Download(true)
 * Will download DataExamples.zip from baseurl/DataExamples.zip to a temp directory,
 * Unzip it to ~/dl4j-example-data/data-dir
 * Return the string "~/dl4j-example-data/data-dir/DataExamples"
 *
 * DATAEXAMPLE.Download(false)
 * will perform the same download and unzip as above
 * But returns the string "~/dl4j-example-data/data-dir" instead
 *
 *
 * @author susaneraly
 */
enum class DownloaderUtility
/**
 * Downloads a zip file from a base url to a specified directory under the user's home directory
 *
 * @param baseURL    URL of file
 * @param zipFile    Name of zipfile to download from baseURL i.e baseURL+"/"+zipFile gives full URL
 * @param dataFolder The folder to extract to under ~/dl4j-examples-data
 * @param md5        of zipfile
 * @param dataSize   of zipfile
 */(
    private val BASE_URL: String,
    private val ZIP_FILE: String,
    private val DATA_FOLDER: String,
    private val MD5: String,
    private val DATA_SIZE: String
) {
    IRISDATA("IrisData.zip", "datavec-examples", "bb49e38bb91089634d7ef37ad8e430b8", "1KB"), ANIMALS(
        "animals.zip",
        "dl4j-examples",
        "1976a1f2b61191d2906e4f615246d63e",
        "820KB"
    ),
    ANOMALYSEQUENCEDATA(
        "anomalysequencedata.zip",
        "dl4j-examples",
        "51bb7c50e265edec3a241a2d7cce0e73",
        "3MB"
    ),
    CAPTCHAIMAGE(
        "captchaImage.zip",
        "dl4j-examples",
        "1d159c9587fdbb1cbfd66f0d62380e61",
        "42MB"
    ),
    CLASSIFICATIONDATA("classification.zip", "dl4j-examples", "dba31e5838fe15993579edbf1c60c355", "77KB"), DATAEXAMPLES(
        "DataExamples.zip",
        "dl4j-examples",
        "e4de9c6f19aaae21fed45bfe2a730cbb",
        "2MB"
    ),
    LOTTERYDATA("lottery.zip", "dl4j-examples", "1e54ac1210e39c948aa55417efee193a", "2MB"), NEWSDATA(
        "NewsData.zip",
        "dl4j-examples",
        "0d08e902faabe6b8bfe5ecdd78af9f64",
        "21MB"
    ),
    NLPDATA(
        "nlp.zip",
        "dl4j-examples",
        "1ac7cd7ca08f13402f0e3b83e20c0512",
        "91MB"
    ),
    PREDICTGENDERDATA(
        "PredictGender.zip",
        "dl4j-examples",
        "42a3fec42afa798217e0b8687667257e",
        "3MB"
    ),
    STYLETRANSFER(
        "styletransfer.zip",
        "dl4j-examples",
        "b2b90834d667679d7ee3dfb1f40abe94",
        "3MB"
    ),
    VIDEOEXAMPLE("video.zip", "dl4j-examples", "56274eb6329a848dce3e20631abc6752", "8.5MB");

    /**
     * For use with resources uploaded to Azure blob storage.
     *
     * @param zipFile    Name of zipfile. Should be a zip of a single directory with the same name
     * @param dataFolder The folder to extract to under ~/dl4j-examples-data
     * @param md5        of zipfile
     * @param dataSize   of zipfile
     */
    constructor(
        zipFile: String,
        dataFolder: String,
        md5: String,
        dataSize: String
    ) : this(AZURE_BLOB_URL + "/" + dataFolder, zipFile, dataFolder, md5, dataSize) {
    }

    @JvmOverloads
    @Throws(Exception::class)
    fun Download(returnSubFolder: Boolean = true): String {
        val dataURL = "$BASE_URL/$ZIP_FILE"
        val downloadPath = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), ZIP_FILE)
        val extractDir = FilenameUtils.concat(
            System.getProperty("user.home"),
            "dl4j-examples-data/$DATA_FOLDER"
        )
        if (!File(extractDir).exists()) File(extractDir).mkdirs()
        var dataPathLocal = extractDir
        if (returnSubFolder) {
            val resourceName = ZIP_FILE.substring(0, ZIP_FILE.lastIndexOf(".zip"))
            dataPathLocal = FilenameUtils.concat(extractDir, resourceName)
        }
        val downloadRetries = 10
        if (!File(dataPathLocal).exists() || File(dataPathLocal).list().size == 0) {
            println("_______________________________________________________________________")
            println("Downloading data ($DATA_SIZE) and extracting to \n\t$dataPathLocal")
            println("_______________________________________________________________________")
            Downloader.downloadAndExtract(
                "files",
                URL(dataURL),
                File(downloadPath),
                File(extractDir),
                MD5,
                downloadRetries
            )
        } else {
            println("_______________________________________________________________________")
            println("Example data present in \n\t$dataPathLocal")
            println("_______________________________________________________________________")
        }
        return dataPathLocal
    }

    companion object {
        private const val AZURE_BLOB_URL = "https://dl4jdata.blob.core.windows.net/dl4j-examples"
    }
}