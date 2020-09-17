package org.klay.examples.utils

import org.apache.commons.compress.archivers.tar.TarArchiveEntry
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.http.HttpEntity
import org.apache.http.client.methods.HttpGet
import org.apache.http.impl.client.CloseableHttpClient
import org.apache.http.impl.client.HttpClientBuilder
import java.io.*


/**
 * Common data utility functions.
 *
 * @author fvaleri
 */
object DataUtilities {
    /**
     * Download a remote file if it doesn't exist.
     * @param remoteUrl URL of the remote file.
     * @param localPath Where to download the file.
     * @return True if and only if the file has been downloaded.
     * @throws Exception IO error.
     */
    @Throws(IOException::class)
    fun downloadFile(remoteUrl: String?, localPath: String?): Boolean {
        var downloaded = false
        if (remoteUrl == null || localPath == null) return downloaded
        val file = File(localPath)
        if (!file.exists()) {
            file.parentFile.mkdirs()
            val builder: HttpClientBuilder = HttpClientBuilder.create()
            val client: CloseableHttpClient = builder.build()
            client.execute(HttpGet(remoteUrl)).use { response ->
                val entity: HttpEntity = response.entity
                FileOutputStream(file).use { outstream ->
                    entity.writeTo(outstream)
                    outstream.flush()
                    outstream.close()
                }
            }
            downloaded = true
        }
        if (!file.exists()) throw IOException("File doesn't exist: $localPath")
        return downloaded
    }

    /**
     * Extract a "tar.gz" file into a local folder.
     * @param inputPath Input file path.
     * @param outputPath Output directory path.
     * @throws IOException IO error.
     */
    @Throws(IOException::class)
    fun extractTarGz(inputPath: String?, outputPath: String?) {
        var path = outputPath
        if (inputPath == null || path == null) return
        val bufferSize = 4096
        if (!path.endsWith("" + File.separatorChar)) path += File.separatorChar
        TarArchiveInputStream(
            GzipCompressorInputStream(BufferedInputStream(FileInputStream(inputPath)))
        ).use { tais ->
            var entry: TarArchiveEntry?
            while (true) {
                entry = tais.nextTarEntry
                if (entry == null) {
                    break
                }
                if (entry.isDirectory) {
                    File(path + entry.name).mkdirs()
                } else {
                    var count: Int
                    val data = ByteArray(bufferSize)
                    val fos = FileOutputStream(path + entry.name)
                    val dest = BufferedOutputStream(fos, bufferSize)
                    while (tais.read(data, 0, bufferSize).also { count = it } != -1) {
                        dest.write(data, 0, count)
                    }
                    dest.close()
                }
            }
        }
    }
}