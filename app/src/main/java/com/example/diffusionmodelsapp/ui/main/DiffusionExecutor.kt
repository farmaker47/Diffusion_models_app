package com.example.diffusionmodelsapp.ui.main

import android.content.Context
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class DiffusionExecutor(
    context: Context,
    private var useGPU: Boolean = false
) {

    private var numberThreads = 7
    private var fullExecutionTime = 0L
    private val interpreterPredict: Interpreter

    init {
        // Interpreter
        interpreterPredict = getInterpreter(context, ENCODER_MODEL, false)
    }

    companion object {
        private const val TAG = "DiffusionExecutor"

        private const val ENCODER_MODEL = "text_encoder_float16.tflite"
    }

    // Function for Interpreter
    fun encoderExecutor(array: Array<Int>) {
        try {
            Log.i(TAG, "running models")

            fullExecutionTime = SystemClock.uptimeMillis()

            // Info of the model
            val inputType = interpreterPredict.getInputTensor(0).dataType()
            val inputName = interpreterPredict.getInputTensor(0).name()
            val inputShape = interpreterPredict.getInputTensor(0).shape()

            val outputName = interpreterPredict.getOutputTensor(0).name()
            var arrayOutputsContext = Array(1) {
                Array(77) {
                    arrayOfNulls<Float>(768)
                }
            }
            var arrayOutputsUnconditionalContext = Array(1) {
                Array(77) {
                    arrayOfNulls<Float>(768)
                }
            }

            val signatures = interpreterPredict.signatureKeys
            Log.i(TAG, signatures.toString())

            val inputs: MutableMap<String, Any> = HashMap()
            inputs["tokens"] = arrayOf(array)
            inputs["batch_size"] = arrayOf(1)
            val outputs: MutableMap<String, Any> = HashMap()
            outputs["context"] = arrayOutputsContext
            outputs["unconditional_context"] = arrayOutputsUnconditionalContext

            interpreterPredict.runSignature(
                inputs, outputs, signatures[0]
            )
            //interpreterPredict.run(tImage.buffer, arrayOutputs)

            Log.i(TAG, "after running")

            fullExecutionTime = SystemClock.uptimeMillis() - fullExecutionTime

            Log.i(TAG, "Time to run everything: $fullExecutionTime")

            //return arrayOutputs[0]

        } catch (e: Exception) {

            val exceptionLog = "something went wrong: ${e.message}"
            Log.e("EXECUTOR", exceptionLog)

            //return longArrayOf()
        }

    }

    @Throws(IOException::class)
    private fun getInterpreter(
        context: Context,
        modelName: String,
        useGpu: Boolean = false
    ): Interpreter {
        val tfliteOptions = Interpreter.Options()

        tfliteOptions.numThreads = numberThreads

        //tfliteOptions.setUseNNAPI(true)     //846ms
        //tfliteOptions.setUseXNNPACK(true) //     Caused by: java.lang.IllegalArgumentException: Internal error: Failed to apply XNNPACK delegate:
        //     Attempting to use a delegate that only supports static-sized tensors with a graph that has dynamic-sized tensors.

        return Interpreter(loadModelFile(context, modelName), tfliteOptions)
    }

    @Throws(IOException::class)
    private fun loadModelFile(context: Context, modelFile: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelFile)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val retFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        fileDescriptor.close()
        return retFile
    }

    fun close() {
        interpreterPredict.close()
    }
}
