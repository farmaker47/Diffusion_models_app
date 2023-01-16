package com.example.diffusionmodelsapp.ui.main

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import com.chaquo.python.Python
import org.tensorflow.lite.Interpreter
import java.io.File
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
    private val interpreterEncoder: Interpreter

    //private val interpreterDiffusion: Interpreter
    private val interpreterDecoder: Interpreter

    init {
        // Interpreter
        interpreterEncoder = getInterpreter(context, ENCODER_MODEL, false)
        //interpreterDiffusion = getInterpreter(context, DIFFUSION_MODEL, false)
        interpreterDecoder = getInterpreter(context, DECODER_MODEL, false)
    }

    companion object {
        private const val TAG = "DiffusionExecutor"

        private const val ENCODER_MODEL = "text_encoder_chollet_float_16.tflite"

        //private const val DIFFUSION_MODEL = "diffusion_model_17.tflite"
        private const val DECODER_MODEL = "decoder.tflite"
        private val intArrayOfPositions = intArrayOf(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76
        )
        private val unconditionalTokens = intArrayOf(
            49406,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407,
            49407
        )
    }

    // Function for Interpreter
    fun encoderExecutor(intArray: IntArray): Bitmap {
        try {
            Log.i(TAG, "running models")

            fullExecutionTime = SystemClock.uptimeMillis()

            // Info of the model
            /*val inputType = interpreterDecoder.getInputTensor(0).dataType()
            val inputName = interpreterDecoder.getInputTensor(0).name()
            val inputShape = interpreterDecoder.getInputTensor(0).shape()
            Log.i(TAG, "$inputType $inputName $inputShape")*/

            val decoderOutput = Array(1) {
                Array(1) {
                    Array(1) {
                        IntArray(3)
                    }
                }
            }
            /*val inputType1 = interpreterEncoder.getInputTensor(1).dataType()
            val inputName1 = interpreterEncoder.getInputTensor(1).name()
            val inputShape1 = interpreterEncoder.getInputTensor(1).shape()
            Log.i(TAG, "$inputType1 $inputName1 $inputShape1")

            val outputType = interpreterEncoder.getOutputTensor(0).dataType()
            val outputName = interpreterEncoder.getOutputTensor(0).name()
            val outputShape = interpreterEncoder.getOutputTensor(0).name()
            Log.i(TAG, "$outputType $outputName $outputShape")*/
            /*val outputType1 = interpreterPredict.getOutputTensor(1).dataType()
            val outputName1 = interpreterPredict.getOutputTensor(1).name()
            val outputShape1 = interpreterPredict.getOutputTensor(1).name()
            Log.i(TAG, "$outputType1 $outputName1 $outputShape1")*/
            val arrayOutputsContext = Array(1) {
                Array(77) {
                    FloatArray(768)
                }
            }
            val arrayOutputsUnconditionalContext = Array(1) {
                Array(77) {
                    FloatArray(768)
                }
            }

            //val signatures = interpreterEncoder.signatureKeys
            //Log.i(TAG, signatures[0].toString())

            // With [1,77] [1,77]
            val intArrayOftext = intArrayOf(
                49406,
                1237,
                3989,
                1960,
                2379,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407,
                49407
            )

            val contextInput = Array(1) {
                intArray // or intArrayOftext
            }
            val unconditionalContextInput = Array(1) {
                unconditionalTokens
            }
            val positionInput = Array(1) {
                intArrayOfPositions
            }

            // For context
            /*val inputs: MutableMap<Int, Any> = HashMap()
            inputs[0] = contextInput
            inputs[1] = positionInput*/
            val outputsContext: MutableMap<Int, Any> = HashMap()
            outputsContext[0] = arrayOutputsContext
            interpreterEncoder.runForMultipleInputsOutputs(
                arrayOf<Any>(
                    contextInput,
                    positionInput
                ), outputsContext
            )
            // For unconditional context
            /*val inputsUnconditional: MutableMap<Int, Any> = HashMap()
            inputsUnconditional[0] = unconditionalContextInput
            inputsUnconditional[1] = positionInput*/
            val outputsUnconditionalContext: MutableMap<Int, Any> = HashMap()
            outputsUnconditionalContext[0] = arrayOutputsUnconditionalContext
            interpreterEncoder.runForMultipleInputsOutputs(
                arrayOf<Any>(
                    unconditionalContextInput,
                    positionInput
                ), outputsUnconditionalContext
            )

            Log.i(TAG, "after running")

            fullExecutionTime = SystemClock.uptimeMillis() - fullExecutionTime

            //
            val python = Python.getInstance()
            val modelfile = python.getModule("run_diffusion_model")

            val diffusionResult = modelfile.callAttr(
                "runDiffusionModel",
                arrayOutputsContext,
                arrayOutputsUnconditionalContext
            ).toJava(Array<Array<Array<FloatArray>>>::class.java)

            Log.v("ChaquopyDiffusion", "diffusionResult.toString()")
            /*diffusionResult[0][0][0].forEach { first ->
                Log.v("Chaquopy", first.toString())
            }*/

            // Decoder
            /*val decoderResult = modelfile.callAttr(
                "runDecoderModel",
                diffusionResult
            ).toJava(Array<Array<Array<IntArray>>>::class.java)*/

            Log.v("Chaquopyy", "startDecoder.toString()")

            // Interpreter
            /*decoderResult[0][0][0].forEach { first ->
                Log.v("ChaquopyD", first.toString())
            }*/
            //interpreterEncoder.close()
            interpreterEncoder.close()

            interpreterDecoder.run(
                diffusionResult, decoderOutput
            )

            Log.i(TAG, "Time to run everything: $fullExecutionTime")
            Log.i(TAG, "Context: ${arrayOutputsContext[0][76][767]}")
            Log.i(TAG, "Unconditional context: ${arrayOutputsUnconditionalContext[0][76][767]}")

            return convertArrayToBitmap(decoderOutput, 512, 512)
        } catch (e: Exception) {

            val exceptionLog = "something went wrong: ${e.message}"
            Log.e(TAG, exceptionLog)
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        }

    }


    private fun convertArrayToBitmap(
        imageArray: Array<Array<Array<IntArray>>>,
        imageWidth: Int,
        imageHeight: Int
    ): Bitmap {
        val conf = Bitmap.Config.ARGB_8888 // see other conf types
        val bitmapImage = Bitmap.createBitmap(imageWidth, imageHeight, conf)

        for (x in imageArray[0].indices) {
            for (y in imageArray[0][0].indices) {
                val color = Color.rgb(
                    ((imageArray[0][x][y][0])),
                    ((imageArray[0][x][y][1])),
                    (imageArray[0][x][y][2])
                )

                // this y, x is in the correct order!!!
                bitmapImage.setPixel(y, x, color)
            }
        }
        return bitmapImage
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
        val mByteBuffer = loadModelFromInternalStorage(context, modelName)
        return Interpreter(mByteBuffer)//Interpreter(loadModelFile(context, modelName), tfliteOptions)
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

    private fun loadModelFromInternalStorage(
        context: Context,
        modelName: String
    ): MappedByteBuffer {
        val modelPath: String = context.filesDir.path + "/" + modelName
        val file = File(modelPath)
        val inputStream = FileInputStream(file)
        Log.v(TAG, file.length().toString())
        Log.v(TAG, Int.MAX_VALUE.toString())
        return inputStream.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }

    fun close() {
        //interpreterEncoder.close()
    }
}
