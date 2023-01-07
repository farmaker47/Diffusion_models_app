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

        private const val ENCODER_MODEL = "text_encoder_chollet_float_16.tflite"
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
    fun encoderExecutor(intArray: IntArray) {
        try {
            Log.i(TAG, "running models")

            fullExecutionTime = SystemClock.uptimeMillis()

            // Info of the model
            val inputType = interpreterPredict.getInputTensor(0).dataType()
            val inputName = interpreterPredict.getInputTensor(0).name()
            val inputShape = interpreterPredict.getInputTensor(0).shape()
            Log.i(TAG, "$inputType $inputName $inputShape")
            val inputType1 = interpreterPredict.getInputTensor(1).dataType()
            val inputName1 = interpreterPredict.getInputTensor(1).name()
            val inputShape1 = interpreterPredict.getInputTensor(1).shape()
            Log.i(TAG, "$inputType1 $inputName1 $inputShape1")

            val outputType = interpreterPredict.getOutputTensor(0).dataType()
            val outputName = interpreterPredict.getOutputTensor(0).name()
            val outputShape = interpreterPredict.getOutputTensor(0).name()
            Log.i(TAG, "$outputType $outputName $outputShape")
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

            val signatures = interpreterPredict.signatureKeys
            Log.i(TAG, signatures[0].toString())

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
            interpreterPredict.runForMultipleInputsOutputs(arrayOf<Any>(contextInput, positionInput) , outputsContext)
            // For unconditional context
            /*val inputsUnconditional: MutableMap<Int, Any> = HashMap()
            inputsUnconditional[0] = unconditionalContextInput
            inputsUnconditional[1] = positionInput*/
            val outputsUnconditionalContext: MutableMap<Int, Any> = HashMap()
            outputsUnconditionalContext[0] = arrayOutputsUnconditionalContext
            interpreterPredict.runForMultipleInputsOutputs(arrayOf<Any>(unconditionalContextInput, positionInput), outputsUnconditionalContext)


            //val tokensBuffer = TensorBuffer.createDynamic(DataType.INT32)
            //val batchSizeBuffer = TensorBuffer.createDynamic(DataType.INT32)

            //val inputs: MutableMap<String, Any> = HashMap()
            // Method
            //val bytesTokens = array.foldIndexed(ByteArray(array.size)) { i, a, v -> a.apply { set(i, v.toByte()) } }
            //val bytesBatch = arrayOf(1).foldIndexed(ByteArray(1)) { i, a, v -> a.apply { set(i, v.toByte()) } }

            /*val byteBuffer = ByteBuffer.allocate(array.size * 4)
            val intBuffer: IntBuffer = byteBuffer.asIntBuffer()
            intBuffer.put(array)
            val tokensArray = byteBuffer.array()*/

            //val bytearray: ByteArray = IntArray.toByteArray(array)

            /////////////////////////////////////
            //////////////////////////////////// with Park [1,1,1], []
            /*val intArrayOftext = intArrayOf(49406, 1237, 3989, 1960, 2379, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407)
            val doubleInput = Array(1) {
                intArrayOftext
            }
            var arrayOutputsContext = Array(1) {
                Array(1) {
                    floatArrayOf(0f)
                }
            }
            var arrayOutputsUnconditionalContext = 1f

            val outputs: MutableMap<Int, Any> = HashMap()
            outputs[0] = arrayOutputsContext
            //outputs[1] = arrayOutputsUnconditionalContext

            val inputs = arrayOf<Any>(1, doubleInput)

            interpreterPredict.runForMultipleInputsOutputs(inputs, outputs)*/
            /////////////////////////////////////////
            /////////////////////////////////////////

            /*val tokensByteArray = convertToByteArray(intArrayOf(49406, 1237, 3989, 1960, 2379, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407))
            val doubleByteArray = Array(1) {
                tokensByteArray
            }*/
            /*val intArrayOftext = intArrayOf(49406, 1237, 3989, 1960, 2379, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407)
            val doubleInput = Array(1) {
                intArrayOftext
            }

            var arrayOutputsUnconditionalContext = Array(1) {
                Array(1) {
                    floatArrayOf(0f)
                }
            }
            var arrayOutputsContext = 0f

            val inputs: MutableMap<String, Any> = HashMap()
            inputs["tokens"] = doubleInput//ByteBuffer.wrap(tokensByteArray)//ByteBuffer.wrap(tokensArray)//ByteBuffer.wrap(bytesTokens)//tokensBuffer.loadArray(array)//arrayOf(array)
            inputs["batch_size"] = 1//ByteBuffer.wrap(ByteBuffer.allocate(Int.SIZE_BYTES).putInt(1).array())//ByteBuffer.wrap(convertToByteArray(intArrayOf(1)))//ByteBuffer.wrap(ByteBuffer.allocate(Int.SIZE_BYTES).putInt(1).array())
            val outputs: MutableMap<String, Any> = HashMap()
            outputs["context"] = arrayOutputsContext
            outputs["unconditional_context"] = arrayOutputsUnconditionalContext
            *//*val inputs: MutableMap<Int, Any> = HashMap()
            inputs[0] = doubleByteArray//ByteBuffer.wrap(tokensByteArray)//ByteBuffer.wrap(tokensArray)//ByteBuffer.wrap(bytesTokens)//tokensBuffer.loadArray(array)//arrayOf(array)
            inputs[1] = 1*//*//ByteBuffer.wrap(ByteBuffer.allocate(Int.SIZE_BYTES).putInt(1).array())//ByteBuffer.wrap(convertToByteArray(intArrayOf(1)))//ByteBuffer.wrap(ByteBuffer.allocate(Int.SIZE_BYTES).putInt(1).array())

            interpreterPredict.runSignature(
                inputs, outputs, signatures[0]
            )*/


            Log.i(TAG, "after running")

            fullExecutionTime = SystemClock.uptimeMillis() - fullExecutionTime

            Log.i(TAG, "Time to run everything: $fullExecutionTime")
            Log.i(TAG, "Context: ${arrayOutputsContext[0][76][767]}")
            Log.i(TAG, "Unconditional context: ${arrayOutputsUnconditionalContext[0][76][767]}")


        } catch (e: Exception) {

            val exceptionLog = "something went wrong: ${e.message}"
            Log.e(TAG, exceptionLog)

        }

    }

    /*fun byteArrayOfInts(vararg ints: Int) = ByteArray(ints.size) { pos -> ints[pos].toByte() }

    fun toByteArray(value: Int): ByteArray {
        val result = ByteArray(4)
        return Conversion.intToByteArray(value, 0, result, 0, 4)
    }*/

    /*fun convertToByteArray(pIntArray: IntArray): ByteArray {
        val array = ByteArray(pIntArray.size * 4)
        for (j in pIntArray.indices) {
            val c = pIntArray[j]
            array[j * 4] = (c and -0x1000000 shr 24).toByte()
            array[j * 4 + 1] = (c and 0xFF0000 shr 16).toByte()
            array[j * 4 + 2] = (c and 0xFF00 shr 8).toByte()
            array[j * 4 + 3] = (c and 0xFF).toByte()
        }
        return array
    }*/

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
