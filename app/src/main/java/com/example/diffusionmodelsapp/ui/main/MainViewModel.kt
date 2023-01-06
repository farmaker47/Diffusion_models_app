package com.example.diffusionmodelsapp.ui.main

import android.content.Context
import androidx.lifecycle.ViewModel

class MainViewModel : ViewModel() {


    fun getEncoderResult(context: Context, array: Array<Int>) {
        val diffusionExecutor = DiffusionExecutor(context)
        diffusionExecutor.encoderExecutor(array)
    }
}