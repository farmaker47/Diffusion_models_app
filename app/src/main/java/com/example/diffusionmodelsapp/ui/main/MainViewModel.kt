package com.example.diffusionmodelsapp.ui.main

import android.content.Context
import android.graphics.Bitmap
import androidx.lifecycle.ViewModel

class MainViewModel : ViewModel() {

    fun getResult(context: Context, array: IntArray): Bitmap {
        val diffusionExecutor = DiffusionExecutor(context)
        return diffusionExecutor.encoderExecutor(array)
    }
}