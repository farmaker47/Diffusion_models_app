package com.example.diffusionmodelsapp

import android.app.Application
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class DiffusionModelsApp : Application() {

    override fun onCreate() {
        super.onCreate()
        // Start Python
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
    }
}
