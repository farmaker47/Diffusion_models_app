package com.example.diffusionmodelsapp

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import com.chaquo.python.Python
import com.example.diffusionmodelsapp.ui.main.MainFragment

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        if (savedInstanceState == null) {
            supportFragmentManager.beginTransaction()
                .replace(R.id.container, MainFragment.newInstance())
                .commitNow()
        }

        // Get string from Python
        val python = Python.getInstance()
        val pythonFile = python.getModule("wav_to_string")
        //Log.v("Chaquopy", path + "heytj2.wav")
        val wavString = pythonFile.callAttr("getStringFromWav", "georg")
        Log.v("Chaquopy", wavString.toString())
    }
}