package com.example.diffusionmodelsapp.ui.main

import androidx.lifecycle.ViewModelProvider
import android.os.Bundle
import android.util.Log
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import com.chaquo.python.Python
import com.example.diffusionmodelsapp.R

class MainFragment : Fragment() {

    companion object {
        fun newInstance() = MainFragment()
    }

    private lateinit var viewModel: MainViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewModel = ViewModelProvider(this).get(MainViewModel::class.java)
        // TODO: Use the ViewModel
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val contentView = inflater.inflate(R.layout.fragment_main, container, false)

        // Get string from Python
        val python = Python.getInstance()
        val pythonFile = python.getModule("encode_text")
        //Log.v("Chaquopy", path + "heytj2.wav")
        val wavString = pythonFile.callAttr("getStringFromWav", "georg")
        Log.v("Chaquopy", wavString.toString())

        return contentView
    }

}