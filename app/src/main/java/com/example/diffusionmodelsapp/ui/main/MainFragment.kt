package com.example.diffusionmodelsapp.ui.main

import android.os.Bundle
import android.text.format.Formatter
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
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

        val runtime = Runtime.getRuntime()
        val maxMemory = runtime.maxMemory()
        Log.v("Chaquopy_max", maxMemory.toString())
        val usedMemory = runtime.totalMemory() - runtime.freeMemory()
        Log.v("Chaquopy_used", usedMemory.toString())
        val availableMemory = maxMemory - usedMemory
        Log.v("Chaquopy_avail", availableMemory.toString())
        val formattedMemorySize: String = Formatter.formatShortFileSize(context, availableMemory)
        Log.v("Chaquopy_avail_form", formattedMemorySize)

        // Get string from Python
        val python = Python.getInstance()
        val pythonFile = python.getModule("encode_text")
        //val encodedString = pythonFile.callAttr("encodeText", "two cats doing research")
        //Log.v("Chaquopy", encodedString.toString())
        val encodedObject: IntArray = pythonFile.callAttr("encodeText", "elephant on washing machine drinking soda").toJava(IntArray::class.java)

        val textView = contentView.findViewById<TextView>(R.id.message)
        textView.text = encodedObject[0].toString()

        val bitmap = viewModel.getResult(requireActivity(), encodedObject)
        val imageView = contentView.findViewById<ImageView>(R.id.imageViewBitmap)
        imageView.setImageBitmap(bitmap)

        return contentView
    }

}
