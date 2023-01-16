package com.example.diffusionmodelsapp.ui.main

import android.os.Bundle
import android.text.format.Formatter
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.ProgressBar
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.chaquo.python.Python
import com.example.diffusionmodelsapp.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

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

        val editText = contentView.findViewById<EditText>(R.id.message)
        val button = contentView.findViewById<Button>(R.id.buttonStart)
        val progress = contentView.findViewById<ProgressBar>(R.id.progressBar)
        button.setOnClickListener {
            progress.visibility = View.VISIBLE
            lifecycleScope.launch(Dispatchers.IO) {
                // Get string from Python
                val python = Python.getInstance()
                val pythonFile = python.getModule("encode_text")
                //val encodedString = pythonFile.callAttr("encodeText", "two cats doing research")
                //Log.v("Chaquopy", encodedString.toString())
                val encodedObject: IntArray =
                    pythonFile.callAttr("encodeText", editText.text.toString())
                        .toJava(IntArray::class.java)

                val bitmap = viewModel.getResult(requireActivity(), encodedObject)
                val imageView = contentView.findViewById<ImageView>(R.id.imageViewBitmap)
                withContext(Dispatchers.Main) {
                    imageView.setImageBitmap(bitmap)
                    progress.visibility = View.GONE
                }
            }

        }

        return contentView
    }

}
