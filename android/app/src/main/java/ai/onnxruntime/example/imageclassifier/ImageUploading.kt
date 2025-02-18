package ai.onnxruntime.example.imageclassifier

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import android.graphics.BitmapFactory
import ai.onnxruntime.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.lifecycle.lifecycleScope
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class ImageUploading : AppCompatActivity() {

    private lateinit var btnChooseImage: Button
    private lateinit var imageView: ImageView
    private lateinit var textViewPredictedClass: TextView
    private lateinit var textViewProbability: TextView
    private lateinit var btnBackToMainUpload: Button

    private var ortEnv: OrtEnvironment? = null

    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }

    val utils_class = Utils(this)
    private val labelData: List<String> by lazy { utils_class.readLabels() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.image_uploading)
        ortEnv = OrtEnvironment.getEnvironment()

        btnChooseImage = findViewById(R.id.btnChooseImage)
        imageView = findViewById(R.id.imageView)
        textViewPredictedClass = findViewById(R.id.textViewPredictedClass)
        textViewProbability = findViewById(R.id.textViewProbability)
        btnBackToMainUpload = findViewById(R.id.btnBackToMainUpload)

        btnChooseImage.setOnClickListener {
            pickImageResult.launch("image/*")
        }

        btnBackToMainUpload.setOnClickListener {
            val intent = Intent(this, FirstScreenActivity::class.java)
            startActivity(intent)
            finish()
        }
    }

    // Image picker result launcher
    private val pickImageResult =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                val bitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(it))
                imageView.setImageBitmap(bitmap)
                imageView.visibility = ImageView.VISIBLE

                // Perform inference after image is selected
                lifecycleScope.launch(Dispatchers.IO) {
                    val session = utils_class.createOrtSession(ortEnv)
                    withContext(Dispatchers.Main) {
                        val analyzer = ORTAnalyzer(session) { result ->
                            updateUI_upload(result)
                        }

                        analyzer.analyzeBitmap(bitmap)
                    }
                }
            }
        }

    private fun updateUI_upload(result: Result) {
        if (result.detectedScore.isEmpty())
            return
        // Display results
        textViewPredictedClass.text = labelData[result.detectedIndices[0]]
        textViewProbability.text = "Probability %.2f%%".format(result.detectedScore[0] * 100)
        textViewPredictedClass.visibility = TextView.VISIBLE
        textViewProbability.visibility = TextView.VISIBLE
    }

    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdown()
        ortEnv?.close()
    }
}
