package ai.onnxruntime.example.imageclassifier

import androidx.appcompat.app.AppCompatActivity
import android.content.Intent
import android.widget.Button
import android.os.Bundle

class FirstScreenActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.first_screen)

        val btnRealTime = findViewById<Button>(R.id.btnRealTimeInference)
        btnRealTime.setOnClickListener {
            // Move to MainActivity
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }

        val btnUpload = findViewById<Button>(R.id.btnUploadImage)
        btnUpload.setOnClickListener {
            val intent = Intent(this, ImageUploading::class.java)
            startActivity(intent)
        }
    }
}