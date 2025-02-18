package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import androidx.camera.core.ImageAnalysis
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

import android.content.Context
import android.util.Log

class Utils(private val context: Context) {

    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }
    private val scope = CoroutineScope(Job() + Dispatchers.Main)

    private var ortEnv: OrtEnvironment? = null

    private suspend fun readModel(): ByteArray = withContext(Dispatchers.IO) {
        val modelID = R.raw.q_best_model_per_ch_d
        context.resources.openRawResource(modelID).readBytes()
    }

    suspend fun createOrtSession(ortEnv: OrtEnvironment?): OrtSession? =
        withContext(Dispatchers.Default) {
            if (ortEnv == null) {
                Log.e("DEBUG", "OrtEnv is NULL")
                return@withContext null
            }
            ortEnv?.createSession(readModel())
        }

    internal fun readLabels(): List<String> {
        return context.resources.openRawResource(R.raw.classes).bufferedReader().readLines()
    }

    internal fun setORTAnalyzer(
        img_an: ImageAnalysis?,
        ortEnv: OrtEnvironment?,
        updateUIFunction: (Result) -> Unit
    ) {
        scope.launch {
            img_an?.clearAnalyzer()
            img_an?.setAnalyzer(
                backgroundExecutor,
                ORTAnalyzer(createOrtSession(ortEnv), updateUIFunction)
            )
        }
    }
}