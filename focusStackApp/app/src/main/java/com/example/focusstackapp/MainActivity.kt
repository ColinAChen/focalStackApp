//https://stackoverflow.com/questions/27526941/class-referenced-in-the-manifest-was-not-found-in-the-project-or-the-libraries
package com.example.focusstackapp
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.media.Image
import android.net.Uri
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.Toast
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import com.example.focusstackapp.databinding.ActivityMainBinding
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
//https://stackoverflow.com/questions/44871481/how-to-access-values-from-strings-xml
//import com.example.cameraxapp.R
typealias LumaListener = (luma: Double) -> Unit

class MainActivity : AppCompatActivity() {
    private var imageCapture: ImageCapture? = null
    //private var image1Proxy:ImageProxy? = null
    //private var image2Proxy:ImageProxy? = null
    private var imagesToCapture = 2 // change this to take more images in the focal stack
    private var imCount:Int = 0 // keep track of what image we are taking
    private var imProxyList:MutableList<ImageProxy> = MutableList(imagesToCapture)
    private lateinit var outputDirectory: File
    private lateinit var cameraExecutor: ExecutorService
    //https://developer.android.com/topic/libraries/view-binding#kotlin
    private lateinit var binding: ActivityMainBinding
    private var firstDone = false
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //setContentView(R.layout.activity_main)
        binding = ActivityMainBinding.inflate(layoutInflater)
        // Request camera permissions
        val view = binding.root
        var camera:Camera? = null
        setContentView(view)
        if (allPermissionsGranted()) {
            camera = startCamera()
            Log.i("setup", "setup complete")
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // Set up the listener for take photo button
        // we'll set this up in startCamera so we still have access to the camera controls
        //camera_capture_button.setOnClickListener { takePhoto() }
        // maybe call the setup fucntion again to reistate tap to focus
        camera_capture_button.setOnClickListener { takePhotoStack(camera) }

        outputDirectory = getOutputDirectory()

        cameraExecutor = Executors.newSingleThreadExecutor()
    }



    private fun startCamera(): Camera? {
        var retCamera:Camera? = null
        // sets up the camera to be used in the preview view finder
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            // Set the preferred implementation mode before starting the preview

            // Preview

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewFinder.surfaceProvider)
                }

            //val previewView = PreviewView
            //preview.setSurfaceProvider(previewView.)
//            // Listen to tap events on the viewfinder and set them as focus regions
//
            imageCapture = ImageCapture.Builder()
                .build()

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                val camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture)
                Log.i("camera", ""+camera)
                retCamera = camera
                val cameraControl = camera.cameraControl
//                fun onTouch(x: Float, y: Float) {
//                    // Create a factory for creating a MeteringPoint
//                    val factory = viewFinder.meteringPointFactory
//
//                    // Convert UI coordinates into camera sensor coordinates
//                    val point = factory.createPoint(x, y)
//
//                    // Prepare focus action to be triggered
//                    val action = FocusMeteringAction.Builder(point).build()
//
//                    // Execute focus action
//                    cameraControl.startFocusAndMetering(action)
//                }
                // set tap to focus
                // will be used for selecting the foreground manually
                viewFinder.setOnTouchListener(View.OnTouchListener setOnTouchListener@{ view: View, motionEvent: MotionEvent ->
                    when (motionEvent.action) {
                        MotionEvent.ACTION_DOWN -> return@setOnTouchListener true
                        MotionEvent.ACTION_UP -> {
                            // Get the MeteringPointFactory from PreviewView
                            val factory = viewFinder.meteringPointFactory

                            // Create a MeteringPoint from the tap coordinates
                            val point = factory.createPoint(motionEvent.x, motionEvent.y)

                            // Create a MeteringAction from the MeteringPoint, you can configure it to specify the metering mode
                            val action = FocusMeteringAction.Builder(point).build()

                            // Trigger the focus and metering. The method returns a ListenableFuture since the operation
                            // is asynchronous. You can use it get notified when the focus is successful or if it fails.

                            camera.cameraControl.startFocusAndMetering(action)



                            return@setOnTouchListener true
                        }
                        else -> return@setOnTouchListener false
                    }
                })
                //@androidx.camera.camera2.interop.ExperimentalCamera2Interop

//
            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
        Log.i("camera", ""+retCamera)
        return retCamera
    }

    private fun takePhoto(){
        // want to take one picture at the autofocus location and one at infinity
        // worse case can sample the photo at periodic x,y since we have no idea of the distances
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return
        //Log.i("takePhoto", "" + imageCapture)

        // Create time-stamped output file to hold the image
        val photoFile = File(
            outputDirectory,
            SimpleDateFormat(FILENAME_FORMAT, Locale.US
            ).format(System.currentTimeMillis()) + ".jpg")

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        // Set up image capture listener, which is triggered after photo has
        // been taken
//        imageCapture.takePicture(
//            outputOptions, ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageSavedCallback {
//                override fun onError(exc: ImageCaptureException) {
//                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
//                }
//
//                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
//                    val savedUri = Uri.fromFile(photoFile)
//                    val msg = "Photo capture succeeded: $savedUri"
//                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
//                    Log.d(TAG, msg)
//                    // https://stackoverflow.com/questions/57786636/cant-take-multiple-images-using-camerax
//                    image.close()
//                }
//                override fun onCaptureSuccess(imageProxy: ImageProxy){
//
//                }
//            })
        // in memory acccess
        var foregroundImage: Image? = null
        var waitForComplete = false
        var retProxy: ImageProxy? = null
        //Log.i("takePhoto", "waitForCompelte " + waitForComplete)
        //https://www.raywenderlich.com/6748203-camerax-getting-started
        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this), object: ImageCapture.OnImageCapturedCallback() {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                    Log.i("takePhoto", "image capture fail")
                }

                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    Log.i("takePhoto", "success")
                    val savedUri = Uri.fromFile(photoFile)
                    val msg = "Photo capture succeeded: $savedUri"
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
//                    val checkNull = imageProxy.image
                    retProxy = imageProxy
//                    if (checkNull != null){
//                        foregroundImage = checkNull
//                    }
                    //firstDone = true
                    Log.d(TAG, msg)
                    waitForComplete=true
                    Log.i("takePhoto", "retProxy" + retProxy)
//                    if (!firstDone){
//                        image1Proxy = retProxy
//                        firstDone = true
//                    }
//                    else{
//                        image2Proxy = retProxy
//                    }
                    imProxyList.add(imCount, imageProxy)
                    imCount++
                    //Log.i("takePhoto", "capture success" + waitForComplete)

                    // https://stackoverflow.com/questions/57786636/cant-take-multiple-images-using-camerax
                    //foregroundImage?.close()
                }

            })
        // wait for image capture to be completed
        var logOnce = false
//        while(!waitForComplete){
//            var one = 1 // no op?
//            if (!logOnce){
//                Log.i("takePhoto", "wait for complete loop")
//                logOnce = true
//            }
//        }

        //return foregroundImage
//        while (!waitForComplete){
//            Thread.sleep(100)
//        }
        //Log.i("takePhoto", "retProxy" + retProxy)
        //return retProxy


    }
    private fun savePhoto(){
        val imageCapture = imageCapture ?: return

        // Create time-stamped output file to hold the image
        val photoFile = File(
            outputDirectory,
            SimpleDateFormat(FILENAME_FORMAT, Locale.US
            ).format(System.currentTimeMillis()) + ".jpg")

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        // Set up image capture listener, which is triggered after photo has
        // been taken
        imageCapture.takePicture(
            outputOptions, ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val savedUri = Uri.fromFile(photoFile)
                    val msg = "Photo capture succeeded: $savedUri"
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                    Log.d(TAG, msg)
                    // https://stackoverflow.com/questions/57786636/cant-take-multiple-images-using-camerax

                }

            })
    }
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun getOutputDirectory(): File {
        val mediaDir = externalMediaDirs.firstOrNull()?.let {
            File(it, resources.getString(R.string.app_name)).apply { mkdirs() } }
        return if (mediaDir != null && mediaDir.exists())
            mediaDir else filesDir
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraXBasic"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }
    /**
     *  convert image proxy to bitmap
     *  @param image
     *  https://stackoverflow.com/questions/61876286/camerax-capturing-photo-as-bitmap
     */
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val planeProxy = image.planes[0]
        val buffer: ByteBuffer = planeProxy.buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
    private fun saveImage(bitmap: Bitmap, fileName:File){
        //https://stackoverflow.com/questions/649154/save-bitmap-to-location
        try {
            //val filename = null
            FileOutputStream(fileName).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out) // bmp is your Bitmap instance
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }
    private fun takePhotoStack(){
        /*
        Use imagesToCapture to determine the focal positions
        for focal position:
            Move lens to position
            take photo
         */
    }
    private fun takePhotoStack(camera:Camera?){
        /*
        take photo
        move lens
        take photo
        is it faster to save to storage then retreive from storage or read directly into memory?

        determine the number of lens positions
        maybe do this without tap to focus or think about this use case later
         */
        Log.i("camera", ""+camera)
        // take the first photo
        var photoFile = File(
            outputDirectory,
            SimpleDateFormat(FILENAME_FORMAT, Locale.US
            ).format(System.currentTimeMillis()) + ".jpg")
        Log.i("takePhoto", ""+photoFile)
        takePhoto() // submit the photo request
        //https://mohak1712.medium.com/kotlin-coroutines-thread-sleep-vs-delay-63171fe8a24
        // do we need to wait before submitting camera requests?
        GlobalScope.launch(context=Dispatchers.Main) {
            while(image1Proxy == null) {
                // wait until the first image has been captured
                delay(10)
            }
            // move the lens and take the second image
            if (camera != null){
                //set focus to infinity, probably needs to be done in take picture somewhere
                //https://groups.google.com/a/android.com/g/camerax-developers/c/VsyQnS1uO_U/m/XEd3PnlIBgAJ
                val minFDistance = Camera2CameraInfo.from(camera.cameraInfo).getCameraCharacteristic(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)
                if (minFDistance != null && minFDistance.compareTo(0) > 0){
                    // check if the device supports focusing
                    var options = CaptureRequestOptions.Builder().setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF).setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE,0F).build() // turn off autofocus.change focus to 1/m = 1/0 = inifinity
                    Camera2CameraControl.from(camera.cameraControl).setCaptureRequestOptions(options)

                }
            }
            Log.i("takePhoto", "lens movement done")
            takePhoto()
        }
        GlobalScope.launch(context=Dispatchers.Main) {
            // use this coroutine to save images
            // ideally this will reduce the time between taking images
            while(image1Proxy == null) {
                // wait until the first image has been captured
                delay(10)
            }

            var photoFile = File(
                outputDirectory,
                SimpleDateFormat(FILENAME_FORMAT, Locale.US
                ).format(System.currentTimeMillis()) + ".jpg")
            Log.i("takePhoto", ""+photoFile)
            if (image1Proxy != null){
                Log.i("takePhoto", "retreive image proxy")
                var checkImage = image1Proxy!!.image
                if (checkImage != null){
                    Log.i("takePhoto", "" + checkImage.height)
                }
                else{
                    Log.i("takePhoto", "failed to retrieve image")
                }
                var firstBitmap :Bitmap?= null
                firstBitmap = imageProxyToBitmap(image1Proxy!!)
                saveImage(firstBitmap, photoFile)
                image1Proxy!!.close()

            }
            else{
                Log.i("takePhoto", "failed to retreive image proxy")
            }
            while(image2Proxy == null) {
                // wait until the second image has been captured
                delay(10)
            }
            photoFile = File(
                outputDirectory,
                SimpleDateFormat(FILENAME_FORMAT, Locale.US
                ).format(System.currentTimeMillis()) + ".jpg")
            if (image2Proxy != null){
                Log.i("takePhoto", "retreive image proxy")
                var checkImage = image2Proxy!!.image
                if (checkImage != null){
                    Log.i("takePhoto", "" + checkImage.height)
                }
                else{
                    Log.i("takePhoto", "failed to retrieve image")
                }
                var secondBitmap :Bitmap?= null
                secondBitmap = imageProxyToBitmap(image2Proxy!!)
                saveImage(secondBitmap, photoFile)
                image2Proxy!!.close()

            }
            else{
                Log.i("takePhoto", "failed to retreive image proxy")
            }

        }





//        var out = false
//        GlobalScope.launch(context=Dispatchers.Main){
//
//            while (!firstDone){
//                delay(100)
//            }
//            Log.i("loop", "in coroutine, out of loop")
//            if (camera != null){
//                //set focus to infinity, probably needs to be done in take picture somewhere
//                //https://groups.google.com/a/android.com/g/camerax-developers/c/VsyQnS1uO_U/m/XEd3PnlIBgAJ
//                val minFDistance = Camera2CameraInfo.from(camera.cameraInfo).getCameraCharacteristic(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)
//                if (minFDistance != null && minFDistance.compareTo(0) > 0){
//                    // check if the device supports focusing
//                    var options = CaptureRequestOptions.Builder().setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF).setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE,0F).build() // turn off autofocus.change focus to 1/m = 1/0 = inifinity
//                    Camera2CameraControl.from(camera.cameraControl).setCaptureRequestOptions(options)
//
//                }
//                photoFile = File(
//                    outputDirectory,
//                    SimpleDateFormat(FILENAME_FORMAT, Locale.US
//                    ).format(System.currentTimeMillis()) + ".jpg")
//                var secondImage = takePhoto()
//                if (secondImage!=null){
//                    var secondBitmap = imageProxyToBitmap(secondImage)
//                    saveImage(secondBitmap, photoFile)
//                }
//                else{
//                    Log.e("save", "image2 save failed")
//                }
//            }
//            else{
//                Log.i("camera", "receive null camera")
//            }
//        }
//        while(!firstDone){
////            if (!out){
////                Log.i("loop","first")
////                out = true
////            }
//            delay()
//
//        }
        //Log.i("loop", "out of loop")
        //takePhotoToFile()
//


        // maybe need to put some delay here? not sure if the code will proceed before lens is in position


        //var secondImage = takePhoto()
        // take the second picture after moving the lens
//        photoFile = File(
//            outputDirectory,
//            SimpleDateFormat(FILENAME_FORMAT, Locale.US
//            ).format(System.currentTimeMillis()) + ".jpg")
//        var secondImage = takePhoto()
//        if (secondImage!=null){
//            var secondBitmap = imageProxyToBitmap(secondImage)
//            saveImage(secondBitmap, photoFile)
//        }
//        else{
//            Log.e("save", "image2 save failed")
//        }



                    // do we need to restore the tap to focus?
//                    viewFinder.setOnTouchListener(View.OnTouchListener setOnTouchListener@{ view: View, motionEvent: MotionEvent ->
//                        when (motionEvent.action) {
//                            MotionEvent.ACTION_DOWN -> return@setOnTouchListener true
//                            MotionEvent.ACTION_UP -> {
//                                // Get the MeteringPointFactory from PreviewView
//                                val factory = viewFinder.meteringPointFactory
//
//                                // Create a MeteringPoint from the tap coordinates
//                                val point = factory.createPoint(motionEvent.x, motionEvent.y)
//
//                                // Create a MeteringAction from the MeteringPoint, you can configure it to specify the metering mode
//                                val action = FocusMeteringAction.Builder(point).build()
//
//                                // Trigger the focus and metering. The method returns a ListenableFuture since the operation
//                                // is asynchronous. You can use it get notified when the focus is successful or if it fails.
//                                cameraControl.startFocusAndMetering(action)
//
//                                return@setOnTouchListener true
//                            }
//                            else -> return@setOnTouchListener false
//                        }
//                    })
    }
//    private fun segmentImage(){
//        // perform image segmentation, do this as soon as an i
//    }
    private fun processFocusStack(){
        
    }
}