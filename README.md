# Number recognizer
It is number recognizer on videos. You can configurate video preprocessing, vision window size and check results after recognizing on flow. This project in active working and I use it to get lab research results without having to look at the indicators every minute)

It uses easyOCR package to recognize. For details and installation instructions see: https://github.com/JaidedAI/EasyOCR. Image processing packadge is OpenCV.

# Installation
Please install it in this order, because openCV can be crashed.
For GPU:
 * Install CUDA for working on GPU: https://developer.nvidia.com/cuda-toolkit-archive
 * Install cuDNN SDK (I don't know it nessesary, but i install it for using GPU ): https://developer.nvidia.com/cudnn

 For CPU and GPU:
 * Install Tesseract OCR engine (necessary for easyOCR package) : https://tesseract-ocr.github.io/tessdoc/Installation.html
 After installation add TesseractOCR folder to PATH (in windows)
 * Install pythorch: https://pytorch.org/get-started/locally/
 * Install openCV for python: https://opencv.org/get-started/

    After installation openCV delete opencv-python-headless package to fix crashing with MethodNotImplemented error from cv2.selectROI function (it provides image cutting interactively), because opencv-python-headless not implement GUI functions and used for servers:

    ```pip uninstall opencv-python-headless```
 * Install easyOCR: https://github.com/JaidedAI/EasyOCR
 * Install python packadges for visualisation and animation:
    * matplotlib
    * PyQt5(for animation in separate window)
    * pympl (for animation in jupyter notebook)
    * tqdm (for animate recognize process)
Example requirements file in repository (for gpu and cpu)
# Features:
 * Process every video frame to increase recognizing quality interactively 
 * Choose framerate you need to recognize 
 * Cut several areas to recognize numbers
 * Configure custom pattern using regexp to check the correctness of recognizing and use slightly wrong results to combine them to get fully correct with verbose
 * Configure smart searching of image preprocessing configurations in case uncorrect recognizing


# Plans:
 * Separate checking, image process and recognizing modules
 * Add checking of correct image cutting
 * Add contrast and more options to image processor
 * Update image processor sweep to find best conditions faster
