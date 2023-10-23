# Number recognizer
It is number recognizer on videos. You can configurate video preprocessing, vision window size and check results after recognizing on flow.

It uses easyOCR packadge to recongize. For details and installation instructions see: https://github.com/JaidedAI/EasyOCR. Image processing packadge is OpenCV.

# Installation
Please install it in this order, beacuse openCV can crash with MethodNotImplemented error from cv2.selectROI function (it can cut image interactively).

 * Install CUDA for working on GPU: https://developer.nvidia.com/cuda-toolkit-archive
 * Install cuDNN SDK (I don't know it nessesary, but i install it): https://developer.nvidia.com/cudnn
 * Install Tesseract OCR engine (need for easyOCR package) : https://tesseract-ocr.github.io/tessdoc/Installation.html
 After installation add TesseractOCR folder to PATH (in windows)
 * Install pythorch: https://pytorch.org/get-started/locally/
 * Install openCV for python: https://opencv.org/get-started/
 * Install easyOCR: https://github.com/JaidedAI/EasyOCR
 * Install python packadges for visualisation and animation: matplotlib,PyQt5(for animation in separate window),pympl (for animation in juoyter nootebook)
Example requirements file in repository (for gpu and cpu)