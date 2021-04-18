## Optical-Character-Recognition
Concepts and tools to develop functional OCR based programs that will be used in other projects involving text, signs, digitalized information from images and videos.

**Credits: Adrian Rosebrock, OCR with OpenCV, Tesseract, and Python, PyImageSearch**

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

### First Program

We will use Tesseract-OCR and ```pytesseract``` to OCR'd the images and will see how tricky and challenging an OCR process might be.

```python
'''
Need to install Tesseract-OCR and pytesseract as they are not pre-installed
on Colab.
For running on system Tesseract-OCR can be installed from:
https://github.com/UB-Mannheim/tesseract/wiki
'''

import pytesseract #Python binding that ties in directly with the Tesseract OCR application running on your system.
import argparse
import cv2

'''
Constructing argument parser and pasring the arguments
Argument parser is helpful to execute program for any image in one line rather than importing all images in our program
'''

arg = argparse.ArgumentParser()
arg.add_argument("-i","--image",required=True,help='path to the image')
args = vars(arg.parse_args())

'''
Read image and apply pytesseract to OCR the image
'''

image = cv2.imread(args['image'])
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
text = pytesseract.image_to_string(image)
print(text)
```

Image | Output  |
| ------------- | ------------- |
| ![](https://github.com/worklifesg/Optical-Character-Recognition/blob/main/images/text.PNG) | In this chapter, you created your very first OCR project using the Tesseract OCR engine, the pytesseract package (used to interact with the Tesseract OCR engine), and the OpenCV library (used to load an input image from disk). |
| ![](https://github.com/worklifesg/Optical-Character-Recognition/blob/main/images/business2.jpg) | ![](https://github.com/worklifesg/Optical-Character-Recognition/blob/main/images/business2_ocrd.PNG) |
| ![](https://github.com/worklifesg/Optical-Character-Recognition/blob/main/images/whole_foods1.png) | ![](https://github.com/worklifesg/Optical-Character-Recognition/blob/main/images/whole_foods1_ocrd.PNG) |

**Analysis:**
* Simple texts in an image (image 1) are predicted 100% but in most of the real world images, it is not like this
* The business card is also predicted well as it was clear and not titled or rotated.
* For mutlicolum or low contrast images like a receipt, the text is not predicted well and this is where OCR is challenging where we need to apply certain image pre-processing techniques as well as deep learning techniques in certain cases.
