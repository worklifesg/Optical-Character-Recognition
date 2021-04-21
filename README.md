## Optical-Character-Recognition
Concepts and tools to develop functional OCR based programs that will be used in other projects involving text, signs, digitalized information from images and videos.

**Credits: Adrian Rosebrock, OCR with OpenCV, Tesseract, and Python, PyImageSearch**

### Table of contents
=================

<!--ts-->
   * [First Program](#first-program)
   * [Detecting and OCR Digits with Tesseract](#detect-and-ocr-digits-with-tesseract)
<!--te-->

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

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Detecting and OCR Text with Tesseract

In previous program, we saw that Tesseract has certain limitations just like any other software tool where it can't recognize text and needs to be updated for different input images. So, detection and applying OCR can be done **using some image processing techniques** as well as **modifying certain Tesseract options**

So in the program, we will **identify and OCR text and digits using Tesseract**. This program will allow us to have more control of what we need to detect. **Also we will NOT USE image processing techniques** for now to see its importance.


```python
'''
Need to install Tesseract-OCR and pytesseract as they are not pre-installed
on Colab.
For running on system Tesseract-OCR can be installed from:
https://github.com/UB-Mannheim/tesseract/wiki
'''

import pytesseract
from pytesseract import Output
import argparse
import cv2
from google.colab.patches import cv2_imshow

# Argument construct and parsing argument
arg = argparse.ArgumentParser()
arg.add_argument("-i", "--image", required=True, help="path to input image to be OCR'd")
arg.add_argument("-c", "--min-conf", type=int, default=0, help="mininum confidence value to filter weak text detection")
args = vars(arg.parse_args())

#image read, convert to gray scale, applying tesseract

image = cv2.imread(args['image'])
image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

text_result = pytesseract.image_to_data(image_rgb,output_type=Output.DICT)

# Constructing bounding box and their OCR'd 
for i in range(0,len(text_result["text"])):
  #coordinates of the box
  x = text_result['left'][i]
  y = text_result['top'][i]
  w = text_result['width'][i]
  h = text_result['height'][i]
  #OCRd text
  text = text_result['text'][i]
  confidence = int(text_result['conf'][i])

  # we need to filter out weak confidence text as some logos can be detected as text
  if confidence > args['min_conf']:
    print("Confidence: {}".format(confidence))
    print("Text: {}".format(text))
    print("")
  
  # printing needed text and digits
    text = ''.join([c if ord(c) < 128 else "" for c in text]).strip()
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(image,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 3)

cv2_imshow(image)
cv2.imwrite('result.jpg',image)
cv2.waitKey(0)
```

Input Image | Output w/o confidence | Output with confidence | Image Processing
| ------------- | ------------- | ------------- | ------------- |
| ![](https://github.com/worklifesg/Optical-Character-Recognition/blob/main/images/apple_support.png) | ![](https://github.com/worklifesg/Optical-Character-Recognition/blob/main/images/apple_support_detect.jpg)  |![](https://github.com/worklifesg/Optical-Character-Recognition/blob/main/images/apple_support_detect_conf.jpg) | Doesn't need **Image Prcessing** |
| ![](https://github.com/worklifesg/Optical-Character-Recognition/blob/main/images/card.jpg) | ![](https://github.com/worklifesg/Optical-Character-Recognition/blob/main/images/result_card_detect.jpg)  |![](https://github.com/worklifesg/Optical-Character-Recognition/blob/main/images/result_card_detect_conf.jpg) | Need **Image Prcessing** as can't detect digits easily |
| ![](https://github.com/worklifesg/Optical-Character-Recognition/blob/main/images/image_floor1.jpg) | ![](https://github.com/worklifesg/Optical-Character-Recognition/blob/main/images/floor_map_detect.jpg)  |![](https://github.com/worklifesg/Optical-Character-Recognition/blob/main/images/floor_map_detect_conf.jpg) | Need **Image Prcessing** as certain text and digits are not detected|

**Analysis:**
* Image processing is needed to make images more understable to the software tool
* Confidence helps in eliminating useless and wrong texts

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

