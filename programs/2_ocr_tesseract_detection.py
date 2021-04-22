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
