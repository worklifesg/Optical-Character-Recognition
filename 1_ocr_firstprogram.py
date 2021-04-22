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
