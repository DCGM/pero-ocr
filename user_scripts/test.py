import os
import configparser
import cv2
import numpy as np
from pero_ocr.core.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser

# Read config file.
config_path = "/mnt/matylda1/ikohut/data/pero-worker_data/layout.2022-04-13.czech_printed.2022-01-13/config_pero-ocr.fit.vutbr.cz.layout.ini"
config = configparser.ConfigParser()
config.read(config_path)

# Init the OCR pipeline.
# You have to specify config_path to be able to use relative paths
# inside the config file.
page_parser = PageParser(config, config_path=os.path.dirname(config_path))

# Read the document page image.
input_image_path = "page_image.jpg"
image = cv2.imread(input_image_path, 1)
print(image.shape)

# Init empty page content.
# This object will be updated by the ocr pipeline. id can be any string and it is used to identify the page.
page_layout = PageLayout(id=input_image_path, page_size=(int(image.shape[0]), int(image.shape[1])))

# Process the image by the OCR pipeline
page_layout = page_parser.process_page(image, page_layout)

page_layout.to_pagexml('output_page.xml') # Save results as Page XML.
page_layout.to_altoxml('output_ALTO.xml') # Save results as ALTO XML.

# Render detected text regions and text lines into the image and
# save it into a file.
rendered_image = page_layout.render_to_image(image)
cv2.imwrite('page_image_render.jpg', rendered_image)

# Save each cropped text line in a separate .jpg file.
for region in page_layout.regions:
  for line in region.lines:
     cv2.imwrite(f'file_id-{line.id}.jpg', line.crop.astype(np.uint8))