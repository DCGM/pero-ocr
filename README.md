# pero-ocr
The package provides a full OCR pipeline including text paragraph detection,  text line detection, text transcription, and text refinement using a language model.
The package can be used as a command line application or as a python package which provides a document processing class and a class which represents document page content.


## Please cite
If you use pero-ocr, please cite:

* O Kodym, M Hradiš: Page Layout Analysis System for Unconstrained Historic Documents. ICDAR, 2021.
* M Kišš, K Beneš, M Hradiš: AT-ST: Self-Training Adaptation Strategy for OCR in Domains with Limited Transcriptions. ICDAR, 2021.
* J Kohút, M Hradiš: TS-Net: OCR Trained to Switch Between Text Transcription Styles. ICDAR, 2021.

## Running stuff
Scripts (as well as tests) assume that it is possible to import ``pero_ocr`` and its components.

For the current shell session, this can be achieved by setting ``PYTHONPATH`` up:
```
export PYTHONPATH=/path/to/the/repo:$PYTHONPATH
```

As a more permanent solution, a very simplistic `setup.py` is prepared:
```
python setup.py develop
```
Beware that the `setup.py` does not promise to bring all the required stuff, e.g. setting CUDA up is up to you.

Pero can be later removed from your Python distribution by running:
```
python setup.py develop --uninstall
```

## Available models
General layout analysis (printed and handwritten) with european printed OCR specialized to czech newspapers can be [downloaded here](https://nextcloud.fit.vutbr.cz/s/AasEy6KnXwkyXKi). The OCR engine is suitable for most european printed documents. It is specialized for low-quality czech newspapers digitized from microfilms, but it provides very good results for almast all types of printed documents in most languages. If you are interested in processing printed fraktur fonts, handwritten documents or medieval manuscripts, feel free to contact the authors. The newest OCR engines are available at [pero-ocr.fit.vutbr.cz](https://pero-ocr.fit.vutbr.cz). OCR engines are available also through API runing at [pero-ocr.fit.vutbr.cz/api](https://pero-ocr.fit.vutbr.cz/api), [github repository](https://github.com/DCGM/pero-ocr-api).

## Command line application
A command line application is ./user_scripts/parse_folder.py. It is able to process images in a directory using an OCR engine. It can render detected lines in an image and provide document content in Page XML and ALTO XML formats. Additionally, it is able to crop all text lines as rectangular regions of normalized size and save them into separate image files.

## Integration of the pero-ocr python module
This example shows how to directly use the OCR pipeline provided by pero-ocr package. This shows how to integrate pero-ocr into other applications. Class PageLayout represents content of a single document page and can be loaded from Page XMl and exported to Page XML and ALTO XML formats. The OCR pipeline is represented by the PageParser class.

```
import os
import configparser
import cv2
from pero_ocr.document_ocr.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser

# Read config file.
config_path = "./config_file.ini"
config = configparser.ConfigParser()
config.read(config_path)

# Init the OCR pipeline. 
# You have to specify config_path to be able to use relative paths
# inside the config file.
Page_parser = PageParser(config, 
    config_path=os.path.dirname(config_path))

# Read the document page image.
input_image_path = "page_image.jpg"
image = cv2.imread(input_image_path, 1)

# Init empty page content. 
# This object will be updated by the ocr pipeline. id can be any string and it is used to identify the page.
page_layout = PageLayout(id=input_image_path,
     page_size=(image.shape[0], image.shape[1]))

# Process the image by the OCR pipeline
page_layout = page_parser.process_page(input_image_path, page_layout)

page_layout.to_pagexml('output_page.xml') # Save results as Page XML.
page_layout.to_altoxml('output_ALTO.xml') # Save results as ALTO XML.

# Render detected text regions and text lines into the image and
# save it into a file.
page_layout.render_to_image(image) 
cv2.imwrite('page_image_render.jpg')

# Save each cropped text line in a separate .jpg file.
for region in page_layout.regions:
  for line in region.lines:
     cv2.imwrite(f'file_id-{line.id}.jpg', line.crop.astype(np.uint8))
```


## Contributing
Working changes are expected to happen on `develop` branch, so if you plan to contribute, you better check it out right during cloning:

```
git clone -b develop git@github.com:DCGM/pero-ocr.git pero-ocr
```

### Testing
Currently, only unittests are provided with the code. Some of the code. So simply run your preferred test runner, e.g.:
```
~/pero-ocr $ green
```

