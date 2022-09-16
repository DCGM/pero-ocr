# pero-ocr
The package provides a full OCR pipeline including text paragraph detection, text line detection, text transcription, and text refinement using a language model.
The package can be used as a command line application or as a python package which provides a document processing class and a class which represents document page content.

The layout analysis uses segmentation-style fully convolutional networks with hand-coded post-processing as described in 

> Kodym, O. and Hradiš, M. TG2: text-guided transformer GAN for restoring document readability and perceived quality. IJDAR (2021). [arxiv](https://arxiv.org/abs/2102.11838v1)


Visual OCR models are trained using CTC and they combine convolutional networks with recurrent layers and special Transcription Style Blocks with allow the network to learn different transcription styles and switch between them during inference.

> Kohút, J. and Hradiš, M. TS-Net: OCR Trained to Switch Between Text Transcription Styles. ICDAR 2021.  [arxiv](https://arxiv.org/abs/2103.05489)


Language models are fairly standard charaacter-level autoregressive recurrent networks. The code provides an efficient implementation of beam serch on the probabilities from an OCR model and the probabilities from a language model.

## Please cite
If you use pero-ocr, please cite:

> Kodym, O. and Hradiš, M. TG2: text-guided transformer GAN for restoring document readability and perceived quality. IJDAR (2021).

> Kodym, O. and Hradiš, M. Page Layout Analysis System for Unconstrained Historic Documents. ICDAR, 2021.

> Kišš, M., Beneš, K., and Hradiš, M. AT-ST: Self-Training Adaptation Strategy for OCR in Domains with Limited Transcriptions. ICDAR, 2021.

> Kohút, J. and Hradiš, M. TS-Net: OCR Trained to Switch Between Text Transcription Styles. ICDAR, 2021.


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
Beware that the `setup.py` does NOT check for dependencies in the current version.

Pero can be later removed from your Python distribution by running:
```
python setup.py develop --uninstall
```

## Processing documents from command line
The package provides command line tool (user_scripts/parse_folder.py.) which can be used to process images. Simple way how to process directory with images is:
```
python user_scripts/parse_folder.py -c PATH_TO_config_file_for_OCR_ENGINE.ini -i path_to_image_directory --output-xml-path PATH_TO_OUTPUT_DIRECTORY
```

## Available models
General layout analysis (printed and handwritten) with european printed OCR specialized to czech newspapers can be [downloaded here](https://www.fit.vut.cz/~ihradis/pero/pero_eu_cz_print_newspapers_2020-10-09.tar.gz). The OCR engine is suitable for most european printed documents. It is specialized for low-quality czech newspapers digitized from microfilms, but it provides very good results for almast all types of printed documents in most languages. If you are interested in processing printed fraktur fonts, handwritten documents or medieval manuscripts, feel free to contact the authors. The newest OCR engines are available at [pero-ocr.fit.vutbr.cz](https://pero-ocr.fit.vutbr.cz). OCR engines are available also through API runing at [pero-ocr.fit.vutbr.cz/api](https://pero-ocr.fit.vutbr.cz/api), [github repository](https://github.com/DCGM/pero-ocr-api).

## Using the python package
The package provides two main classes: 

1. PageLayout which represents page content, can be exported to PAGE XML, ALTO XML, text and rendered (and also loded from PAGE XML), 
2. PageParser which can load OCR engine from a configuration file and process images. 

A basic example how to load an OCR engine, process an image, export results and render the layou follows:
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


## Developing & Contributing
Working changes are expected to happen on `develop` branch, so if you plan to contribute, you better check it out right during cloning:

```
git clone -b develop git@github.com:DCGM/pero-ocr.git pero-ocr
```

### Testing
Currently, only unittests are provided with the code. Some of the code. So simply run your preferred test runner, e.g.:
```
~/pero-ocr $ green
```
