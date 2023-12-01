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

## Running command line application in container
A docker container can be built from the sourcecode to run scripts and programs based on the pero-ocr. Example of running the `parse_folder.py` script to generate page-xml files for images in input directory:
```shell
docker run --rm --tty --interactive \
     --volume path/to/input/dir:/input \
     --volume path/to/output/dir:/output \
     --volume path/to/ocr/engine:/engine \
     --gpus all \
     pero-ocr /usr/bin/python3 user_scripts/parse_folder.py \
          --config /engine/config.ini \
          --input-image-path /input \
          --output-xml-path /output
```
Be sure to use container internal paths for passed in data in the command. All input and output data locations have to be passed to container via `--volume` argument due to container isolation. See [docker run command reference](https://docs.docker.com/engine/reference/run/) for more information.

Container can be built like this:
```shell
docker build -f Dockerfile -t pero-ocr .
```

## Integration of the pero-ocr python module
This example shows how to directly use the OCR pipeline provided by pero-ocr package. This shows how to integrate pero-ocr into other applications. Class PageLayout represents content of a single document page and can be loaded from Page XMl and exported to Page XML and ALTO XML formats. The OCR pipeline is represented by the PageParser class.

```python
import os
import configparser
import cv2
import numpy as np
from pero_ocr.document_ocr.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser

# Read config file.
config_path = "./config_file.ini"
config = configparser.ConfigParser()
config.read(config_path)

# Init the OCR pipeline. 
# You have to specify config_path to be able to use relative paths
# inside the config file.
page_parser = PageParser(config, config_path=os.path.dirname(config_path))

# Read the document page image.
input_image_path = "page_image.jpg"
image = cv2.imread(input_image_path, 1)

# Init empty page content. 
# This object will be updated by the ocr pipeline. id can be any string and it is used to identify the page.
page_layout = PageLayout(id=input_image_path,
     page_size=(image.shape[0], image.shape[1]))

# Process the image by the OCR pipeline
page_layout = page_parser.process_page(image, page_layout)

page_layout.to_page_xml('output_page.xml') # Save results as Page XML.
page_layout.to_alto_xml('output_ALTO.xml') # Save results as ALTO XML.

# Render detected text regions and text lines into the image and
# save it into a file.
rendered_image = page_layout.render_to_image(image) 
cv2.imwrite('page_image_render.jpg', rendered_image)

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

#### Simple regression testing
Regression testing can be done by `test/processing_test.sh`. Script calls containerized `parser_folder.py` to process input images and page-xml files and calls user suplied comparison script to compare outputs to example outputs suplied by user. `PERO-OCR` container have to be built in advance to run the test, see 'Running command line application in container' chapter. Script can be called like this:
```shell
sh test/processing_test.sh \
     --input-images path/to/input/image/directory \
     --input-xmls path/to/input/page-xml/directory \
     --output-dir path/to/output/dir \
     --configuration path/to/ocr/engine/config.ini \
     --example path/to/example/output/data \
     --test-utility path/to/test/script \
     --test-output path/to/testscript/output/dir \
     --gpu-ids gpu ids for docker container
```

First 4 arguments are manadatory, `--gpu-ids` is preset by value 'all' which passes all gpus to the container. Test utility, example outputs and test output folder have to be set only if comparison of results should be performed. Test utility is expected to be path to `eval_ocr_pipeline_xml.py` script from `pero` repository. Be sure to correctly set PYTHONPATH and install dependencies for `pero` repository for the utility to work. Other script can be used if takes the same arguments. In other cases output data can be of course compared manually after processing.
