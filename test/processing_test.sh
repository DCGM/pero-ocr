#!/bin/sh

INPUT_IMAGE_DIR=
INPUT_XML_DIR=
OUTPUT_DIR=
CONFIG=
EXAMPLE=
TEST_UTIL=
TEST_OUTPUT_DIR=
TEST_SET=

print_help() {
    echo "Processing test"
    echo "Tests processing of images using PERO-OCR."
    echo "$ sh processing_test.sh -i in_dir -o out_dir -c config.ini -e example_out_dir"
    echo "Options:"
    echo "  -i|--input-images   Input directory with test images."
    echo "  -x|--input-xmls     Input directory with xml files."
    echo "  -o|--output-dir     Output directory for results."
    echo "  -c|--configuration  Configuration file for ocr."
    echo "  -e|--example        Example outputs for comparison."
    echo "  -u|--test-utility   Path to test utility."
    echo "  -t|--test-output    Test utility output folder."
    echo "  -h|--help           Shows this help message."
}

# parse args
while true; do
    case "$1" in
        --input-images|-i )
            INPUT_IMAGE_DIR="$2"
            shift 2
            ;;
        --input-xmls|-x )
            INPUT_XML_DIR="$2"
            shift 2
            ;;
        --output-dir|-o )
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --configuration|-c )
            CONFIG="$2"
            shift 2
            ;;
        --example|-e )
            EXAMPLE="$2"
            shift 2
            ;;
        --test-utility|-u )
            TEST_UTIL="$2"
            shift 2
            ;;
        --test-output|-t )
            TEST_OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h )
            print_help
            exit 0
            ;;
        * )
            break
            ;;
    esac
done

if [ -z "$INPUT_IMAGE_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$CONFIG" ]; then
    echo 'Input image directory, output directory and configuration path must be set!' >&2
    exit 1
fi

config_name="$(basename "$CONFIG")"
config_path="$(dirname "$CONFIG")"

# generate results
if [ -z "$INPUT_XML_DIR" ]; then
    docker run --rm --tty --interactive \
    --volume "$INPUT_IMAGE_DIR":/input \
    --volume "$OUTPUT_DIR":/output \
    --volume "$dirname":/engine pero-ocr \
    /usr/bin/python3 user_scripts/parse_folder.py \
        --config /engine/"$config_name" \
        --input-image-path /input \
        --output-xml-path /output \
        --device cpu
else
    docker run --rm --tty --interactive \
    --volume "$INPUT_IMAGE_DIR":/input \
    --volume "$INPUT_XML_DIR":/input_xml \
    --volume "$OUTPUT_DIR":/output \
    --volume "$dirname":/engine pero-ocr \
    /usr/bin/python3 user_scripts/parse_folder.py \
        --config /engine/"$config_name" \
        --input-image-path /input \
        --input-xml-path /input_xml \
        --output-xml-path /output \
        --device cpu
fi

# test if all options for tests are set
[ -n "$EXAMPLE" ] && TEST_SET=1
[ -n "$TEST_UTIL" ] && TEST_SET=1
[ -n "$TEST_OUTPUT_DIR" ] && TEST_SET=1

# compare with example output
if [ -n "$EXAMPLE" ] && [ -n "$TEST_OUTPUT_DIR" ] && [ -n "$TEST_UTIL" ]; then
    python3 "$TEST_UTIL" \
        --input-path "$OUTPUT_DIR" \
        --gt-path "$EXAMPLE" \
        --image-path "$INPUT_IMAGE_DIR" \
        --out-image-path "$TEST_OUTPUT_DIR/output-images" \
        --log-path "$TEST_OUTPUT_DIR/log.json"
else
    if [ -n "$TEST_SET" ]; then
        echo "For running test, example output directory, test utility path"
        echo "and test output dir have to be set!"
    fi
fi
