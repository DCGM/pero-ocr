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

# generate results
if [ -z "$INPUT_XML_DIR" ]; then
    python3 user_scripts/parse_folder.py \
        -c "$CONFIG" \
        --output-xml-path "$OUTPUT_DIR" \
        -i "$INPUT_IMAGE_DIR"
else
    python3 user_scripts/parse_folder.py \
        -c "$CONFIG" \
        --output-xml-path "$OUTPUT_DIR" \
        -i "$INPUT_IMAGE_DIR" \
        -x "$INPUT_XML_DIR"
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
