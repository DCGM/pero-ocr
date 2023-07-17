#!/bin/sh

INPUT_DIR=
OUTPUT_DIR=
CONFIG=
EXAMPLE=

. ./.venv/bin/activate

print_help() {
    echo "Processing test"
    echo "Tests processing of images using PERO-OCR."
    echo "$ sh processing_test.sh -i in_dir -o out_dir -c config.ini -e example_out_dir"
    echo "Options:"
    echo "  -i|--input-dir      Input directory with tests."
    echo "  -o|--output-dir     Output directory for results."
    echo "  -c|--configuration  Configuration file for ocr."
    echo "  -e|--example        Example outputs for comparison."
    echo "  -h|--help           Shows this help message."
}

# parse args
while true; do
    case "$1" in
        --input-dir|-i )
            INPUT_DIR="$2"
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
        --help|-h )
            print_help
            exit 0
            ;;
        * )
            break
            ;;
    esac
done

# run parse_folder (in container) with correct params
# (include all inputs for given test)
# compare results with example results

# generate results
for f in $(ls "$INPUT_DIR"); do
    user_scripts/parse_folder.py \
        -c "$CONFIG" \
        --output-xml-path "$OUTPUT_DIR" \
        -i "$f"
done

# compare with example output
user_scripts/compare_page_xml_texts.py \
    --hyp "$EXAMPLE/$f" \
    --ref "$OUTPUT_DIR/$f" \
    --print-all
