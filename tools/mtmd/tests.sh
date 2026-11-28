#!/usr/bin/env bash

# make sure we are in the right directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

#export LLAMA_CACHE="$SCRIPT_DIR/tmp"

set -eux

mkdir -p $SCRIPT_DIR/output

PROJ_ROOT="$SCRIPT_DIR/../.."
cd $PROJ_ROOT

export MTMD_TEST_RESPONSE_MARKER="<MTMD_TEST_RESPONSE_MARKER>"

# Check if the first argument is "big", then run test with big models
# This is useful if we're running the script on a larger machine, so we can test the big models
RUN_BIG_TESTS=false
if [ "${1:-}" = "big" ]; then
    RUN_BIG_TESTS=true
    echo "Include BIG models..."
fi

RUN_HUGE_TESTS=false
if [ "${1:-}" = "huge" ]; then
    RUN_HUGE_TESTS=true
    RUN_BIG_TESTS=true
    echo "Include BIG and HUGE models..."
fi

USE_VIDEO=false
if [ "${1:-}" = "video" ]; then
    USE_VIDEO=true
    echo "Using video as input..."
    # behavior of USE_VIDEO:
    # do NOT check if the output contains "new york", only verify if the exit code is 0
    # when printing the result, print the OK/FAIL line then print the generated text
fi

# Check if the second argument is "flash", then enable flash attention
# This is useful to test if flash attention off works correctly
FLASH_ATTN="on"
if [ "${2:-}" = "flash_off" ] || [ "${1:-}" = "flash_off" ]; then
    FLASH_ATTN="off"
    echo "Flash attention disabled..."
fi

###############

arr_prefix=()
arr_hf=()
arr_extra_args=()
arr_file=()
# Sparse: only rows that declare an expectation are checked. See expect_encodes()/expect_log().
arr_encodes=()
arr_expect_log=()

# Expected number of image-encode calls for the row just added, one per slice plus one for the
# overview. Use it where the point of the row is the tile structure rather than the answer.
expect_encodes() {
    arr_encodes[$(( ${#arr_hf[@]} - 1 ))]=$1
}

# Fixed strings that must (expect_log) or must not (expect_no_log) appear in the row's output.
# The prompt-assembly lines they match are debug level, so a row using these has to pass -v.
# Stored newline separated, one leading + or - per pattern.
expect_log() {
    local i=$(( ${#arr_hf[@]} - 1 ))
    arr_expect_log[$i]="${arr_expect_log[$i]:-}+$1"$'\n'
}

expect_no_log() {
    local i=$(( ${#arr_hf[@]} - 1 ))
    arr_expect_log[$i]="${arr_expect_log[$i]:-}-$1"$'\n'
}

add_test_vision() {
    local hf=$1
    shift
    local extra_args=""
    if [ $# -gt 0 ]; then
        extra_args=$(printf " %q" "$@")
    fi
    if [ "$USE_VIDEO" = true ]; then
        arr_file+=("test-3.mp4")
    else
        arr_file+=("test-1.jpeg")
    fi
    arr_prefix+=("[vision]")
    arr_hf+=("$hf")
    arr_extra_args+=("$extra_args")
}

add_test_audio() {
    if [ "$USE_VIDEO" = true ]; then
        return 0
    fi
    local hf=$1
    shift
    local extra_args=""
    if [ $# -gt 0 ]; then
        extra_args=$(printf " %q" "$@")
    fi
    arr_prefix+=("[audio] ")
    arr_hf+=("$hf")
    arr_extra_args+=("$extra_args")
    arr_file+=("test-2.mp3")
}

add_test_vision "ggml-org/SmolVLM-500M-Instruct-GGUF:Q8_0"
add_test_vision "ggml-org/SmolVLM2-2.2B-Instruct-GGUF:Q4_K_M"
add_test_vision "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF:Q8_0"
# Passed the same image twice, because the `<image: N>` ordinal labels are only emitted when a
# prompt carries more than one image, and nothing else in this file exercises that path.
# test-1.jpeg is 640x488, which the base rule refines to 2048x2048, so each image is a 4x4 grid
# plus its overview: 2 x 17.
#
# The chunk total is the sequence assertion. Per image it is one text chunk for the ordinal and
# the overview delimiter, the overview image, then a delimiter text chunk and an image chunk per
# slice: 1 + 1 + 2*16 = 34, twice, plus the trailing text chunk after the last image. Drop the
# overview or a delimiter and the total moves, which the answer text never does.
add_test_vision "qvac/VisionPsy-Nano-460M-GGUFs:Q8_0" --image "$SCRIPT_DIR/test-1.jpeg" -v
expect_encodes 34
expect_log "add_text: <image: 0>"
expect_log "add_text: <image: 1>"
expect_log "adding overview image first"
expect_log "adding 16 slices (4 rows x 4 cols)"
expect_log "add_text: <row_1_col_1>"
expect_log "add_text: <row_4_col_4>"
expect_log "total = 69"
expect_no_log "adding overview image last"
# Flash is the same architecture with the no-upscale preprocessing rule; the published mmproj
# does not carry the key yet, so the flag is what selects it. Same image refines to 1024x512
# instead, a 2x1 grid plus its overview, which is the whole point of the variant.
# Chunk total 7: text, overview, delimiter, slice, delimiter, slice, text.
add_test_vision "qvac/VisionPsy-Nano-460M-Flash-GGUFs:Q8_0" --image-no-upscale on -v
expect_encodes 3
expect_log "adding overview image first"
expect_log "adding 2 slices (1 rows x 2 cols)"
expect_log "add_text: <row_1_col_1>"
expect_log "add_text: <row_1_col_2>"
expect_log "total = 7"
expect_no_log "adding overview image last"
# One image, so no ordinal label, and a 2x1 grid, so no third column and no second row.
expect_no_log "add_text: <image: 0>"
expect_no_log "add_text: <row_1_col_3>"
expect_no_log "add_text: <row_2_col_1>"
add_test_vision "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"
add_test_vision "THUDM/glm-edge-v-5b-gguf:Q4_K_M" -p "name of the newspaper?<__media__>"
add_test_vision "second-state/Llava-v1.5-7B-GGUF:Q2_K" --chat-template vicuna
add_test_vision "cjpais/llava-1.6-mistral-7b-gguf:Q3_K_M" --chat-template vicuna
add_test_vision "ibm-research/granite-vision-3.2-2b-GGUF:Q4_K_M"
add_test_vision "second-state/MiniCPM-Llama3-V-2_5-GGUF:Q2_K"  # model from openbmb is corrupted
add_test_vision "openbmb/MiniCPM-V-2_6-gguf:Q2_K"
add_test_vision "openbmb/MiniCPM-o-2_6-gguf:Q4_0"
add_test_vision "bartowski/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M"
add_test_vision "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M"
add_test_vision "ggml-org/InternVL2_5-1B-GGUF:Q8_0"
add_test_vision "ggml-org/InternVL3-1B-Instruct-GGUF:Q8_0"
add_test_vision "ggml-org/Qwen2.5-Omni-3B-GGUF:Q4_K_M"
add_test_vision "ggml-org/LFM2-VL-450M-GGUF:Q8_0"
add_test_vision "ggml-org/granite-docling-258M-GGUF:Q8_0"
add_test_vision "ggml-org/LightOnOCR-1B-1025-GGUF:Q8_0"
add_test_vision "ggml-org/DeepSeek-OCR-GGUF:Q8_0" -p "Free OCR." --chat-template deepseek-ocr
add_test_vision "ggml-org/dots.ocr-GGUF:Q8_0" -p "OCR"
add_test_vision "ggml-org/HunyuanOCR-GGUF:Q8_0" -p "OCR"
add_test_vision "ggml-org/gemma-4-E2B-it-GGUF:Q8_0" --jinja

add_test_audio  "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF:Q8_0"
add_test_audio  "ggml-org/Qwen2.5-Omni-3B-GGUF:Q4_K_M"
add_test_audio  "ggml-org/Voxtral-Mini-3B-2507-GGUF:Q4_K_M"
add_test_audio  "ggml-org/LFM2-Audio-1.5B-GGUF:Q8_0"
add_test_audio  "ggml-org/gemma-4-E2B-it-GGUF:Q8_0" --jinja
add_test_audio  "ggml-org/Qwen3-ASR-0.6B-GGUF:Q8_0"

# to test the big models, run: ./tests.sh big
if [ "$RUN_BIG_TESTS" = true ]; then
    add_test_vision "ggml-org/pixtral-12b-GGUF:Q4_K_M"
    add_test_vision "ggml-org/Mistral-Small-3.1-24B-Instruct-2503-GGUF" --chat-template mistral-v7
    add_test_vision "ggml-org/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M"
    add_test_vision "ggml-org/Qwen2-VL-7B-Instruct-GGUF:Q4_K_M"
    add_test_vision "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M"
    add_test_vision "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M"
    add_test_vision "ggml-org/Qwen3-VL-2B-Instruct-GGUF:Q8_0"
    add_test_vision "ggml-org/InternVL3-8B-Instruct-GGUF:Q4_K_M"
    add_test_vision "ggml-org/InternVL3-14B-Instruct-GGUF:Q4_K_M"
    add_test_vision "ggml-org/Qwen2.5-Omni-7B-GGUF:Q4_K_M"
    # add_test_vision "ggml-org/Qwen2.5-VL-32B-Instruct-GGUF:Q4_K_M" # does not work on my mac M3 Ultra
    # add_test_vision "ggml-org/Kimi-VL-A3B-Thinking-2506-GGUF:Q4_K_M" # not always working
    add_test_vision "ggml-org/GLM-4.6V-Flash-GGUF:Q4_K_M" -p "extract all texts from this image"

    add_test_audio  "ggml-org/ultravox-v0_5-llama-3_1-8b-GGUF:Q4_K_M"
    add_test_audio  "ggml-org/Qwen2.5-Omni-7B-GGUF:Q4_K_M"
fi

# to test the huge models, run: ./tests.sh huge
# this will run both the big and huge models
# huge models are > 32B parameters
if [ "$RUN_HUGE_TESTS" = true ]; then
    add_test_vision "ggml-org/Qwen2.5-VL-72B-Instruct-GGUF:Q4_K_M"
    add_test_vision "ggml-org/Llama-4-Scout-17B-16E-Instruct-GGUF:IQ1_S"
fi

# these models always give the wrong answer, not sure why
# add_test_vision "ggml-org/SmolVLM-Instruct-GGUF:Q4_K_M"
# add_test_vision "ggml-org/SmolVLM-256M-Instruct-GGUF:Q8_0"
# add_test_vision "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q8_0"

# this model has broken chat template, not usable
# add_test_vision "cmp-nct/Yi-VL-6B-GGUF:Q5_K"
# add_test_vision "guinmoon/MobileVLM-3B-GGUF:Q4_K_M" "deepseek"

###############

cmake --build build -j --target llama-mtmd-cli

arr_res=()

for i in "${!arr_hf[@]}"; do
    bin="llama-mtmd-cli"
    prefix="${arr_prefix[$i]}"
    hf="${arr_hf[$i]}"
    extra_args="${arr_extra_args[$i]}"
    inp_file="${arr_file[$i]}"

    echo "Running test with binary: $bin and HF model: $hf"
    echo ""
    echo ""

    cmd="$(printf %q "$PROJ_ROOT/build/bin/$bin") \
        -hf $(printf %q "$hf") \
        --image $(printf %q "$SCRIPT_DIR/$inp_file") \
        --temp 0 -n 128 \
        --flash-attn $(printf %q "$FLASH_ATTN") \
        ${extra_args}"

    # if extra_args does not contain -p, we add a default prompt
    if ! [[ "$extra_args" =~ "-p" ]]; then
        cmd+=" -p \"what is the publisher name of the newspaper?\""
    fi

    exit_code=0
    output=$(eval "$cmd" 2>&1 | tee /dev/tty) || exit_code=$?

    echo "$output" > $SCRIPT_DIR/output/$bin-$(echo "$hf" | tr '/' '-').log

    if [ "$USE_VIDEO" = true ]; then
        # for video, only check exit code; do not grep for "new york"
        if [ $exit_code -eq 0 ]; then
            result="$prefix \033[32mOK\033[0m:   $hf"
        else
            result="$prefix \033[31mFAIL\033[0m: $hf"
        fi
        # append generated text (after the response marker)
        generated_text=$(echo "$output" | sed "1,/${MTMD_TEST_RESPONSE_MARKER}/d" | tail -10)
        if [ -n "$generated_text" ]; then
            result+="\n$generated_text"
        fi
        echo -e "$result"
    else
        # either contains "new york" or both "men" and "walk"
        if echo "$output" | grep -iq "new york" \
                || (echo "$output" | grep -iq "men" && echo "$output" | grep -iq "walk")
        then
            result="$prefix \033[32mOK\033[0m:   $hf"
        else
            result="$prefix \033[31mFAIL\033[0m: $hf"
        fi
        # Where the tile structure is part of what the row tests, check it: the answer text
        # survives a wrong slice count, so a preprocessing regression is invisible here without
        # counting the encodes.
        want_encodes="${arr_encodes[$i]:-}"
        if [ -n "$want_encodes" ]; then
            got_encodes=$(echo "$output" | grep -c "encoding mtmd batch" || true)
            if [ "$got_encodes" != "$want_encodes" ]; then
                result="$prefix \033[31mFAIL\033[0m: $hf (encodes: got $got_encodes, want $want_encodes)"
            fi
        fi
        # Same idea for the prompt structure: the delimiters, the ordinal labels and the chunk
        # total say where every image went, and a wrong sequence still answers the question.
        patterns="${arr_expect_log[$i]:-}"
        if [ -n "$patterns" ]; then
            while IFS= read -r pat; do
                if [ -z "$pat" ]; then
                    continue
                fi
                want_present=true
                if [ "${pat:0:1}" = "-" ]; then
                    want_present=false
                fi
                pat="${pat:1}"
                found=true
                echo "$output" | grep -qF -- "$pat" || found=false
                if [ "$found" != "$want_present" ]; then
                    if [ "$want_present" = true ]; then
                        result="$prefix \033[31mFAIL\033[0m: $hf (log is missing '$pat')"
                    else
                        result="$prefix \033[31mFAIL\033[0m: $hf (log should not contain '$pat')"
                    fi
                fi
            done <<< "$patterns"
        fi
        echo -e "$result"
    fi
    arr_res+=("$result")

    echo ""
    echo ""
    echo ""
    echo "#################################################"
    echo "#################################################"
    echo ""
    echo ""
done

set +x

for i in "${!arr_res[@]}"; do
    echo -e "${arr_res[$i]}"
done
echo ""
echo "Output logs are saved in $SCRIPT_DIR/output"
