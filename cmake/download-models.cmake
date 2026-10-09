get_filename_component(DEST_DIR "${DEST}" DIRECTORY)
file(MAKE_DIRECTORY "${DEST_DIR}")

# Split the "ALGO=value" hash spec so we can verify the download ourselves.
string(REPLACE "=" ";" hash_parts "${HASH}")
list(GET hash_parts 0 hash_algo)
list(GET hash_parts 1 hash_value)

# Reuse an already-downloaded, valid copy.
if(EXISTS "${DEST}")
    file(${hash_algo} "${DEST}" have_hash)
    if(have_hash STREQUAL hash_value)
        return()
    endif()
endif()

# Source URL defaults to the ggml-org HF repo but can be overridden, e.g. to a
# mirror/proxy, or to httpbin when testing the retry path.
if(NOT DEFINED MODEL_URL)
    set(MODEL_URL "https://huggingface.co/ggml-org/models/resolve/main/${NAME}?download=true")
endif()
message(STATUS "Downloading ${NAME} from ${MODEL_URL}...")

# HuggingFace rate-limits by IP (HTTP 429) and answers with a short
# "Retry-After: 1" header. curl/wget honor that header over their own backoff,
# so their built-in retries fire ~1s apart and exhaust within a few seconds
# while the limit is still active (the CI failure we kept hitting). Drive the
# transfer with cmake's own file(DOWNLOAD) -- which has no retry of its own to
# be talked out of by the header -- and hand-roll the backoff here with a
# "cmake -E sleep" wait the server cannot shorten.
set(max_attempts 5)
set(rc 1)
foreach(attempt RANGE 1 ${max_attempts})
    file(DOWNLOAD "${MODEL_URL}" "${DEST}"
        TLS_VERIFY ON INACTIVITY_TIMEOUT 30 STATUS status)
    list(GET status 0 rc)
    if(rc EQUAL 0)
        break()
    endif()
    list(GET status 1 err)

    if(attempt LESS max_attempts)
        # Exponential backoff (15,30,60,120s, capped at 120) plus 0-9s of jitter
        # so parallel matrix jobs do not retry in lockstep and re-trigger the
        # limit.
        math(EXPR backoff "15 * (1 << (${attempt} - 1))")
        if(backoff GREATER 120)
            set(backoff 120)
        endif()
        string(RANDOM LENGTH 1 ALPHABET "0123456789" jitter)
        math(EXPR sleep_s "${backoff} + ${jitter}")
        message(STATUS "Download attempt ${attempt}/${max_attempts} failed (${err}); retrying in ${sleep_s}s...")
        execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep ${sleep_s})
    endif()
endforeach()

if(NOT rc EQUAL 0)
    file(REMOVE "${DEST}")
    message(FATAL_ERROR "Failed to download ${NAME} after ${max_attempts} attempts (status ${rc}: ${err})")
endif()

file(${hash_algo} "${DEST}" have_hash)
if(NOT have_hash STREQUAL hash_value)
    file(REMOVE "${DEST}")
    message(FATAL_ERROR "Hash mismatch for ${NAME}: expected ${hash_value}, got ${have_hash}")
endif()
