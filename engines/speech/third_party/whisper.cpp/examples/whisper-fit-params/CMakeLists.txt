# QVAC (see PATCHES.md): memory-fit preflight CLI
set(TARGET whisper-fit-params)
add_executable(${TARGET} whisper-fit-params.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE whisper ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} RUNTIME)
