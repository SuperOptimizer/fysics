cmake_minimum_required(VERSION 3.16)
project(fysics LANGUAGES C VERSION 0.1.0)

# Pure C, CPU-only, no SIMD intrinsics -- rely on aggressive auto-vectorization.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Position-independent so vc3d can link the static lib into its own shared lib.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Aggressive, vectorization-friendly flags. -march=native targets the build host;
# override with -DFYSICS_ARCH=... for a portable build.
set(FYSICS_ARCH "native" CACHE STRING "target architecture for -march")
set(FYSICS_OPT_FLAGS
  -Ofast
  -ffast-math
  -funroll-loops
  -fno-math-errno
  -march=${FYSICS_ARCH}
  -mtune=${FYSICS_ARCH}
)
# enable vectorizer reports when FYSICS_VECREPORT=ON (diagnostic)
option(FYSICS_VECREPORT "emit auto-vectorization reports" OFF)
if(FYSICS_VECREPORT)
  list(APPEND FYSICS_OPT_FLAGS -fopt-info-vec-optimized)
endif()

# OpenMP for the pipeline tile loop (the whole point of the C rewrite: no GIL).
find_package(OpenMP)

# S3 is ALWAYS on: the zarr reader speaks s3:// via the vendored libs3 (+ libcurl). Required.
find_package(CURL REQUIRED)
set(FYSICS_LIBS3 ${CMAKE_CURRENT_SOURCE_DIR}/vendor/libs3/libs3.c)

set(FYSICS_CORE_SRC fft.c paganin.c dering.c downscale.c stream.c guided.c musica.c zdrift.c
    noise.c register.c phasecorr.c zarr_io.c pipeline.c ${FYSICS_LIBS3})

# ---- the library vc3d imports (S3-enabled) ----
add_library(fysics STATIC ${FYSICS_CORE_SRC})
target_include_directories(fysics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/vendor/libs3)
target_compile_options(fysics PRIVATE ${FYSICS_OPT_FLAGS})
target_compile_definitions(fysics PRIVATE FYSICS_S3)
target_link_libraries(fysics PUBLIC m CURL::libcurl)
if(OpenMP_C_FOUND)
  target_link_libraries(fysics PUBLIC OpenMP::OpenMP_C)
endif()

# also build a shared lib (some importers prefer it)
add_library(fysics_shared SHARED ${FYSICS_CORE_SRC})
set_target_properties(fysics_shared PROPERTIES OUTPUT_NAME fysics)
target_include_directories(fysics_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/vendor/libs3)
target_compile_options(fysics_shared PRIVATE ${FYSICS_OPT_FLAGS})
target_compile_definitions(fysics_shared PRIVATE FYSICS_S3)
target_link_libraries(fysics_shared PUBLIC m CURL::libcurl)
if(OpenMP_C_FOUND)
  target_link_libraries(fysics_shared PUBLIC OpenMP::OpenMP_C)
endif()

# ---- single CLI: the whole-volume preprocessing pipeline (zarr -> zarr) ----
add_executable(fysics-process tools/fysics_process_main.c)
target_link_libraries(fysics-process PRIVATE fysics m)
target_compile_options(fysics-process PRIVATE ${FYSICS_OPT_FLAGS})
if(OpenMP_C_FOUND)
  target_link_libraries(fysics-process PRIVATE OpenMP::OpenMP_C)
endif()
install(TARGETS fysics-process RUNTIME DESTINATION bin)
install(PROGRAMS tools/read_meta.py DESTINATION bin)
# copy the python metadata stub next to the binary in the build tree too
add_custom_command(TARGET fysics-process POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/read_meta.py
          $<TARGET_FILE_DIR:fysics-process>/read_meta.py)

# ---- mca_export: FUSED single-pass preprocess -> matter-compressor archive, reads local OR s3:// ----
# Links the S3-enabled fysics lib + the vendored matter-compressor codec (callback-driven build).
add_executable(mca_export
  tools/mca_export.c
  vendor/matter/matter_compressor.c
  vendor/matter/c3d.c)
target_include_directories(mca_export PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/vendor/matter)
target_compile_options(mca_export PRIVATE ${FYSICS_OPT_FLAGS})
set_source_files_properties(vendor/matter/matter_compressor.c
                            vendor/matter/c3d.c
  PROPERTIES COMPILE_FLAGS "-std=c23")
find_package(Threads REQUIRED)
# zstd: mc_volume's zarr reader (inside the unified matter_compressor.c) decompresses zstd chunks
target_link_libraries(mca_export PRIVATE fysics CURL::libcurl m Threads::Threads zstd)
if(OpenMP_C_FOUND)
  target_link_libraries(mca_export PRIVATE OpenMP::OpenMP_C)
endif()
add_custom_command(TARGET mca_export POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/read_meta.py
          $<TARGET_FILE_DIR:mca_export>/read_meta.py)
install(TARGETS mca_export RUNTIME DESTINATION bin)

# ---- tests ----
enable_testing()
add_executable(test_fysics tests/test_fysics.c)
target_link_libraries(test_fysics PRIVATE fysics)
target_compile_options(test_fysics PRIVATE ${FYSICS_OPT_FLAGS})
add_test(NAME fysics_correctness COMMAND test_fysics)

# ---- bench (optional) ----
add_executable(bench_fysics tests/bench_fysics.c)
target_link_libraries(bench_fysics PRIVATE fysics)
target_compile_options(bench_fysics PRIVATE ${FYSICS_OPT_FLAGS})

install(TARGETS fysics fysics_shared
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(FILES fysics.h DESTINATION include)
