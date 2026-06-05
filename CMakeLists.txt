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

# ---- the library vc3d imports ----
add_library(fysics STATIC fft.c paganin.c denoise.c rings.c mask.c clahe.c glcae.c process.c stream.c guided.c musica.c fsc.c zdrift.c sheetness.c noise.c register.c phasecorr.c compact.c fuse.c)
target_include_directories(fysics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(fysics PRIVATE ${FYSICS_OPT_FLAGS})
target_link_libraries(fysics PUBLIC m)

# also build a shared lib (some importers prefer it)
add_library(fysics_shared SHARED fft.c paganin.c denoise.c rings.c mask.c clahe.c glcae.c process.c stream.c guided.c musica.c fsc.c zdrift.c sheetness.c noise.c register.c phasecorr.c compact.c fuse.c)
set_target_properties(fysics_shared PROPERTIES OUTPUT_NAME fysics)
target_include_directories(fysics_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(fysics_shared PRIVATE ${FYSICS_OPT_FLAGS})
target_link_libraries(fysics_shared PUBLIC m)

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
