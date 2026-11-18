# Set the minimum CMake version and policies for highest tested version
cmake_minimum_required(VERSION 3.26...3.27)
include(ExternalProject)

# Variables
set_directory_properties(PROPERTIES EP_BASE ${CMAKE_BINARY_DIR}/third_party)
get_directory_property(THIRDPARTY_BASE_PATH EP_BASE)
set(THIRDPARTY_INSTALL_PATH ${THIRDPARTY_BASE_PATH}/install)

set(TBLIS_COMMIT eb719e718976572e0ab53975f4e0c799faeb35f2)
set(BLIS_GIT_TAG a8037cfc4330030c726efaec1ffeb1f93d468aa3)
set(TAPP_COMMIT 77c32d744ee6d339f504620cc80b8679601669bc)

# Set default build type in CMake if skbuild is set
message(STATUS "CMAKE_BUILD_TYPE set to '${CMAKE_BUILD_TYPE}'")

# Set up the project and ensure there is a working C++ compiler
project(varipeps_extensions LANGUAGES CXX)

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.13
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
  COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED)

# Find jaxlib include dir
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import inspect; import jaxlib; import pathlib; p = pathlib.Path(inspect.getfile(jaxlib)); print(p.parent / 'include')"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE jaxlib_INCLUDE_DIR
  COMMAND_ERROR_IS_FATAL ANY)

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  _svd_only_u_vt

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Declare the extension safe to run without the GIL on
  # free-threaded Python (3.13t, 3.14t, ...). Without this,
  # importing it re-enables the GIL for the whole process.
  # Has no effect on regular builds
  FREE_THREADED

  # Source code goes here
  varipeps/utils/extensions/svd_ffi.cpp
)

target_include_directories(_svd_only_u_vt PRIVATE "${jaxlib_INCLUDE_DIR}")

# target_link_libraries(_svd_only_vt PRIVATE lapack)

# Install directive for scikit-build-core
install(TARGETS _svd_only_u_vt LIBRARY DESTINATION varipeps/utils/extensions)

# The tblis backend (tblis builds BLIS with plain make and GNU inline assembly)
# is not built on Windows; variPEPS runs there with JAX contractions only.
if(WIN32)
  set(VARIPEPS_WITH_TBLIS_DEFAULT OFF)
else()
  set(VARIPEPS_WITH_TBLIS_DEFAULT ON)
endif()
option(VARIPEPS_WITH_TBLIS "Build the tblis contraction backend" ${VARIPEPS_WITH_TBLIS_DEFAULT})
message(STATUS "tblis backend: ${VARIPEPS_WITH_TBLIS}")
if(VARIPEPS_WITH_TBLIS)

# Download and build the tblis lib.
set(TBLIS_ROOT ${THIRDPARTY_INSTALL_PATH}/tblis CACHE INTERNAL "")

file(WRITE
  "${CMAKE_BINARY_DIR}/blis-git-tag"
  "${BLIS_GIT_TAG}\n"
)

# BLIS sub-configurations: all of the CPU family, the matching one is chosen at
# runtime (x86_64: Intel and AMD; arm64: ARMv8 incl. SVE). Override with
# -DVARIPEPS_BLIS_CONFIG_FAMILY=... (e.g. via CMAKE_ARGS or scikit-build options).
if(NOT VARIPEPS_BLIS_CONFIG_FAMILY)
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(VARIPEPS_BLIS_CONFIG_FAMILY firestorm)  # BLIS selects firestorm on every Apple CPU anyway
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(VARIPEPS_BLIS_CONFIG_FAMILY arm64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(VARIPEPS_BLIS_CONFIG_FAMILY x86_64)
  else()
    set(VARIPEPS_BLIS_CONFIG_FAMILY auto)
  endif()
endif()
message(STATUS "BLIS config family: ${VARIPEPS_BLIS_CONFIG_FAMILY}")
# Parallel jobs for building tblis. tblis builds BLIS and its BLIS plugin with
# plain `make`, which only runs in parallel through a make jobserver, i.e. with
# a Makefile generator; under Ninja (scikit-build-core) it would compile
# serially. MAKEFLAGS passes the number of jobs to these make calls.
if(NOT VARIPEPS_TBLIS_BUILD_JOBS)
  if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL} AND NOT "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}" STREQUAL "")
    set(VARIPEPS_TBLIS_BUILD_JOBS $ENV{CMAKE_BUILD_PARALLEL_LEVEL})
  else()
    cmake_host_system_information(RESULT VARIPEPS_TBLIS_BUILD_JOBS QUERY NUMBER_OF_LOGICAL_CORES)
  endif()
endif()
message(STATUS "tblis build jobs: ${VARIPEPS_TBLIS_BUILD_JOBS}")

# Compilers for tblis/BLIS: by default those of this build, which CMake takes
# from CC/CXX (or -DCMAKE_C_COMPILER/-DCMAKE_CXX_COMPILER), otherwise from PATH;
# -DVARIPEPS_TBLIS_CC/-DVARIPEPS_TBLIS_CXX choose others for tblis/BLIS only.
# BLIS takes the compiler vendor from `$CC --version`, which reads "cc (Ubuntu
# ...)" for Debian's cc, so a compiler named cc is replaced by the gcc or clang
# next to it.
enable_language(C)
set(VARIPEPS_TBLIS_CC_DEFAULT "${CMAKE_C_COMPILER}")
get_filename_component(VARIPEPS_TBLIS_CC_DIR "${CMAKE_C_COMPILER}" DIRECTORY)
get_filename_component(VARIPEPS_TBLIS_CC_NAME "${CMAKE_C_COMPILER}" NAME)
if(VARIPEPS_TBLIS_CC_NAME STREQUAL "cc")
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND EXISTS "${VARIPEPS_TBLIS_CC_DIR}/gcc")
    set(VARIPEPS_TBLIS_CC_DEFAULT "${VARIPEPS_TBLIS_CC_DIR}/gcc")
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang" AND EXISTS "${VARIPEPS_TBLIS_CC_DIR}/clang")
    set(VARIPEPS_TBLIS_CC_DEFAULT "${VARIPEPS_TBLIS_CC_DIR}/clang")
  endif()
endif()
set(VARIPEPS_TBLIS_CC "${VARIPEPS_TBLIS_CC_DEFAULT}" CACHE STRING "C compiler for tblis/BLIS")
set(VARIPEPS_TBLIS_CXX "${CMAKE_CXX_COMPILER}" CACHE STRING "C++ compiler for tblis/BLIS")
# tblis is only called single-threaded (tblis_single, the parts run on XLA's
# pool), so BLIS and TCI need no thread model at run time. Linux keeps the tested
# OpenMP build; macOS (Apple clang has no OpenMP) builds without threading.
# hwloc only gives tblis the default thread count of calls without a
# communicator, which variPEPS does not make, hence off (no system library).
if(APPLE)
  set(VARIPEPS_TBLIS_THREADING_DEFAULT OFF)
else()
  set(VARIPEPS_TBLIS_THREADING_DEFAULT ON)
endif()
option(VARIPEPS_TBLIS_OPENMP "Build BLIS and TCI with OpenMP" ${VARIPEPS_TBLIS_THREADING_DEFAULT})
option(VARIPEPS_TBLIS_HWLOC "Build tblis with hwloc" OFF)
if(VARIPEPS_TBLIS_OPENMP)
  set(VARIPEPS_TBLIS_ARGS -DBLIS_THREAD_MODEL=openmp)
else()
  set(VARIPEPS_TBLIS_ARGS -DBLIS_THREAD_MODEL=single -DENABLE_THREAD_MODEL=none)
endif()
if(APPLE)  # only the static libraries are linked; ld64 rejects undefined symbols in a dylib
  list(APPEND VARIPEPS_TBLIS_ARGS -DENABLE_SHARED=OFF)
endif()
set(VARIPEPS_TBLIS_ENV CC=${VARIPEPS_TBLIS_CC} CXX=${VARIPEPS_TBLIS_CXX})
if(APPLE)
  # The compilers of the inner build (BLIS uses plain make) do not find the SDK
  # headers (stdio.h) without SDKROOT.
  set(VARIPEPS_TBLIS_SDKROOT "${CMAKE_OSX_SYSROOT}")
  if(NOT IS_ABSOLUTE "${VARIPEPS_TBLIS_SDKROOT}")  # empty (CMake 4) or an SDK name
    set(VARIPEPS_TBLIS_XCRUN xcrun)
    if(VARIPEPS_TBLIS_SDKROOT)
      list(APPEND VARIPEPS_TBLIS_XCRUN --sdk ${VARIPEPS_TBLIS_SDKROOT})
    endif()
    execute_process(COMMAND ${VARIPEPS_TBLIS_XCRUN} --show-sdk-path OUTPUT_VARIABLE VARIPEPS_TBLIS_SDKROOT
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()
  if(VARIPEPS_TBLIS_SDKROOT)
    list(APPEND VARIPEPS_TBLIS_ENV SDKROOT=${VARIPEPS_TBLIS_SDKROOT})
    list(APPEND VARIPEPS_TBLIS_ARGS -DCMAKE_OSX_SYSROOT=${VARIPEPS_TBLIS_SDKROOT})
  endif()
  if(CMAKE_OSX_DEPLOYMENT_TARGET)  # same minimum macOS as the extension (wheels: 11.0)
    list(APPEND VARIPEPS_TBLIS_ENV MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
    list(APPEND VARIPEPS_TBLIS_ARGS -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
  endif()
endif()
message(STATUS "tblis: OpenMP ${VARIPEPS_TBLIS_OPENMP}, hwloc ${VARIPEPS_TBLIS_HWLOC}, env ${VARIPEPS_TBLIS_ENV}")

ExternalProject_Add(tblis
    GIT_REPOSITORY https://github.com/MatthewsResearchGroup/tblis.git
    GIT_TAG ${TBLIS_COMMIT}
    GIT_SHALLOW FALSE
    GIT_SUBMODULES_RECURSE TRUE

    PATCH_COMMAND
        ${CMAKE_COMMAND} -E copy
        "${CMAKE_BINARY_DIR}/blis-git-tag"
        "<SOURCE_DIR>/blis-git-tag"
        # ExternalProject may repeat its patch step on later builds.
        COMMAND sh -c [=[git apply --reverse --check "$1" 2>/dev/null || git apply "$1"]=]
            sh "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tblis-plugin-flags.patch"

    # CMAKE_ARGS -DBLIS_CONFIG_FAMILY=x86_64 -DBLIS_THREAD_MODEL=openmp -DCMAKE_INSTALL_PREFIX=${TBLIS_ROOT} -DCMAKE_VERBOSE_MAKEFILE=ON
    CONFIGURE_COMMAND
        ${CMAKE_COMMAND} -E env ${VARIPEPS_TBLIS_ENV}
        ${CMAKE_COMMAND}
        <SOURCE_DIR>
        -DBLIS_CONFIG_FAMILY=${VARIPEPS_BLIS_CONFIG_FAMILY}
        ${VARIPEPS_TBLIS_ARGS}
        -DENABLE_HWLOC=${VARIPEPS_TBLIS_HWLOC}
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${TBLIS_ROOT}
        -DCMAKE_VERBOSE_MAKEFILE=ON
        -DCMAKE_CXX_STANDARD=20

    BUILD_COMMAND
        ${CMAKE_COMMAND} -E env ${VARIPEPS_TBLIS_ENV}
        MAKEFLAGS=-j${VARIPEPS_TBLIS_BUILD_JOBS}
        ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel ${VARIPEPS_TBLIS_BUILD_JOBS} --verbose

    INSTALL_COMMAND
        ${CMAKE_COMMAND} -E env ${VARIPEPS_TBLIS_ENV}
        ${CMAKE_COMMAND} --install <BINARY_DIR> --verbose

    # Produced by the install step; lets ninja relink the extension in the same run.
    INSTALL_BYPRODUCTS
        ${TBLIS_ROOT}/lib/libtblis.a
        ${TBLIS_ROOT}/lib/tblis/libblis_tblis.a
        ${TBLIS_ROOT}/lib/tblis/libblis_core.a
        ${TBLIS_ROOT}/lib/libtci.a
)

# Only the headers of the TAPP interface are needed, the implementation of
# TAPP on top of tblis is in varipeps/utils/extensions/tapp_tblis.cpp.
ExternalProject_Add(tapp
    GIT_REPOSITORY https://github.com/TAPPorg/reference-implementation.git
    GIT_TAG ${TAPP_COMMIT}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(tblis SOURCE_DIR BINARY_DIR)
set(TBLIS_SOURCE_DIR ${SOURCE_DIR})
set(TBLIS_BINARY_DIR ${BINARY_DIR})
ExternalProject_Get_Property(tapp SOURCE_DIR)
set(TAPP_SOURCE_DIR ${SOURCE_DIR})

nanobind_add_module(
  _tblis_contract
  NB_STATIC
  FREE_THREADED
  # -O3 instead of nanobind's -Os
  NOMINSIZE
  varipeps/utils/extensions/tblis_ffi.cpp
  varipeps/utils/extensions/tapp_tblis.cpp
)
add_dependencies(_tblis_contract tblis tapp)
# Only the Python entry point is exported (version script): the symbols of the
# static tblis, BLIS and TCI libraries and of the TAPP implementation stay
# local, so they cannot clash with another BLIS or TAPP library in the process.
set_target_properties(_tblis_contract PROPERTIES
  CXX_STANDARD 20 CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(UNIX AND NOT APPLE)  # GNU ld, gold, lld (ELF)
  target_link_options(_tblis_contract PRIVATE
    "LINKER:--exclude-libs,ALL"
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/tblis_contract.map")
  set_property(TARGET _tblis_contract APPEND PROPERTY
    LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tblis_contract.map)
elseif(APPLE)
  target_link_options(_tblis_contract PRIVATE "LINKER:-exported_symbol,_PyInit__tblis_contract")
endif()
target_include_directories(_tblis_contract PRIVATE
  "${jaxlib_INCLUDE_DIR}" "${TAPP_SOURCE_DIR}/api/include" "${TBLIS_ROOT}/include")
set(TBLIS_LIBS
  ${TBLIS_ROOT}/lib/libtblis.a
  ${TBLIS_ROOT}/lib/tblis/libblis_tblis.a
  ${TBLIS_ROOT}/lib/tblis/libblis_core.a
  ${TBLIS_ROOT}/lib/libtci.a)
if(APPLE)  # ld64 searches static archives repeatedly; atomics are in compiler-rt
  target_link_libraries(_tblis_contract PRIVATE ${TBLIS_LIBS})
else()
  target_link_libraries(_tblis_contract PRIVATE -Wl,--start-group ${TBLIS_LIBS} -Wl,--end-group atomic)
endif()
if(VARIPEPS_TBLIS_HWLOC)
  target_link_libraries(_tblis_contract PRIVATE hwloc)
endif()
if(VARIPEPS_TBLIS_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(_tblis_contract PRIVATE OpenMP::OpenMP_CXX)
endif()
install(TARGETS _tblis_contract LIBRARY DESTINATION varipeps/utils/extensions)

endif()  # VARIPEPS_WITH_TBLIS

# License texts of the third-party code compiled into the extensions (the
# BSD-3-Clause licenses require them in binary redistributions), taken from the
# sources that were built and installed next to the extensions, so that every
# binary distribution (wheel, conda package, editable install) carries them.
set(THIRD_PARTY_LICENSE_DIR varipeps/utils/extensions/third_party_licenses)
file(GLOB NANOBIND_LICENSE
  "${nanobind_DIR}/../../nanobind-*.dist-info/licenses/LICENSE*" "${nanobind_DIR}/../LICENSE*")
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import importlib.metadata as m; print(m.version('nanobind'))"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NANOBIND_VERSION ERROR_QUIET)
if(VARIPEPS_WITH_TBLIS)
  configure_file(cmake/THIRD_PARTY_NOTICES.md.in ${CMAKE_BINARY_DIR}/THIRD_PARTY_NOTICES.md @ONLY)
  install(FILES ${TBLIS_SOURCE_DIR}/LICENSE DESTINATION ${THIRD_PARTY_LICENSE_DIR}/tblis)
  foreach(dep marray stl_ext tci)
    install(FILES ${TBLIS_SOURCE_DIR}/tblis/external/${dep}/LICENSE DESTINATION ${THIRD_PARTY_LICENSE_DIR}/${dep})
  endforeach()
  install(FILES ${TBLIS_BINARY_DIR}/_deps/blis-src/LICENSE DESTINATION ${THIRD_PARTY_LICENSE_DIR}/blis)
  install(FILES ${TAPP_SOURCE_DIR}/LICENSE.md ${TAPP_SOURCE_DIR}/AUTHORS.md DESTINATION ${THIRD_PARTY_LICENSE_DIR}/tapp)
  if(UNIX AND NOT APPLE)  # GCC runtime libraries, bundled into wheels by auditwheel
    install(FILES cmake/GCC-RUNTIME-LIBRARY-EXCEPTION.txt DESTINATION ${THIRD_PARTY_LICENSE_DIR}/gcc_runtime)
  endif()
else()
  file(WRITE ${CMAKE_BINARY_DIR}/THIRD_PARTY_NOTICES.md
    "# Third-party software in the variPEPS extension modules\n\n"
    "`_svd_only_u_vt` contains nanobind ${NANOBIND_VERSION} (https://github.com/wjakob/nanobind, "
    "BSD-3-Clause, `nanobind/LICENSE`), linked statically.\n")
endif()
install(FILES ${CMAKE_BINARY_DIR}/THIRD_PARTY_NOTICES.md DESTINATION ${THIRD_PARTY_LICENSE_DIR})
if(NANOBIND_LICENSE)
  list(GET NANOBIND_LICENSE 0 NANOBIND_LICENSE)
  install(FILES ${NANOBIND_LICENSE} DESTINATION ${THIRD_PARTY_LICENSE_DIR}/nanobind RENAME LICENSE)
else()
  message(WARNING "nanobind license file not found; add it to ${THIRD_PARTY_LICENSE_DIR}/nanobind")
endif()

unset(TBLIS_DOWNLOAD_URL)
unset(TBLIS_ROOT)
