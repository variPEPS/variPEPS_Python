# Set the minimum CMake version and policies for highest tested version
cmake_minimum_required(VERSION 3.15...3.27)

# Set default build type in CMake if skbuild is set
message(STATUS "CMAKE_BUILD_TYPE set to '${CMAKE_BUILD_TYPE}'")

# Set up the project and ensure there is a working C++ compiler
project(varipeps_extensions LANGUAGES CXX)

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
execute_process(
  COMMAND /usr/bin/env python3 -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Find jaxlib include dir
IF (WIN32)
  execute_process(
    COMMAND python -c "import inspect; import jaxlib; import pathlib; p = pathlib.Path(inspect.getfile(jaxlib)); print(p.parent / 'include')"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE jaxlib_INCLUDE_DIR)
ELSE()
  execute_process(
    COMMAND /usr/bin/env python3 -c "import inspect; import jaxlib; import pathlib; p = pathlib.Path(inspect.getfile(jaxlib)); print(p.parent / 'include')"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE jaxlib_INCLUDE_DIR)
ENDIF()


# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  _svd_only_u_vt

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  varipeps/utils/extensions/svd_ffi.cpp
)

target_include_directories(_svd_only_u_vt PRIVATE "${jaxlib_INCLUDE_DIR}")

# target_link_libraries(_svd_only_vt PRIVATE lapack)

# Install directive for scikit-build-core
install(TARGETS _svd_only_u_vt LIBRARY DESTINATION varipeps/utils/extensions)
