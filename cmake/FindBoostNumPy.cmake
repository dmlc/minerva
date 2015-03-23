MACRO(BOOSTNUMPY_REPORT_NOT_FOUND REASON_MSG)
  UNSET(BOOSTNUMPY_FOUND)
  UNSET(BOOSTNUMPY_INCLUDE_DIRS)
  UNSET(BOOSTNUMPY_LIBRARIES)
  # Make results of search visible in the CMake GUI if Boost.numpy has not
  # been found so that user does not have to toggle to advanced view.
  MARK_AS_ADVANCED(CLEAR BOOSTNUMPY_INCLUDE_DIR
                         BOOSTNUMPY_LIBRARY)
  # Note <package>_FIND_[REQUIRED/QUIETLY] variables defined by FindPackage()
  # use the camelcase library name, not uppercase.
  IF (BoostNumPy_FIND_QUIETLY)
    MESSAGE(STATUS "Failed to find Boost.NumPy - " ${REASON_MSG} ${ARGN})
  ELSEIF (BoostNumPy_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Failed to find Boost.NumPy - " ${REASON_MSG} ${ARGN})
  ELSE()
    # Neither QUIETLY nor REQUIRED, use no priority which emits a message
    # but continues configuration and allows generation.
    MESSAGE("-- Failed to find Boost.NumPy - " ${REASON_MSG} ${ARGN})
  ENDIF ()
ENDMACRO(BOOSTNUMPY_REPORT_NOT_FOUND)

# Search user-installed locations first, so that we prefer user installs
# to system installs where both exist.
#
# TODO: Add standard Windows search locations for Boost.NumPy.
LIST(APPEND BOOSTNUMPY_CHECK_INCLUDE_DIRS
  /usr/local/include
  /usr/local/homebrew/include # Mac OS X
  /opt/local/var/macports/software # Mac OS X.
  /opt/local/include
  /usr/include
  ${BOOSTNUMPY_ROOT_DIR}/include)
LIST(APPEND BOOSTNUMPY_CHECK_LIBRARY_DIRS
  /usr/local/lib
  /usr/local/homebrew/lib # Mac OS X.
  /opt/local/lib
  /usr/lib
  ${BOOSTNUMPY_ROOT_DIR}/lib)

# Search supplied hint directories first if supplied.
FIND_PATH(BOOSTNUMPY_INCLUDE_DIR
  NAMES boost/numpy.hpp
  PATHS ${BOOSTNUMPY_INCLUDE_DIR_HINTS}
  ${BOOSTNUMPY_CHECK_INCLUDE_DIRS})
IF (NOT BOOSTNUMPY_INCLUDE_DIR OR
    NOT EXISTS ${BOOSTNUMPY_INCLUDE_DIR})
  BOOSTNUMPY_REPORT_NOT_FOUND(
    "Could not find Boost.NumPy include directory, set BOOSTNUMPY_INCLUDE_DIR "
    "to directory containing boost/numpy.hpp")
ENDIF (NOT BOOSTNUMPY_INCLUDE_DIR OR
       NOT EXISTS ${BOOSTNUMPY_INCLUDE_DIR})

FIND_LIBRARY(BOOSTNUMPY_LIBRARY NAMES boost_numpy
  PATHS ${BOOSTNUMPY_LIBRARY_DIR_HINTS}
  ${BOOSTNUMPY_CHECK_LIBRARY_DIRS})
IF (NOT BOOSTNUMPY_LIBRARY OR
    NOT EXISTS ${BOOSTNUMPY_LIBRARY})
  BOOSTNUMPY_REPORT_NOT_FOUND(
    "Could not find Boost.NumPy library, set BOOSTNUMPY_LIBRARY "
    "to full path to libboost_numpy")
ENDIF (NOT BOOSTNUMPY_LIBRARY OR
       NOT EXISTS ${BOOSTNUMPY_LIBRARY})

# Mark internally as found, then verify. BOOSTNUMPY_REPORT_NOT_FOUND() unsets
# if called.
SET(BOOSTNUMPY_FOUND TRUE)

# BoostNumPy does not seem to provide any record of the version in its
# source tree, thus cannot extract version.

# Catch case when caller has set BOOSTNUMPY_INCLUDE_DIR in the cache / GUI and
# thus FIND_[PATH/LIBRARY] are not called, but specified locations are
# invalid, otherwise we would report the library as found.
IF (BOOSTNUMPY_INCLUDE_DIR AND
    NOT EXISTS ${BOOSTNUMPY_INCLUDE_DIR}/boost/numpy.hpp)
  BOOSTNUMPY_REPORT_NOT_FOUND(
    "Caller defined BOOSTNUMPY_INCLUDE_DIR:"
    " ${BOOSTNUMPY_INCLUDE_DIR} does not contain boost/numpy.hpp header.")
ENDIF (BOOSTNUMPY_INCLUDE_DIR AND
       NOT EXISTS ${BOOSTNUMPY_INCLUDE_DIR}/boost/numpy.hpp)
STRING(TOLOWER "${BOOSTNUMPY_LIBRARY}" LOWERCASE_BOOSTNUMPY_LIBRARY)
IF (BOOSTNUMPY_LIBRARY AND
    NOT "${LOWERCASE_BOOSTNUMPY_LIBRARY}" MATCHES ".*boost_numpy[^/]*")
  BOOSTNUMPY_REPORT_NOT_FOUND(
    "Caller defined BOOSTNUMPY_LIBRARY: "
    "${BOOSTNUMPY_LIBRARY} does not match boost_numpy.")
ENDIF (BOOSTNUMPY_LIBRARY AND
       NOT "${LOWERCASE_BOOSTNUMPY_LIBRARY}" MATCHES ".*boost_numpy[^/]*")

# Set standard CMake FindPackage variables if found.
IF (BOOSTNUMPY_FOUND)
  SET(BOOSTNUMPY_INCLUDE_DIRS ${BOOSTNUMPY_INCLUDE_DIR})
  SET(BOOSTNUMPY_LIBRARIES ${BOOSTNUMPY_LIBRARY})
ENDIF (BOOSTNUMPY_FOUND)

# Handle REQUIRED / QUIET optional arguments.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(BoostNumPy DEFAULT_MSG
  BOOSTNUMPY_INCLUDE_DIRS BOOSTNUMPY_LIBRARIES)

# Only mark internal variables as advanced if we found Boost.Numpy, otherwise
# leave them visible in the standard GUI for the user to set manually.
IF (BOOSTNUMPY_FOUND)
  MARK_AS_ADVANCED(FORCE BOOSTNUMPY_INCLUDE_DIR
                         BOOSTNUMPY_LIBRARY)
ENDIF (BOOSTNUMPY_FOUND)
