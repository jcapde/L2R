cmake_minimum_required (VERSION 3.0)
set (CMAKE_CXX_STANDARD 14)
project (pfasamplers)
set(CMAKE_CXX_FLAGS "-O3  -fopenmp")
#set(CMAKE_CXX_FLAGS "-fopenmp")

#SET(Boost_USE_STATIC_LIBS ON)

FIND_PACKAGE(PythonLibs 3.5 REQUIRED)
FIND_PACKAGE(PythonInterp 3.5 REQUIRED)

INCLUDE_DIRECTORIES(BEFORE ${PYTHON_INCLUDE_DIRS})

FIND_PACKAGE(GSL REQUIRED)
FIND_PACKAGE(BLAS REQUIRED)

FIND_PACKAGE(Boost 1.58 COMPONENTS log system python python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} math_tr1 chrono REQUIRED)

ADD_DEFINITIONS(-DBOOST_LOG_DYN_LINK)
INCLUDE_DIRECTORIES(AFTER ${Boost_INCLUDE_DIRS})


include_directories(BEFORE
        /usr/include/eigen3/
        /usr/include/eigen3/unsupported/
        )

add_executable (test-script test.cpp loader.cpp pfa_inference.cpp exact.cpp partitions.cpp samplers.cpp utils.cpp est_Exact.cpp est_DS.cpp est_HM.cpp est_L2R_E.cpp est_L2R.cpp est_HM.cpp)
add_library(pfa SHARED pfa_python.cpp loader.cpp pfa_inference.cpp exact.cpp partitions.cpp samplers.cpp utils.cpp est_Exact.cpp est_DS.cpp est_HM.cpp est_L2R_E.cpp est_L2R.cpp est_HM.cpp)

SET_TARGET_PROPERTIES(pfa PROPERTIES PREFIX "")
LINK_DIRECTORIES("/usr/lib/python3.6")
LINK_DIRECTORIES("/usr/lib/python3.5")

target_link_libraries(pfa boost_python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} ${Boost_LOG_LIBRARY} ${Boost_MATH_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ${GSL_LIBRARY} ${BLAS_LIBRARIES} matio)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${GSL_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${BLAS_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(test-script ${Boost_LOG_LIBRARY} ${Boost_MATH_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} matio)

install(TARGETS test-script RUNTIME DESTINATION bin COMPONENT test-script)
install(TARGETS pfa LIBRARY DESTINATION lib COMPONENT pfa)

add_custom_target(pfa-install
        DEPENDS pfa
        COMMAND
        "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=pfa
        -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
        )

