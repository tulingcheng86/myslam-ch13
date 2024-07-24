
#总编译文件，是整个VO工程的CMakeLists.txt

cmake_minimum_required(VERSION 3.1)
project(myslam)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++17 -O3 -fopenmp -pthread")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
#为了使用g2o等库,需要配以FindG2O.cmake等文件，列出FindG2O.cmake等文件的位置
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)  #指明可执行文件的输出保存位置
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)   #指明生成的库文件的输出保存位置

set(G2O_ROOT /usr/local/lib/cmake/g2o)
list(APPEND CMAKE_MODULE_PATH ${G2O_ROOT}/cmake_modules)
include_directories(${G2O_ROOT}/../include)

set(CSPARSE_INCLUDE_DIR /usr/include/suitesparse)
set(CSPARSE_LIBRARY /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so)
include_directories(${CSPARSE_INCLUDE_DIR})

add_compile_definitions(GLOG_USE_GLOG_EXPORT)




############### dependencies(找到涉及的各种库) ######################
# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# G2O
#find_package(G2O REQUIRED)
#include_directories(${G2O_INCLUDE_DIRS})
include_directories(${G2O_ROOT}/../include)
# glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})
link_directories(${GLOG_LIBRARY_DIRS})

# gtest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# gflags
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

# csparse
include_directories(${CSPARSE_INCLUDE_DIR})
find_library(CSPARSE_LIBRARY NAMES csparse HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib)
if(NOT CSPARSE_LIBRARY)
    message(FATAL_ERROR "Cannot find CSPARSE library")
endif()

#将后面的一大堆OpenCV,Pangolin....等库文件打包为THIRD_PARTY_LIBS
set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${Pangolin_LIBRARIES} GL GLU GLEW glut
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
        ${GTEST_BOTH_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        pthread
        ${CSPARSE_LIBRARY}
        )

enable_testing() #允许测试

############### source and test ######################
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(app)


