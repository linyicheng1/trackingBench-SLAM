cmake_minimum_required(VERSION 2.8)
project(tracking_bench)

#set( CMAKE_BUILD_TYPE Release )
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_definitions(-DCOMPILEDWITHC11)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(G2O REQUIRED )
find_package(Cholmod REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${CHOLMOD_INCLUDE_DIR}
        ${Sophus_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME}
        src/Viewer.cpp
        src/types/Frame.cpp
        src/types/Map.cpp
        src/types/MapPoint.cpp
        src/extractors/ORBextractor.cpp
        src/extractors/FASTextractor.cpp
        src/matchers/matcher.cpp
        src/camera/CameraModel.cpp
        src/mapping/DepthFilter.cpp
        src/mapping/LocalBA.cpp
        third_part/fast_lib/src/fast_10.cpp
        third_part/fast_lib/src/fast_10_score.cpp
        third_part/fast_lib/src/faster_corner_10_sse.cpp
        third_part/fast_lib/src/nonmax_3x3.cpp
        )

file(GLOB G2O_LIB ${PROJECT_SOURCE_DIR}/third_part/g2o/lib/*.so)

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/third_part/DBoW2/lib/libDBoW2.so
        ${G2O_LIB}
        ${CHOLMOD_LIBRARIES}
        )

# Build examples

add_executable(main main.cpp)
target_link_libraries(main ${PROJECT_NAME} )

add_executable(test_viewer test/test_viewer.cpp)
target_link_libraries(test_viewer ${PROJECT_NAME} )

add_executable(test_frame test/test_frame.cpp)
target_link_libraries(test_frame ${PROJECT_NAME})

add_executable(test_matcher test/test_matcher.cpp)
target_link_libraries(test_matcher ${PROJECT_NAME})

add_executable(test_VO test/test_vo.cpp)
target_link_libraries(test_VO ${PROJECT_NAME})

add_executable(test_projection test/test_projection.cpp)
target_link_libraries(test_projection ${PROJECT_NAME})

