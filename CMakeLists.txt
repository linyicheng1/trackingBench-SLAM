cmake_minimum_required(VERSION 2.8)
project(tracking_bench)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_definitions(-DCOMPILEDWITHC11)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME}
        src/Viewer.cpp
        src/types/Frame.cpp
        src/types/KeyFrame.cpp
        src/types/Map.cpp
        src/types/MapPoint.cpp
        src/extractors/extractor.cpp
        src/extractors/ORBextractor.cpp
        src/extractors/FASTextractor.cpp
        src/matchers/matcher.cpp
        third_part/fast_lib/src/fast_10.cpp
        third_part/fast_lib/src/fast_10_score.cpp
        third_part/fast_lib/src/faster_corner_10_sse.cpp
        third_part/fast_lib/src/nonmax_3x3.cpp
        )

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
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
