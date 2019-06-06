cmake_minimum_required(VERSION 2.6)
project (Renderer)
find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(PNG REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++0x -w")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

file(GLOB SOURCES "*.cpp")
add_executable(Renderer main.cpp)
target_link_libraries(Renderer opencv_core opencv_imgproc opencv_imgcodecs ${GFlags_LIBS} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARY} ${PNG_LIBRARY} gflags OpenMeshCore OpenMeshTools)
