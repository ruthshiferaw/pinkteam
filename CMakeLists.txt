cmake_minimum_required(VERSION 3.15)
project(PinkTeamProject)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)

# Add OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the source files
add_executable(PinkTeamApp 
    streaming_queues.cpp 
    enhancement_helpers.cpp
)

# Link OpenCV libraries
target_link_libraries(PinkTeamApp ${OpenCV_LIBS})

# Include header files
target_include_directories(PinkTeamApp PRIVATE ${CMAKE_SOURCE_DIR})

# Enable optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

# Set the build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

