# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\mcserver\Desktop\pinkteam

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\mcserver\Desktop\pinkteam\build

# Include any dependencies generated for this target.
include CMakeFiles/PinkTeamApp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/PinkTeamApp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/PinkTeamApp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/PinkTeamApp.dir/flags.make

CMakeFiles/PinkTeamApp.dir/codegen:
.PHONY : CMakeFiles/PinkTeamApp.dir/codegen

CMakeFiles/PinkTeamApp.dir/streaming_queues.cpp.obj: CMakeFiles/PinkTeamApp.dir/flags.make
CMakeFiles/PinkTeamApp.dir/streaming_queues.cpp.obj: CMakeFiles/PinkTeamApp.dir/includes_CXX.rsp
CMakeFiles/PinkTeamApp.dir/streaming_queues.cpp.obj: C:/Users/mcserver/Desktop/pinkteam/streaming_queues.cpp
CMakeFiles/PinkTeamApp.dir/streaming_queues.cpp.obj: CMakeFiles/PinkTeamApp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\mcserver\Desktop\pinkteam\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/PinkTeamApp.dir/streaming_queues.cpp.obj"
	C:\msys64\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/PinkTeamApp.dir/streaming_queues.cpp.obj -MF CMakeFiles\PinkTeamApp.dir\streaming_queues.cpp.obj.d -o CMakeFiles\PinkTeamApp.dir\streaming_queues.cpp.obj -c C:\Users\mcserver\Desktop\pinkteam\streaming_queues.cpp

CMakeFiles/PinkTeamApp.dir/streaming_queues.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/PinkTeamApp.dir/streaming_queues.cpp.i"
	C:\msys64\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\mcserver\Desktop\pinkteam\streaming_queues.cpp > CMakeFiles\PinkTeamApp.dir\streaming_queues.cpp.i

CMakeFiles/PinkTeamApp.dir/streaming_queues.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/PinkTeamApp.dir/streaming_queues.cpp.s"
	C:\msys64\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\mcserver\Desktop\pinkteam\streaming_queues.cpp -o CMakeFiles\PinkTeamApp.dir\streaming_queues.cpp.s

CMakeFiles/PinkTeamApp.dir/enhancement_helpers.cpp.obj: CMakeFiles/PinkTeamApp.dir/flags.make
CMakeFiles/PinkTeamApp.dir/enhancement_helpers.cpp.obj: CMakeFiles/PinkTeamApp.dir/includes_CXX.rsp
CMakeFiles/PinkTeamApp.dir/enhancement_helpers.cpp.obj: C:/Users/mcserver/Desktop/pinkteam/enhancement_helpers.cpp
CMakeFiles/PinkTeamApp.dir/enhancement_helpers.cpp.obj: CMakeFiles/PinkTeamApp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\mcserver\Desktop\pinkteam\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/PinkTeamApp.dir/enhancement_helpers.cpp.obj"
	C:\msys64\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/PinkTeamApp.dir/enhancement_helpers.cpp.obj -MF CMakeFiles\PinkTeamApp.dir\enhancement_helpers.cpp.obj.d -o CMakeFiles\PinkTeamApp.dir\enhancement_helpers.cpp.obj -c C:\Users\mcserver\Desktop\pinkteam\enhancement_helpers.cpp

CMakeFiles/PinkTeamApp.dir/enhancement_helpers.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/PinkTeamApp.dir/enhancement_helpers.cpp.i"
	C:\msys64\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\mcserver\Desktop\pinkteam\enhancement_helpers.cpp > CMakeFiles\PinkTeamApp.dir\enhancement_helpers.cpp.i

CMakeFiles/PinkTeamApp.dir/enhancement_helpers.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/PinkTeamApp.dir/enhancement_helpers.cpp.s"
	C:\msys64\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\mcserver\Desktop\pinkteam\enhancement_helpers.cpp -o CMakeFiles\PinkTeamApp.dir\enhancement_helpers.cpp.s

# Object files for target PinkTeamApp
PinkTeamApp_OBJECTS = \
"CMakeFiles/PinkTeamApp.dir/streaming_queues.cpp.obj" \
"CMakeFiles/PinkTeamApp.dir/enhancement_helpers.cpp.obj"

# External object files for target PinkTeamApp
PinkTeamApp_EXTERNAL_OBJECTS =

PinkTeamApp.exe: CMakeFiles/PinkTeamApp.dir/streaming_queues.cpp.obj
PinkTeamApp.exe: CMakeFiles/PinkTeamApp.dir/enhancement_helpers.cpp.obj
PinkTeamApp.exe: CMakeFiles/PinkTeamApp.dir/build.make
PinkTeamApp.exe: C:/opencv_mingw/x64/mingw/lib/libopencv_gapi4100.dll.a
PinkTeamApp.exe: C:/opencv_mingw/x64/mingw/lib/libopencv_highgui4100.dll.a
PinkTeamApp.exe: C:/opencv_mingw/x64/mingw/lib/libopencv_ml4100.dll.a
PinkTeamApp.exe: C:/opencv_mingw/x64/mingw/lib/libopencv_objdetect4100.dll.a
PinkTeamApp.exe: C:/opencv_mingw/x64/mingw/lib/libopencv_photo4100.dll.a
PinkTeamApp.exe: C:/opencv_mingw/x64/mingw/lib/libopencv_stitching4100.dll.a
PinkTeamApp.exe: C:/opencv_mingw/x64/mingw/lib/libopencv_video4100.dll.a
PinkTeamApp.exe: C:/opencv_mingw/x64/mingw/lib/libopencv_videoio4100.dll.a
PinkTeamApp.exe: C:/opencv_mingw/x64/mingw/lib/libopencv_imgcodecs4100.dll.a
PinkTeamApp.exe: C:/opencv_mingw/x64/mingw/lib/libopencv_dnn4100.dll.a
PinkTeamApp.exe: C:/opencv_mingw/x64/mingw/lib/libopencv_calib3d4100.dll.a
PinkTeamApp.exe: C:/opencv_mingw/x64/mingw/lib/libopencv_features2d4100.dll.a
PinkTeamApp.exe: C:/opencv_mingw/x64/mingw/lib/libopencv_flann4100.dll.a
PinkTeamApp.exe: C:/opencv_mingw/x64/mingw/lib/libopencv_imgproc4100.dll.a
PinkTeamApp.exe: C:/opencv_mingw/x64/mingw/lib/libopencv_core4100.dll.a
PinkTeamApp.exe: CMakeFiles/PinkTeamApp.dir/linkLibs.rsp
PinkTeamApp.exe: CMakeFiles/PinkTeamApp.dir/objects1.rsp
PinkTeamApp.exe: CMakeFiles/PinkTeamApp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=C:\Users\mcserver\Desktop\pinkteam\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable PinkTeamApp.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\PinkTeamApp.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/PinkTeamApp.dir/build: PinkTeamApp.exe
.PHONY : CMakeFiles/PinkTeamApp.dir/build

CMakeFiles/PinkTeamApp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\PinkTeamApp.dir\cmake_clean.cmake
.PHONY : CMakeFiles/PinkTeamApp.dir/clean

CMakeFiles/PinkTeamApp.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\mcserver\Desktop\pinkteam C:\Users\mcserver\Desktop\pinkteam C:\Users\mcserver\Desktop\pinkteam\build C:\Users\mcserver\Desktop\pinkteam\build C:\Users\mcserver\Desktop\pinkteam\build\CMakeFiles\PinkTeamApp.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/PinkTeamApp.dir/depend

