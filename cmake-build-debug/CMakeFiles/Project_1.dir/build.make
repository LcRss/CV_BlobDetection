# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/clion-2017.3.4/bin/cmake/bin/cmake

# The command to remove a file.
RM = /opt/clion-2017.3.4/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/luca/CLionProjects/Project_1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/luca/CLionProjects/Project_1/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Project_1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Project_1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Project_1.dir/flags.make

CMakeFiles/Project_1.dir/main.cpp.o: CMakeFiles/Project_1.dir/flags.make
CMakeFiles/Project_1.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luca/CLionProjects/Project_1/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Project_1.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Project_1.dir/main.cpp.o -c /home/luca/CLionProjects/Project_1/main.cpp

CMakeFiles/Project_1.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Project_1.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/luca/CLionProjects/Project_1/main.cpp > CMakeFiles/Project_1.dir/main.cpp.i

CMakeFiles/Project_1.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Project_1.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/luca/CLionProjects/Project_1/main.cpp -o CMakeFiles/Project_1.dir/main.cpp.s

CMakeFiles/Project_1.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/Project_1.dir/main.cpp.o.requires

CMakeFiles/Project_1.dir/main.cpp.o.provides: CMakeFiles/Project_1.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/Project_1.dir/build.make CMakeFiles/Project_1.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/Project_1.dir/main.cpp.o.provides

CMakeFiles/Project_1.dir/main.cpp.o.provides.build: CMakeFiles/Project_1.dir/main.cpp.o


CMakeFiles/Project_1.dir/Tools.cpp.o: CMakeFiles/Project_1.dir/flags.make
CMakeFiles/Project_1.dir/Tools.cpp.o: ../Tools.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luca/CLionProjects/Project_1/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Project_1.dir/Tools.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Project_1.dir/Tools.cpp.o -c /home/luca/CLionProjects/Project_1/Tools.cpp

CMakeFiles/Project_1.dir/Tools.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Project_1.dir/Tools.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/luca/CLionProjects/Project_1/Tools.cpp > CMakeFiles/Project_1.dir/Tools.cpp.i

CMakeFiles/Project_1.dir/Tools.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Project_1.dir/Tools.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/luca/CLionProjects/Project_1/Tools.cpp -o CMakeFiles/Project_1.dir/Tools.cpp.s

CMakeFiles/Project_1.dir/Tools.cpp.o.requires:

.PHONY : CMakeFiles/Project_1.dir/Tools.cpp.o.requires

CMakeFiles/Project_1.dir/Tools.cpp.o.provides: CMakeFiles/Project_1.dir/Tools.cpp.o.requires
	$(MAKE) -f CMakeFiles/Project_1.dir/build.make CMakeFiles/Project_1.dir/Tools.cpp.o.provides.build
.PHONY : CMakeFiles/Project_1.dir/Tools.cpp.o.provides

CMakeFiles/Project_1.dir/Tools.cpp.o.provides.build: CMakeFiles/Project_1.dir/Tools.cpp.o


# Object files for target Project_1
Project_1_OBJECTS = \
"CMakeFiles/Project_1.dir/main.cpp.o" \
"CMakeFiles/Project_1.dir/Tools.cpp.o"

# External object files for target Project_1
Project_1_EXTERNAL_OBJECTS =

Project_1: CMakeFiles/Project_1.dir/main.cpp.o
Project_1: CMakeFiles/Project_1.dir/Tools.cpp.o
Project_1: CMakeFiles/Project_1.dir/build.make
Project_1: /usr/local/lib/libopencv_stitching.so.3.3.1
Project_1: /usr/local/lib/libopencv_superres.so.3.3.1
Project_1: /usr/local/lib/libopencv_videostab.so.3.3.1
Project_1: /usr/local/lib/libopencv_aruco.so.3.3.1
Project_1: /usr/local/lib/libopencv_bgsegm.so.3.3.1
Project_1: /usr/local/lib/libopencv_bioinspired.so.3.3.1
Project_1: /usr/local/lib/libopencv_ccalib.so.3.3.1
Project_1: /usr/local/lib/libopencv_cvv.so.3.3.1
Project_1: /usr/local/lib/libopencv_dpm.so.3.3.1
Project_1: /usr/local/lib/libopencv_face.so.3.3.1
Project_1: /usr/local/lib/libopencv_freetype.so.3.3.1
Project_1: /usr/local/lib/libopencv_fuzzy.so.3.3.1
Project_1: /usr/local/lib/libopencv_img_hash.so.3.3.1
Project_1: /usr/local/lib/libopencv_line_descriptor.so.3.3.1
Project_1: /usr/local/lib/libopencv_optflow.so.3.3.1
Project_1: /usr/local/lib/libopencv_reg.so.3.3.1
Project_1: /usr/local/lib/libopencv_rgbd.so.3.3.1
Project_1: /usr/local/lib/libopencv_saliency.so.3.3.1
Project_1: /usr/local/lib/libopencv_stereo.so.3.3.1
Project_1: /usr/local/lib/libopencv_structured_light.so.3.3.1
Project_1: /usr/local/lib/libopencv_surface_matching.so.3.3.1
Project_1: /usr/local/lib/libopencv_tracking.so.3.3.1
Project_1: /usr/local/lib/libopencv_xfeatures2d.so.3.3.1
Project_1: /usr/local/lib/libopencv_ximgproc.so.3.3.1
Project_1: /usr/local/lib/libopencv_xobjdetect.so.3.3.1
Project_1: /usr/local/lib/libopencv_xphoto.so.3.3.1
Project_1: /usr/local/lib/libopencv_shape.so.3.3.1
Project_1: /usr/local/lib/libopencv_photo.so.3.3.1
Project_1: /usr/local/lib/libopencv_calib3d.so.3.3.1
Project_1: /usr/local/lib/libopencv_phase_unwrapping.so.3.3.1
Project_1: /usr/local/lib/libopencv_video.so.3.3.1
Project_1: /usr/local/lib/libopencv_datasets.so.3.3.1
Project_1: /usr/local/lib/libopencv_plot.so.3.3.1
Project_1: /usr/local/lib/libopencv_text.so.3.3.1
Project_1: /usr/local/lib/libopencv_dnn.so.3.3.1
Project_1: /usr/local/lib/libopencv_features2d.so.3.3.1
Project_1: /usr/local/lib/libopencv_flann.so.3.3.1
Project_1: /usr/local/lib/libopencv_highgui.so.3.3.1
Project_1: /usr/local/lib/libopencv_ml.so.3.3.1
Project_1: /usr/local/lib/libopencv_videoio.so.3.3.1
Project_1: /usr/local/lib/libopencv_imgcodecs.so.3.3.1
Project_1: /usr/local/lib/libopencv_objdetect.so.3.3.1
Project_1: /usr/local/lib/libopencv_imgproc.so.3.3.1
Project_1: /usr/local/lib/libopencv_core.so.3.3.1
Project_1: CMakeFiles/Project_1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/luca/CLionProjects/Project_1/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable Project_1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Project_1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Project_1.dir/build: Project_1

.PHONY : CMakeFiles/Project_1.dir/build

CMakeFiles/Project_1.dir/requires: CMakeFiles/Project_1.dir/main.cpp.o.requires
CMakeFiles/Project_1.dir/requires: CMakeFiles/Project_1.dir/Tools.cpp.o.requires

.PHONY : CMakeFiles/Project_1.dir/requires

CMakeFiles/Project_1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Project_1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Project_1.dir/clean

CMakeFiles/Project_1.dir/depend:
	cd /home/luca/CLionProjects/Project_1/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/luca/CLionProjects/Project_1 /home/luca/CLionProjects/Project_1 /home/luca/CLionProjects/Project_1/cmake-build-debug /home/luca/CLionProjects/Project_1/cmake-build-debug /home/luca/CLionProjects/Project_1/cmake-build-debug/CMakeFiles/Project_1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Project_1.dir/depend

