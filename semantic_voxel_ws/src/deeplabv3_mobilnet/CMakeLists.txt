cmake_minimum_required(VERSION 3.0.2)
project(deeplabv3_mobilnet)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "-O3 -pipe -Wall -Wextra -latomic -Wdeprecated-declarations")

SET(MODULE_NAME algorithm_sdk)
add_compile_options(-fPIC)
set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
)
find_package (OpenCV REQUIRED NO_CMAKE_FIND_ROOT_PATH)
## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES deeplabv3_mobilnet
#  CATKIN_DEPENDS image_transport roscpp rospy
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/deeplabv3_mobilnet.cpp
# )
find_package(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;)
# SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61,;-std=c++11;)
SET(GPU_ARCHS 61)
include_directories(${CUDA_INCLUDE_DIRS})
# If GPU_ARCHS is user defined, build specifically for specified SM
if (DEFINED GPU_ARCHS)
 message(STATUS "GPU_ARCH defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
 separate_arguments(GPU_ARCHS)
# Else list out default SMs to build for.
else()
 message(STATUS "GPU_ARCH is not defined. Generating CUDA code for default SMs.")
 list(APPEND GPU_ARCHS
     #35  # 由于改架构不支持half运算，会导致编译出现 error-fp16-cuda-compilation-error
     # 52 # 960显卡，arch = 52
     53 # Nano ARCH
     61 # 1070, 1080
     62 # Tx2 ARCH
     70
     )
 # Add SM 75 for CUDA versions >= 10.0
 if (NOT ("${CUDA_VERSION}" VERSION_LESS "10.0"))
   list(APPEND GPU_ARCHS
     75)
 endif()
endif()
include_directories(${CUDA_INCLUDE_DIRS})

# 平台识别，以适应不同的库环境，获取不同的唯一识别码

IF(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l")
    MESSAGE(STATUS "Now is TK1 UNIX-like OS's.")
    add_definitions(-DTK1_DEVICE)
    SET(MODULE_NAME "${MODULE_NAME}_armv7l")
ELSEIF(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    MESSAGE(STATUS "Now is jetson TX2 or Nano UNIX-like OS's.")
    #Set library path
    add_definitions(-DARM64_DEVICE)
    set(TensorRT_DIR "/usr")
    if(DEFINED TensorRT_DIR)
        include_directories("${TensorRT_DIR}/include/aarch64-linux-gnu/" )
        link_directories("${TensorRT_DIR}/lib/aarch64-linux-gnu/")
        link_directories("/usr/local/cuda/lib64")
    else()
        set(TensorRT_DIR /usr)
        include_directories("${TensorRT_DIR}/include/aarch64-linux-gnu/" )
        link_directories("${TensorRT_DIR}/lib/aarch64-linux-gnu/")
        link_directories("/usr/local/cuda/lib64")
    endif(DEFINED TensorRT_DIR)

    SET(MODULE_NAME "${MODULE_NAME}_aarch64")
ELSE()
    IF(WIN32)
        MESSAGE(STATUS "Now is WIN32 OS's.")
        add_definitions(-DWIN32)
        add_definitions(-DDLL_EXPORTS)
        if(DEFINED TensorRT_DIR)
            message(STATUS "tensorrt path:" ${TensorRT_DIR})
            include_directories("${TensorRT_DIR}/include" )
            link_directories("${TensorRT_DIR}/lib")
    #        link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/lib/x64")
        else()
            MESSAGE(STATUS "set tensorrt dir.")
            set(TensorRT_DIR C:/TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.2.cudnn7.6/TensorRT-7.0.0.11/)
            include_directories("${TensorRT_DIR}/include" )
            link_directories("${TensorRT_DIR}/lib")
            link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/lib/x64")
        endif(DEFINED TensorRT_DIR)
        SET(MODULE_NAME "${MODULE_NAME}_x64")
    ELSE()
        MESSAGE(STATUS "Now is Desktop UNIX-like OS's.")
        if(DEFINED TensorRT_DIR)
            include_directories("${TensorRT_DIR}/include" )
            link_directories("${TensorRT_DIR}/lib")
            link_directories("/usr/local/cuda/lib64")
        else()
            set(TensorRT_DIR /usr)
            include_directories("${TensorRT_DIR}/include" )
            link_directories("${TensorRT_DIR}/lib")
            link_directories("/usr/local/cuda/lib64")
        endif(DEFINED TensorRT_DIR)
        ENDIF()

        if(NOT DEFINED OpenCV_DIR)
            set(OpenCV_DIR /usr/share/OpenCV)

        endif(NOT DEFINED OpenCV_DIR)



ENDIF()


find_package (OpenCV REQUIRED NO_CMAKE_FIND_ROOT_PATH)
if(OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
endif()



file(GLOB_RECURSE CURRENT_HEADERS src/export_inc/*.h  src/export_inc/*.hpp src/private_inc/*.h  src/private_inc/*.hpp src/*.h src/*.hpp  src/inc/*.h)
file(GLOB CURRENT_SOURCES src/src/*.cpp src/src/*.c  )
include_directories(
        src/export_inc/
        src/private_inc/
        src/src/
          )
IF(WIN32)
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include")
ELSE()
include_directories("/usr/local/cuda/include")
ENDIF()
# 库
cuda_add_library( ${MODULE_NAME} SHARED
  ${CURRENT_HEADERS} ${CURRENT_SOURCES} ${COMMON_SOURCES}
  )




target_compile_features(${MODULE_NAME} PUBLIC cxx_range_for )
link_directories("/usr/local/cuda/lib64" "/usr/lib" /usr/lib/x86_64-linux-gnu )
set_target_properties(${MODULE_NAME} PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )


target_link_libraries(${MODULE_NAME} "nvinfer" "nvinfer_plugin"   "nvonnxparser"  #${CUDA_LIBRARIES}
        )


# 测试程序
add_executable(client src/main.cpp ${TEST_SOURCES})
target_link_libraries(client ${MODULE_NAME}  ${OpenCV_LIBS} ${catkin_LIBRARIES} "pthread")
#add_executable(client src/main.cpp ${TEST_SOURCES})
#target_link_libraries(client ${catkin_INCLUDE_DIRS} ${MODULE_NAME}  ${OpenCV_LIBS} "pthread")
## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/deeplabv3_mobilnet_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_deeplabv3_mobilnet.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
