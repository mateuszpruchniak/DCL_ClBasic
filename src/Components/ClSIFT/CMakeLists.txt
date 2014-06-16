# Include the directory itself as a path to include directories
SET(CMAKE_INCLUDE_CURRENT_DIR ON)

# Find OpenCV library files
FIND_PACKAGE( OpenCV REQUIRED )
FIND_PACKAGE( OpenCL REQUIRED )

include_directories( ${OPENCL_INCLUDE_DIRS} )


# Create a variable containing all .cpp files:
FILE(GLOB files *.cpp)

# Create an executable file from sources:
ADD_LIBRARY(ClSIFT SHARED ${files})

# Link external libraries
TARGET_LINK_LIBRARIES(ClSIFT ${DisCODe_LIBRARIES} ${OpenCV_LIBS} ${OPENCL_LIBRARIES} )

INSTALL_COMPONENT(ClSIFT)
