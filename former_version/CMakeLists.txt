project( ConvNet )
find_package( OpenCV REQUIRED )
add_executable( ConvNet ConvNet )
target_link_libraries (ConvNet ${OpenCV_LIBS})