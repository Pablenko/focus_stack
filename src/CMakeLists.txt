set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -pedantic -Wextra -g")
set(APPLICATION_NAME image_magic)

find_package(OpenCV REQUIRED)

set(APPLICATION_SOURCES
  main.cpp
  img_algs.cpp
  common.cpp
)

set(APPLICATION_LIBRARIES
  ${OpenCV_LIBRARIES}
  stdc++fs
  )

add_executable(${APPLICATION_NAME} ${APPLICATION_SOURCES})
target_link_libraries(${APPLICATION_NAME} ${APPLICATION_LIBRARIES})
