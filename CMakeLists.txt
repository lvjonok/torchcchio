cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torchcchio)

find_package(Torch REQUIRED)
find_package(pinocchio REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# include_directories(
#   # /home/leo/github.com/lvjonok/torchcchio/venv/lib/python3.10/site-packages/cmeel.prefix/include
#   /home/leo/github.com/lvjonok/torchcchio/venv/lib/python3.10/site-packages/cmeel.prefix/include/eigen3
#   # ${EIGEN3_INCLUDE_DIR}
#   ${PINOCCHIO_INCLUDE_DIRS}
# )

# # add_executable(
# #   invdyn
# #   invdyn.cpp
# #   ${${PROJECT_NAME}_HEADERS}
# # )
# # TARGET_LINK_LIBRARIES(${EXAMPLE_NAME} PUBLIC ${PROJECT_NAME})
# add_executable(invdyn invdyn.cpp ${${PROJECT_NAME}_HEADERS})
# target_link_libraries(invdyn PUBLIC "${TORCH_LIBRARIES}" ${PINOCCHIO_LIBRARIES})
# set_property(TARGET invdyn PROPERTY CXX_STANDARD 17)

# # target_link_libraries(
#   # ${PY_NAME}
#   # PRIVATE
#   # casadi
#   # ${casadi_LIBRARIES}
#   # ${PINOCCHIO_LIBRARIES}

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)