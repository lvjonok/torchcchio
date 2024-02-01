cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME torchcchio)
set(PROJECT_DESCRIPTION
        "Example of pinocchio usage with torch")
set(PROJECT_URL "http://github.com/lvjonok/torchcchio")

project(${PROJECT_NAME} LANGUAGES CXX)

# pinocchio dependency
find_package(pinocchio REQUIRED)
# torch dependency
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${TORCH_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

# Create the main executable
add_executable(main main.cpp)

# Link main to its dependencies
target_link_libraries(main PUBLIC "${TORCH_LIBRARIES}" pinocchio::pinocchio)

##cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)';$(python3 -m cmeel cmake)` -DPINOCCHIO_WITH_URDFDOM=TRUE ..
##export CMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'):$(python3 -m cmeel cmake)
