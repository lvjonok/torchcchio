# torchcchio

1. Setup environment

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
pip3 install pin
```

2. Build

```bash
export CMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'):$(python3 -m cmeel cmake)
mkdir build && cd build
cmake ..
make
```

3. Run

```bash
./main
```
