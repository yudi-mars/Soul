# NeuSim

## Prerequisites

* g++ >= 14
* cmake >= 3.25
* uv
* python >= 3.12

## Build

`${NeuSimHome}`: the root directory of NeuSim

`${NoximHome}=${NeuSimHome}/../noxim`: the root directory of Noxim

### Noxim

1. unzip Noxim.
```bash
unzip -d noxim.zip ${NoximHome}
```

2. compile systemc-2.3.1 and yaml-cpp-0.8.0
```bash
cd ${NoximHome}/third_party/systemc-2.3.1
./configure --prefix=${NoximHome}/bin/libs/systemc-2.3.1
make -j$(nproc)
make install
```

```bash
cd ${NoximHome}/third_party/yaml-cpp-0.8.0
cmake -S . -B build
cmake --build build
cmake --install build --prefix=${NoximHome}/bin/libs/yaml-cpp
```

3. compile Noxim
```bash
cd ${NoximHome}/bin
make -j$(nproc)
```

### NeuSim
1. compile NeuSim
```bash
cd ${NeuSimHome}
cmake -DCMAKE_BUILD_TYPE:STRING=debug -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc-14 -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++-14 -S . -B build/cmake-build-debug
cmake --build build/cmake-build-debug
```

2. run tests
```bash
uv sync
uv run tests/noc_tests/test.py
```

## Contribute to NeuSim

### Prerequisties

1. pre-commit hooks
```bash
uv sync
pre-commit install
```

### Development
1. checkout a new branch `feature/xxx`
```bash
git checkout -b feature/xxx
```

2. commit your changes (pre-commit will automatically format your code)
```bash
git commit -m "feat: xxx"
```

3. push to remote `feature/xxx`
```bash
git push
```

4. before pull request, please rebase to current `dev` branch and resolve conflicts
```bash
git checkout dev
git pull
git checkout feature/xxx
git rebase dev
# resolve conflicts
git push --force-with-lease
```

5. create a pull request

### Notes

1. If `feature/xxx` is shared between multiple people, always use `git pull --rebase` on `feature/xxx` to get changes from others. Once the feature is developed, rebase to `dev` and DO NOT make another commit on `feature/xxx`.

2. `git push --force` only you know what you are doing.
