# Third-party licenses and attribution

envX is distributed under Apache-2.0. Components incorporated into it retain
their original notices and licenses.

## OGBench Cube

The Cube task definition and minimal MuJoCo model-building utilities are
derived from OGBench v1.2.1 and the `fracapuano/cube-mjx` changes.

Copyright (c) 2024 OGBench Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Reacher MJX

Copyright (c) 2026 Francesco Capuano. Distributed under the MIT License. The
runtime model and task constants come from `dm_control.suite.reacher`, which is
an external dependency and is not redistributed here.

## PushT and Two-Room references

The pure-JAX ports retain the Apache-2.0 license and attribution notices from
their standalone repositories. PushT task constants and compatibility behavior
are based on the Apache-2.0-licensed implementations in Diffusion Policy and
Hugging Face gym-pusht; their simulator source is not included here.

The Two-Room task contracts are derived from Stable World Model (MIT), PLDM
(MIT), and EB-JEPA (Apache-2.0). Their source code is not included here. The
PLDM notice is reproduced below:

Copyright (c) 2025 Vlad Sobal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

EB-JEPA is Copyright (c) Meta Platforms, Inc. and affiliates and is
distributed under Apache-2.0. The full Apache-2.0 text is in `LICENSE`. See
`NOTICE` for the exact incorporated envX source revisions.

## UR5e and Robotiq assets

The complete BSD license texts are retained at:

- `src/envx/cube/_vendor/descriptions/universal_robots_ur5e/LICENSE`
- `src/envx/cube/_vendor/descriptions/robotiq_2f85/LICENSE`
