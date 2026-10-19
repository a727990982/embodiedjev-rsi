# Jev-as-Policy

The Cartesian motor questions and signed, endpoint-clamped control in
`libero_rsi/cartesian_policy.py` and `libero_rsi/harness/robot.py` are adapted
from https://github.com/YuanKJing/Jev-as-Policy (`jev_policy.py`, `app.py`,
main inspected 2026-10-20). The migration uses this project's visual estimates,
image transport and LIBERO controller; it does not copy the upstream scene,
simulator object-state observations, or threaded physics loop.

MIT License

Copyright (c) 2026 Junqi(Junairi) Jing

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
