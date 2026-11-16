# 原生模板匹配组件

形状匹配源码来自 meiqua/shape_based_matching，固定版本
`3df935f8e4b8c8a8ed72984211543e1b7696452e`，许可证 BSD 2-Clause。
源码原始许可证位于 vendor/shape_based_matching/LICENSE，发行时随库提供。

该源码包含 MIPP，许可证 MIT。下列通知随二进制发行保留：

Copyright (c) 2016 MIPP

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

mipp.h 按编译目标指令集条件包含内嵌的超越函数头（NEON 包含 neon_mathfun.h，
AVX512F 包含 avx512_mathfun.h，AVX 包含 avx_mathfun.h，SSE 包含
sse_mathfun.h），进入最终二进制的仅限编译目标对应的那一个头。下列三个
zlib 许可组件的通知随源码及二进制发行保留：

sse_mathfun.h —— Copyright (C) 2007 Julien Pommier

avx_mathfun.h —— Copyright (C) 2012 Giovanni Garberoglio
Interdisciplinary Laboratory for Computational Science (LISC),
Fondazione Bruno Kessler and University of Trento

neon_mathfun.h —— Copyright (C) 2011 Julien Pommier

上述三个头文件内的 zlib 许可证正文一致，原文如下（各自文件内重复出现）：

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)

avx512_mathfun.h 及其实现 avx512_mathfun.hxx 为 MIT 许可证，Copyright (C)
2017 Adrien Cassagne，随源码分发保留。

OpenCV 4.10 为构建依赖，使用 Apache 2.0 许可证。构建脚本将实际 SDK 的
LICENSE 随运行库装配。Python cv2 wheel 不代替这个 C++ SDK。
