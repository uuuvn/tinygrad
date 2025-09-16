import subprocess
from extra.macos.support.mtlcompiler import MTLRequestType
from extra.macos.metal_compile.compile_direct import compile_direct
from extra.macos.metal_compile.compile_common import get_test_kernel

src = get_test_kernel(1.0)

with open("test.metal", "w+") as fd: fd.write(src)

air_lib = subprocess.check_output("xcrun metal -o - -x metal -".split(), input=src.encode())
mtlb_lib = compile_direct(src, MTLRequestType.MTLBuildLibraryFromSourceToArchive)
unk_lib = compile_direct(src, MTLRequestType.MTLBuildLibraryFromSource)

print(air_lib.index(b"BC\xC0\xDE"))
print(mtlb_lib.index(b"BC\xC0\xDE"))
print(unk_lib.index(b"BC\xC0\xDE"))

with open("test.air", "wb+") as fd: fd.write(air_lib)
with open("test.mtlb", "wb+") as fd: fd.write(mtlb_lib)
with open("test.unk", "wb+") as fd: fd.write(unk_lib)
