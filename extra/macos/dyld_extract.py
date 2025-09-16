from extra.macos.support.block import blockify
from tinygrad.helpers import tqdm
import ctypes, ctypes.util

src = "/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld/dyld_shared_cache_arm64e"
# src = "/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld/dyld_shared_cache_x86_64"
dst = "/tmp/libraries"

dsc_extractor = ctypes.CDLL("/usr/lib/dsc_extractor.bundle")

dsc_extractor.dyld_shared_cache_extract_dylibs_progress.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
dsc_extractor.dyld_shared_cache_extract_dylibs_progress.restype = ctypes.c_int

bar: tqdm = tqdm(desc='Extracting', unit=' libraries')

def progress(a, b):
  bar.t = b
  bar.update(a-bar.n+1)

dsc_extractor.dyld_shared_cache_extract_dylibs_progress(src.encode('ascii'), dst.encode('ascii'), blockify(progress, None, ctypes.c_int, ctypes.c_int))

bar.update(close=True)
