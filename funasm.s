.kernel E_64_32_4n1
.args 8
.arch gfx1100

s_load_b64 s[0:1], s[0:1], 0:int
s_waitcnt 0:ushort
v_mov_b32 v0, 0xdeadc0de:uint
v_mov_b32 v1, 0xdeadc0de:uint
v_mov_b32 v2, 0:int
global_store_b64 v2, v[0:1], s[0:1]
s_mov_b32 s2, 0:int
loop:
  s_add_i32 s2, s2, 1:int
  s_cmp_lt_i32 s2, 2147483646:int
  s_cbranch_scc1 loop
s_nop 0:int
s_sendmsg 3:int
s_endpgm

.trap
v_mov_b32 v0, 0x1337cafe:uint
v_mov_b32 v1, 0xdeadbeef:uint
v_mov_b32 v2, 0:int
global_store_b64 v2, v[0:1], s[0:1]
s_endpgm
s_rfe_b64 ttmp[0:1]
