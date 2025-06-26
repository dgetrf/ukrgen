# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import subprocess
import os, os.path
from string import Template

from asmgen.asmblocks.avx_fma import fma128,fma256,avx512
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.sve import sve

from asmgen.asmblocks.operations import widening_method as wm

from asmgen.registers import (
    asm_data_type as adt,
    adt_triple,
    adt_size,
    reg_tracker
)

from asmgen.compilation.tools import compiler

from ukrgen.specializers.asm import lsc_specializer
from ukrgen.components import simple_ukr_tile,dimension_type,dimension_properties
from ukrgen.generators.mm import mm,order2D
from ukrgen.models.load_store_cpu import load_store_cpu
from ukrgen.schedulers import simple_dependency_scheduler


import unittest

asmgen_map = {
    'avx128' : fma128,
    'avx256' : fma256,
    'avx512' : avx512,
    'rvv' : rvv,
    'rvv071' : rvv071,
    'neon' : neon,
    'sve' : sve,
    'sme' : sme,
}

# Save the VL in Bytes for later
pre_block = ["\"ld t1, %[a]\\n\\t\"\\\n",
             "\"ld t2, %[b]\\n\\t\"\\\n",
             "\"ld t3, %[c]\\n\\t\"\\\n",
             "\"vsetvli t4, zero, e8, m1, ta, ma\\n\\t\"\\\n"]

post_block = ["\\\n",
              ": [dummy_c] \"+m\"(*(double(*)[])C)\\\n",
              ": [a] \"m\"(A), [b] \"m\"(B), [c] \"m\"(C)\\\n",
              ": \"t0\", \"t1\", \"t2\", \"t3\", \"t4\", \"f0\",\\\n",
              "  \"v0\",\"v1\",\"v2\",\"v3\",\"v4\",\"v5\",\"v6\",\"v7\",\\\n",
              "  \"v8\",\"v9\",\"v10\",\"v11\",\"v12\",\"v13\",\"v14\",\"v15\",\\\n",
              "  \"v16\",\"v17\",\"v18\",\"v19\",\"v20\",\"v21\",\"v22\",\"v23\",\\\n",
              "  \"v24\",\"v25\",\"v26\",\"v27\",\"v28\",\"v29\",\"v30\",\"v31\"\n\n"]


def adt_to_C_type_str(a_dt : adt) -> str:
    if a_dt == adt.SINGLE:
        return "float"
    elif a_dt == adt.DOUBLE:
        return "double"
    elif a_dt == adt.UINT32:
        return "uint32_t"
    elif a_dt == adt.SINT64:
        return "int64_t"

    
class test_asm_specializer(unittest.TestCase):
    # Prepare lists of parameters
    def setUp(self):
        # Choose isa here
        self.gen = asmgen_map['rvv']()
        self.op = 'fma'

        # Try all possible combinations of data types
        self.triples = [adt_triple(adt.DOUBLE, adt.DOUBLE, adt.DOUBLE),
                        adt_triple(adt.SINGLE, adt.SINGLE, adt.SINGLE)]
        
        # Leave 0 for now, but needs to be selected with respect to triples
        self.variant = 0
        
        # Try all possible combinations of m, n, k that can be mapped to 32 vec registers
        self.mkn_list = [[m, n, k] for m in range(1, 32) for n in range(1, 32) for k in range(1, 5) if m * n + m <= 32]
        self.order = order2D("mnkMNK")

        # Parameters for scheduler, dummy values for now
        self.sched_rar_distance = 0
        self.sched_raw_distance = 0
        self.sched_war_distance = 0
        self.sched_waw_distance = 0

        # Specify compiler via environment variable
        self.cxx = compiler('g++','rvv')
        cxx_exec = os.getenv("CXX_COMPILER")
        if cxx_exec is not None:
            self.cxx.executable = cxx_exec
        
        # Arguments for remote execution can be passed via environment variables
        self.remote_hostname = os.getenv("REMOTE_HOSTNAME")
        self.remote_dir = os.getenv("REMOTE_DIR")
        if self.remote_dir is None:
            if self.remote_hostname is None:
                self.remote_dir = os.path.join(os.getcwd(), "tests", "specializers")
            else:
                self.remote_dir = "~/"
            
    # Generate ukernels for a specific triple and all sizes
    def gen_ukrs(self, triple : adt_triple) -> list[list[str]]:
        asm_block_list = []
        for mnk in self.mkn_list:
            rt = reg_tracker([('greg', self.gen.max_gregs),
                              ('freg', self.gen.max_fregs),
                              ('vreg', self.gen.max_vregs),
                              ('treg', self.gen.max_tregs(adt.FP64))])

            specializer = lsc_specializer(model=None, gen=self.gen, rt=rt)

            op_support_list = [sup for sup in specializer.op_support_map[self.op] if triple == sup.triple]
            sup = op_support_list[self.variant]

            ways = adt_size(triple.c)//adt_size(triple.a)

            m = mnk[0]
            n = mnk[1]
            k = mnk[2]

            nc = n
            nb = n
            if self.op == 'fma' and sup.b_tile.dima == sup.a_tile.dima:
                sup.b_tile.dima = dimension_properties(dt=dimension_type.fixed, size=1,
                                                       sdt=dimension_type.fixed, sd_size=1)

            a_tile = simple_ukr_tile(a_size=m, b_size=k,
                                     subdims=(sup.a_tile.dima, sup.a_tile.dimb))
            b_tile = simple_ukr_tile(a_size=k, b_size=nb,
                                     subdims=(sup.b_tile.dima, sup.b_tile.dimb))
            c_tile = simple_ukr_tile(a_size=m, b_size=nc,
                                     subdims=(sup.c_tile.dima, sup.c_tile.dimb))

            genmm = mm(a=a_tile, b=b_tile, c=c_tile, lo=self.order, opstr=self.op)

            mm_ops = genmm.generate()

            a_addr_regs = 1
            b_addr_regs = 1
            c_addr_regs = 1

            addr_offset_ranges=[
                [(0,self.gen.max_load_voff) for i in range(a_addr_regs)], 
                [(0,self.gen.max_load_voff) for i in range(b_addr_regs)], 
                [(0,self.gen.max_load_voff) for i in range(c_addr_regs)], 
            ]

            is_tile_scalar = lambda t : t.dima.dt == dimension_type.fixed and \
                t.dima.size == 1 and \
                t.dimb.dt == dimension_type.fixed and \
                t.dimb.size == 1

            if is_tile_scalar(sup.a_tile):
                addr_offset_ranges[0] = [(0,self.gen.max_fload_immoff(dt=sup.triple.a)) for i in range(a_addr_regs)]
            if is_tile_scalar(sup.b_tile):
                addr_offset_ranges[1] = [(0,self.gen.max_fload_immoff(dt=sup.triple.b)) for i in range(b_addr_regs)]
            if is_tile_scalar(sup.c_tile):
                addr_offset_ranges[2] = [(0,self.gen.max_fload_immoff(dt=sup.triple.c)) for i in range(c_addr_regs)]


            # Ensure the specializer doesn't generate impossible voffsets for loads/stores of
            # C regs
            if getattr(self.gen, self.op).widening_method == wm.SPLIT_INSTRUCTIONS:
                for i in range(len(addr_offset_ranges[2])):
                    addr_offset_ranges[2][i] = (addr_offset_ranges[2][i][0],addr_offset_ranges[2][i][1]//ways)

            ac_mapper = lambda tile, idx : tile.dima.size*m*idx[1]+idx[0]
            b_mapper = lambda tile, idx : tile.dima.size*n*idx[0]+idx[1]

            a_data_regs = m
            b_data_regs = 1
            c_data_regs = m * n
            a_preload = 1
            b_preload = 1
            model = load_store_cpu(res_counts=[a_data_regs,
                                               b_data_regs,
                                               c_data_regs],
                                   res_steps=[1,1,1],
                                   addr_counts=[a_addr_regs,
                                                b_addr_regs,
                                                c_addr_regs],
                                   addr_offset_ranges=addr_offset_ranges,
                                   addr_starts=[
                                       [i*sup.a_tile.dima.size for i in range(a_addr_regs)],
                                       [i*sup.b_tile.dima.size for i in range(b_addr_regs)],
                                       [i*sup.c_tile.dima.size for i in range(c_addr_regs)]
                                   ],
                                   preload_counts=[a_preload,
                                                   b_preload,
                                                   c_data_regs],
                                   offset_mappers=[ac_mapper,b_mapper,ac_mapper],
                                   op=self.op)

            mm_ops_next = genmm.generate(add_dims=[0,0,0,0,0,k])
            #print("\n".join(map(str,inspector(mm_ops_next))))

            #import pdb; pdb.set_trace()

            preload = model.preload(mm_ops)
            mainblock = model(mm_ops)
            storeblock = model.store_modified()
            preload_mb = model.preload(mm_ops_next,
                                       zero_addrs=False,
                                       ignore_dims=[2])

            specializer.analyse(preload)
            specializer.analyse(mainblock)
            specializer.analyse(storeblock)
            specializer.analyse(preload_mb)

            preload = specializer.pre_specialize(ops=preload, triple=triple)
            mainblock = specializer.pre_specialize(ops=mainblock, triple=triple)
            storeblock = specializer.pre_specialize(ops=storeblock, triple=triple)
            preload_mb = specializer.pre_specialize(ops=preload_mb, triple=triple)

            scheduler = simple_dependency_scheduler(
                self.sched_rar_distance,
                self.sched_raw_distance,
                self.sched_war_distance,
                self.sched_waw_distance,
                debug_on=False)

            rs_preload = scheduler(preload, loop=False)
            rs_mbpl = scheduler(mainblock+preload_mb)
            # will fail with rvv
            rs_store = storeblock
            #rs_store = scheduler(storeblock, loop=False)

            initblock = specializer.code_init(triple=triple)

            asm_rs_preload = specializer.specialize(
                ops=[op.op for op in rs_preload],
                triple=triple)
            asm_rs_mbpl = specializer.specialize(
                ops=[op.op for op in rs_mbpl],
                triple=triple)
            asm_rs_store = specializer.specialize(
                ops=rs_store,
                triple=triple)

            finiblock = specializer.code_fini(triple=triple)

            asm_block = [initblock] + asm_rs_preload + asm_rs_mbpl + asm_rs_store + [finiblock]
            asm_block_list += [asm_block]
            
        return asm_block_list

        
    # Run ukernel tests for all parameters
    def test_multi(self):
        for triple in self.triples:
            # Launch subtest for each datatype triple 
            with self.subTest(triple):
                asm_block_list = self.gen_ukrs(triple)

                blockdefines = ""
                for i, (asm_block, mnk) in enumerate(zip(asm_block_list, self.mkn_list)):
                    gemm_asm = [] + pre_block
                    gemm_asm += [a.replace("\"\n","\"\\\n") for a in asm_block[:-1]]
                    # Insert after first vsetvli in asm_block
                    gemm_asm.insert(5, "\"mv t0, t4\\n\\t\"\\\n")
                    gemm_asm += post_block

                    blockdefines += (f"#define M{i} " + f"{mnk[0]}" + f"\n#define N{i} " + f"{mnk[1]}" + f"\n#define K{i} " + f"{mnk[2]}" + "\n")
                    blockdefines += (f"#define GEMMBLOCK{i} \\\n" + "".join(gemm_asm) + "\n")

                evaluations = "\n".join([f"test_gemm(GEMMBLOCK{i}, M{i}, N{i}, K{i})" for i in range(len(asm_block_list))])

                substitutions = {
                    "DTAB" : adt_to_C_type_str(triple.a),
                    "DTC" : adt_to_C_type_str(triple.c),
                    "BLOCKDEFINES" : blockdefines,
                    "EVALUATIONS" : evaluations,
                    "GETSIMDSIZE" : self.gen.c_simd_size_function
                }

                prefix = os.path.join(os.getcwd(), "tests", "specializers")
                src_path = os.path.join(prefix, "test_asm_specializer.c.in")
                source_code = ""
                with open(src_path, 'r') as f:
                    src = Template(f.read())
                    source_code = src.substitute(substitutions)

                output_filename = "test_asm_specializer.x"
                output_path = os.path.join(prefix, output_filename)

                result = self.cxx.compile_exe(source=source_code,
                                              output_filename=output_path,
                                              libs=[], cross_compile='native',
                                              extraflags=['-static'])
                if not result:
                    raise RuntimeError("compilation failed")

                if self.remote_hostname is not None:
                    cmd = ["scp", output_path, f"{self.remote_hostname}:{self.remote_dir}"]
                    print(" ".join(cmd))
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if result.returncode != 0:
                        msg = "ssh copy failed"
                        print(msg)
                        print(result.stderr.decode())
                        raise RuntimeError(msg)

                binary_path = os.path.join(self.remote_dir, output_filename)
                cmd = [f"{binary_path}"]
                if self.remote_hostname is not None:
                    cmd = ["ssh", self.remote_hostname] + cmd

                print(" ".join(cmd))
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(result.stdout.decode())
                if result.returncode != 0:
                    print(result.stderr.decode())

                self.assertEqual(result.returncode, 0, "At least one ukernel failed.")
                

def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_asm_specializer('test_multi'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

