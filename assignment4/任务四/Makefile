# ========================================================================
# 任务四: NICE 端到端推理 Makefile
#
# 用法 (在 Ubuntu 虚拟机中):
#   make convert        -> 把任务三文本权重转成二进制 (.bin), 只需跑一次
#   make                -> 编译全部 (权重自检 + 前向推理[标量])
#   make run-load       -> 第一步: 权重读取自检
#   make run-infer      -> 第二步: 前向推理 (标量算子, x86 主机校验)
#   make nice_infer_rvv -> 第三步: 交叉编译 RVV 版推理 (RISC-V)
#   make run-infer-rvv  -> 第三步: 在 Spike 上跑 RVV 版
#
# RVV 版链接任务一的算子源文件 (位于上级目录 dachuang/)。
# ========================================================================

HOST_CC   = gcc
RISCV_CC  = /opt/riscv-elf/bin/riscv64-unknown-elf-gcc

HOST_CFLAGS  = -O2 -Wall -std=c99
RISCV_CFLAGS = -march=rv64gcv_zicsr_zifencei -O3 -Wall

# ---- 第一步: 权重读取自检 ----
LOAD_BIN  = load_weights_test
LOAD_SRCS = load_weights.c main.c

# ---- 第二步: NICE 前向推理 (标量算子, x86 主机校验) ----
INFER_BIN  = nice_infer
INFER_SRCS = load_weights.c circulant_scalar_v3.c inference.c infer_main.c

# ---- 第三步: NICE 前向推理 (RVV 算子, RISC-V + Spike) ----
INFER_RVV_BIN = nice_infer_rvv
RVVPATH       = ..
INFER_RVV_SRCS = load_weights.c inference.c infer_main.c circulant_v3.c \
                 $(RVVPATH)/irdfft_butterfly_compute.c \
                 $(RVVPATH)/irdfft_butterfly_v3.c \
                 $(RVVPATH)/irdfft_external_v3.c \
                 $(RVVPATH)/irdfft_half_complex_mul_v3.c \
                 $(RVVPATH)/rdfft_butterfly_compute.c \
                 $(RVVPATH)/rdfft_butterfly_v3.c \
                 $(RVVPATH)/external_v3.c \
                 $(RVVPATH)/twiddle_factors.c

# Spike + proxy kernel 路径 (按虚拟机实际安装位置调整)
SPIKE = /opt/riscv/bin/spike
PK    = /opt/riscv/riscv64-unknown-elf/bin/pk

.PHONY: all convert run-load run-infer run-infer-rvv clean

all: $(LOAD_BIN) $(INFER_BIN)

$(LOAD_BIN): $(LOAD_SRCS) load_weights.h
	$(HOST_CC) $(HOST_CFLAGS) -o $@ $(LOAD_SRCS) -lm

$(INFER_BIN): $(INFER_SRCS) load_weights.h circulant_op.h inference.h
	$(HOST_CC) $(HOST_CFLAGS) -o $@ $(INFER_SRCS) -lm

$(INFER_RVV_BIN): $(INFER_RVV_SRCS) load_weights.h circulant_op.h inference.h
	$(RISCV_CC) $(RISCV_CFLAGS) -o $@ $(INFER_RVV_SRCS) -lm

convert:
	python3 convert_weights.py

run-load: $(LOAD_BIN)
	./$(LOAD_BIN) circulant_weights.bin scaling_weights.bin

run-infer: $(INFER_BIN)
	./$(INFER_BIN) circulant_weights.bin scaling_weights.bin z_input.txt > x_out.txt

run-infer-rvv: $(INFER_RVV_BIN)
	$(SPIKE) --isa=rv64gcv_zicsr_zifencei $(PK) ./$(INFER_RVV_BIN) \
	  circulant_weights.bin scaling_weights.bin z_input.txt > x_out_rvv.txt

clean:
	rm -f $(LOAD_BIN) $(INFER_BIN) $(INFER_RVV_BIN)
