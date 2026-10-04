# rdFFT RISC-V Vector 编译脚本
# 用法（在 WSL 中）: make && make run
#
# Windows + VS 集成用法:
#   在 VS 中配置"生成后事件"调用 wsl make

CC      = riscv64-unknown-linux-gnu-gcc
CFLAGS  = -march=rv64gcv -O3 -Wall
LDFLAGS = -lm
TARGET  = rdfft_rvv

SRCS    = main.c external.c twiddle_factors.c rdfft_butterfly_compute.c

# RVV 模拟器选择（二选一）
# 方式 A: Spike + proxy kernel
# EMU = spike --isa=rv64gcv pk
# 方式 B: QEMU user-mode（推荐，安装更简单）
EMU     = qemu-riscv64 -cpu rv64,v=true,vlen=256
EMU_DBG = qemu-riscv64 -cpu rv64,v=true,vlen=256 -d in_asm,cpu

.PHONY: all run debug clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run: $(TARGET)
	$(EMU) ./$(TARGET)

debug: $(TARGET)
	$(EMU_DBG) ./$(TARGET) 2>&1 | head -200

clean:
	rm -f $(TARGET)
