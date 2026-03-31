# WannChi 代码架构详细理解报告

**项目**: WannChi — 基于 Wannier Hamiltonian 的响应函数与 RPA 计算程序
**作者**: Chao Cao, Siqi Wu, Chenchao Xu, Guo-Xiang Zhi (浙江大学)
**位置**: `/Users/ykxu/Projects/hrJS/wannchi`
**参考文档**: understand.md

---

## 目录

1. [模块详细子程序分析](#1-模块详细子程序分析)
   - 1.1 constants — 基础常数与类型定义
   - 1.2 para — MPI 并行封装
   - 1.3 wanndata — Wannier Hamiltonian 数据结构
   - 1.4 linalgwrap — BLAS/LAPACK 线性代数封装
   - 1.5 symmetry_module — 角动量与对称性操作
   - 1.6 simp_module — Impurity 简化与自能打包
   - 1.7 lattice — 晶格结构、k-mesh 与自能插值
   - 1.8 IntRPA — RPA FF/CC 块结构
   - 1.9 pade — Padé 求和
   - 1.10 transp_calc — 输运性质计算
   - 1.11 gp_bo — 高斯过程贝叶斯优化
   - 1.12 cma_es — CMA-ES 优化器
   - 1.13 classical_mc — 经典海森堡蒙特卡洛
   - 1.14 wannlog — 计时与日志
   - 1.15 wanneff_JS — J-S Kondo 耦合拟合工作流
2. [源文件详细子程序分析](#2-源文件详细子程序分析)
   - 2.1 input — 输入文件解析
   - 2.2 green.f90 — 格林函数计算
   - 2.3 output.f90 / output_chi.f90 — 输出工具
   - 2.4 compute_chi.f90 — 响应函数核心计算
   - 2.5 wannchi.f90 — 主程序 (bare susceptibility)
   - 2.6 wannchiRPA.f90 — 主程序 (RPA-dressed susceptibility)
   - 2.7 postchi.f90 — 响应函数后处理
   - 2.8 wannband.f90 — 谱函数计算
   - 2.9 wanneff_JS.f90 — J-S Kondo 交换耦合拟合
3. [Downfolding 架构说明](#3-downfolding-架构说明)
4. [测试文档](#4-测试文档)
5. [构建说明](#5-构建说明)
6. [问答](#6-问答)
7. [附录：Fortran 语法参考](#7-附录fortran-语法参考)

---

## 1. 模块详细子程序分析

### 1.1 constants — 基础常数与类型定义

**文件**: [`modules/constants.f90`](wannchi/modules/constants.f90) (56 行)

#### 1.1.1 常量定义

| 常量 | 类型 | 值 | 说明 |
|------|------|-----|------|
| `dp` | integer | `selected_real_kind(14, 200)` | 双精度浮点数类型参数 |
| `twopi` | real(dp) | `6.283185307179586_dp` | $2\pi$ |
| `sqrtpi` | real(dp) | `1.772453850905516_dp` | $\sqrt{\pi}$ |
| `sqrt2` | real(dp) | `1.414213562373095_dp` | $\sqrt{2}$ |
| `logpi_2` | real(dp) | `0.572364942924700_dp` | $\ln(\pi/2)$ |
| `cmplx_1` | complex(dp) | `cmplx(1,0,dp)` | 复数 1 |
| `cmplx_i` | complex(dp) | `cmplx(0,1,dp)` | 复数虚数单位 $i$ |
| `cmplx_0` | complex(dp) | `cmplx(0,0,dp)` | 复数 0 |

#### 1.1.2 I/O 单元号

| 常量 | 值 | 用途 |
|------|-----|------|
| `stdin` | 5 | 标准输入 |
| `stdout` | 6 | 标准输出 |
| `fin` | 10 | 主输入文件单元 |
| `fout` | 11 | 主输出文件单元 |
| `fout2` | 12 | 辅助输出文件单元 |
| `fin3-fin6` | 13-16 | 辅助输入文件单元 |
| `fout3-fout6` | 17-20 | 辅助输出文件单元 |
| `fdebug` | 30 | 调试输出单元 |

#### 1.1.3 数值容差

| 常量 | 值 | 用途 |
|------|-----|------|
| `eps4` | 1.0d-4 | 一般容差 |
| `eps6` | 1.0d-6 | 高精度容差 |
| `eps9` | 1.0d-9 | 超高精度容差 |
| `eps12` | 1.0d-12 | 极限精度容差 |
| `eps18` | 1.0d-18 | 极端精度容差 |
| `eps36` | 1.0d-36 | 极端精度容差 |
| `maxint` | 2147483647 | 32位整数最大值 |

---

### 1.2 para — MPI 并行封装

**文件**: [`modules/para.f90`](wannchi/modules/para.f90) (373 行)

**依赖**: `constants`, `mpi`

#### 全局变量

| 变量 | 类型 | 说明 |
|------|------|------|
| `inode` | integer | 当前进程编号 (0 为 master) |
| `nnode` | integer | 总进程数 |
| `first_idx` | integer | 当前进程负责的任务起始索引 |
| `last_idx` | integer | 当前进程负责的任务结束索引 |
| `map(nnode, 2)` | integer, allocatable | 各进程任务范围的映射表 |

#### 子程序详细说明

##### 1.2.1 `init_para(codename)`

**功能**: 初始化 MPI 环境，获取进程信息

**接口**:
```fortran
SUBROUTINE init_para(codename)
    character(*), intent(in) :: codename  ! 程序名称字符串
```

**实现方法**:
- 如果定义了 `__MPI`，调用 `mpi_init` 初始化 MPI
- 通过 `mpi_comm_rank` 获取当前进程号 (`inode`)
- 通过 `mpi_comm_size` 获取总进程数 (`nnode`)
- 分配 `map(nnode, 2)` 数组
- 在 master 进程 (inode=0) 输出程序运行信息

**算法**:
```
if (defined __MPI):
    mpi_init()
    mpi_comm_rank(MPI_COMM_WORLD, inode)
    mpi_comm_size(MPI_COMM_WORLD, nnode)
    allocate(map(nnode, 2))
    if (inode == 0) print "codename running on nnode nodes"
else:
    inode = 0
    nnode = 1
    print "codename serial"
```

---

##### 1.2.2 `finalize_para()`

**功能**: 结束 MPI 环境

**接口**:
```fortran
SUBROUTINE finalize_para()
```

**实现方法**:
- 如果定义了 `__MPI`，调用 `mpi_finalize`

**算法**:
```
if (defined __MPI):
    mpi_finalize()
```

---

##### 1.2.3 `distribute_calc(nidx)`

**功能**: 将 nidx 个任务分配给各进程，设置 first_idx/last_idx

**接口**:
```fortran
SUBROUTINE distribute_calc(nidx)
    integer, intent(in) :: nidx  ! 总任务数
```

**实现方法**:
- 计算每个进程应处理的任务数: `nidx/nnode`
- 设置 first_idx 和 last_idx 范围
- 通过 `para_merge_int` 同步 map 信息到所有进程

**算法** (伪代码):
```
block_size = nidx / nnode
first_idx = inode * block_size + 1
last_idx = (inode + 1) * block_size
map[inode+1, 1] = first_idx - 1
map[inode+1, 2] = last_idx - first_idx + 1
if (defined __MPI):
    para_merge_int(map, 2*nnode)
```

**公式**:
- 每个进程任务数: $N_{\text{proc}} = \lfloor n_{\text{idx}} / n_{\text{node}} \rfloor$
- 起始索引: $i_{\text{start}} = i_{\text{node}} \times N_{\text{proc}} + 1$
- 结束索引: $i_{\text{end}} = (i_{\text{node}} + 1) \times N_{\text{proc}}$

---

##### 1.2.4 `para_barrier()`

**功能**: MPI 同步栅栏

**接口**:
```fortran
SUBROUTINE para_barrier()
```

**实现方法**:
- 如果定义了 `__MPI`，调用 `mpi_barrier`

---

##### 1.2.5 `para_sync_int0(dat)`

**功能**: 广播单个整数 (从 node 0)

**接口**:
```fortran
SUBROUTINE para_sync_int0(dat)
    integer, intent(inout) :: dat
```

**实现方法**:
- 如果定义了 `__MPI`，调用 `mpi_bcast(dat, 1, MPI_INTEGER, 0, MPI_COMM_WORLD)`

---

##### 1.2.6 `para_sync_real0(dat)`

**功能**: 广播单个实数 (从 node 0)

**接口**:
```fortran
SUBROUTINE para_sync_real0(dat)
    real(dp), intent(inout) :: dat
```

**实现方法**:
- 如果定义了 `__MPI`，调用 `mpi_bcast(dat, 1, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD)`

---

##### 1.2.7 `para_sync_cmplx(dat, dat_size)`

**功能**: 广播复数数组 (从 node 0)

**接口**:
```fortran
SUBROUTINE para_sync_cmplx(dat, dat_size)
    complex(dp), intent(inout) :: dat(*)
    integer, intent(in) :: dat_size
```

**实现方法**:
- 如果定义了 `__MPI`，调用 `mpi_bcast(dat, dat_size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD)`

---

##### 1.2.8 `para_merge_real0(dat)`

**功能**: 全局归约求和 (实数标量)

**接口**:
```fortran
SUBROUTINE para_merge_real0(dat)
    real(dp), intent(inout) :: dat
```

**实现方法**:
- 如果定义了 `__MPI`，调用 `mpi_allreduce(MPI_IN_PLACE, dat, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD)`

**公式**:
$$x_{\text{total}} = \sum_{i=0}^{n_{\text{node}}-1} x_i$$

---

##### 1.2.9 `para_merge_cmplx(dat, dat_size)`

**功能**: 全局归约求和 (复数数组)

**接口**:
```fortran
SUBROUTINE para_merge_cmplx(dat, dat_size)
    complex(dp), intent(inout) :: dat(*)
    integer, intent(in) :: dat_size
```

**实现方法**:
- 如果定义了 `__MPI`，调用 `mpi_allreduce(MPI_IN_PLACE, dat, dat_size, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD)`

---

##### 1.2.10 `para_collect_cmplx(fulldat, dat, blk_size)`

**功能**: 从各进程收集数据到 node 0

**接口**:
```fortran
SUBROUTINE para_collect_cmplx(fulldat, dat, blk_size)
    complex(dp), intent(out) :: fulldat(*)
    complex(dp), intent(in) :: dat(*)
    integer, intent(in) :: blk_size
```

**实现方法**:
- 创建连续的 MPI 数据类型 `blk_cmplx`
- 使用 `mpi_gatherv` 收集数据到 fulldat

**算法**:
```
blk_cmplx = MPI_CONTIGUOUS(blk_size, MPI_DOUBLE_COMPLEX)
mpi_type_commit(blk_cmplx)
mpi_gatherv(dat, map[inode+1, 2], blk_cmplx, 
            fulldat, map[:, 2], map[:, 1], 
            blk_cmplx, 0, MPI_COMM_WORLD)
mpi_type_free(blk_cmplx)
```

---

##### 1.2.11 `para_distribute_cmplx(fulldat, dat, blk_size)`

**功能**: 从 node 0 分发数据到各进程

**接口**:
```fortran
SUBROUTINE para_distribute_cmplx(fulldat, dat, blk_size)
    complex(dp), intent(in) :: fulldat(*)
    complex(dp), intent(out) :: dat(*)
    integer, intent(in) :: blk_size
```

**实现方法**:
- 创建连续的 MPI 数据类型 `blk_cmplx`
- 使用 `mpi_scatterv` 分发数据

---

### 1.3 wanndata — Wannier Hamiltonian 数据结构

**文件**: [`modules/wanndata.f90`](wannchi/modules/wanndata.f90) (321 行)

**依赖**: `constants`, `para`

#### 类型定义

##### 1.3.1 TYPE wannham

```fortran
TYPE wannham
    INTEGER :: norb           ! 轨道数
    REAL(DP), ALLOCATABLE :: tau(:,:)  ! 轨道位置 (3, norb)，单位为直接格矢
    INTEGER :: nrpt           ! R-space 格点总数
    INTEGER :: r000           ! (0,0,0) 格的索引（on-site 项）
    COMPLEX(DP), ALLOCATABLE :: hr(:,:,:)  ! R-space Hamiltonian (norb, norb, nrpt)
    REAL(DP), ALLOCATABLE :: weight(:)    ! 各 R 格点的权重（用于 Fourier 变换）
    REAL(DP), ALLOCATABLE :: rvec(:,:)    ! R 格点坐标 (3, nrpt)
END TYPE
```

#### 子程序详细说明

##### 1.3.2 `read_ham(ham, seed)`

**功能**: 从 `{seed}_hr.dat` 读取 Wannier Hamiltonian (R-space)

**接口**:
```fortran
SUBROUTINE read_ham(ham, seed)
    TYPE(wannham), intent(out) :: ham
    character(*), intent(in) :: seed
```

**实现方法**:
- 在 master 进程 (inode=0) 读取文件
- 分配 `ham%hr(norb, norb, nrpt)` 等数组
- 读取权重和 R 矢量
- 读取 Hamiltonian 矩阵元
- 通过 `para_sync_*` 同步数据到所有进程

**算法**:
```
if (inode == 0):
    open trim(seed)//"_hr.dat"
    read norb, nrpt
    allocate arrays
    read weight(1:nrpt)
    for irpt=1 to nrpt:
        for iorb=1 to norb:
            for jorb=1 to norb:
                read Rvec, i, j, Re[H], Im[H]
                if (R == (0,0,0)) r000 = irpt
                hr(j,i,irpt) = complex(Re, Im)
    close file
sync hr to all nodes
sync weight to all nodes
sync rvec to all nodes
sync r000 to all nodes
```

**关键公式**:
- 判断 R=0 格子: $R_x^2 + R_y^2 + R_z^2 = 0$

---

##### 1.3.3 `read_ham_dim(ham, seed)`

**功能**: 仅读取 Hamiltonian 维度信息 (不读完整数据)

**接口**:
```fortran
SUBROUTINE read_ham_dim(ham, seed)
    TYPE(wannham), intent(out) :: ham
    character(*), intent(in) :: seed
```

**实现方法**:
- 仅读取 norb 和 nrpt
- 分配 tau 数组 (其他数组由 read_ham 分配)

---

##### 1.3.4 `wannham_shift_ef(ham, mu)`

**功能**: 将 on-site 项 $H_{ii} \leftarrow H_{ii} - \mu$ (化学势移动)

**接口**:
```fortran
SUBROUTINE wannham_shift_ef(ham, mu)
    TYPE(wannham), intent(inout) :: ham
    real(dp), intent(in) :: mu
```

**实现方法**:
- 对所有轨道 i，将 (0,0,0) 位置的 Hamiltonian 对角元减去 mu

**公式**:
$$H_{ii}^{\text{shifted}} = H_{ii} - \mu$$

---

##### 1.3.5 `calc_hk(hk, ham, kvec)`

**功能**: R-space → k-space Fourier 变换，计算 $H(\mathbf{k})$

**接口**:
```fortran
SUBROUTINE calc_hk(hk, ham, kvec)
    TYPE(wannham), intent(in) :: ham
    real(dp), dimension(3), intent(in) :: kvec  ! k 点坐标 (分数坐标)
    complex(dp), dimension(ham%norb, ham%norb), intent(out) :: hk
```

**实现方法**:
- 计算每个轨道的相位因子 $e^{i\mathbf{k}\cdot\tau_i}$
- 对每个 R 格子，计算 $e^{i\mathbf{k}\cdot\mathbf{R}}$ 因子
- 累加所有 R 格子的贡献

**算法** (伪代码):
```
hk = 0
! 计算轨道位置相位因子
do io = 1 to norb:
    ktau = sum(kvec(:) * tau(:, io)) * 2*pi
    phase(io) = exp(i * ktau)

! Fourier 变换
do ir = 1 to nrpt:
    rdotk = sum(kvec(:) * rvec(:, ir)) * 2*pi
    fact = exp(i * rdotk) / weight(ir)
    do io = 1 to norb:
        do jo = 1 to norb:
            hk(io, jo) += fact * conjg(phase(io)) * phase(jo) * hr(io, jo, ir)
```

**公式**:
$$H_{ij}(\mathbf{k}) = \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot\mathbf{R}} \frac{H_{ij}(\mathbf{R})}{\text{weight}(\mathbf{R})} e^{i\mathbf{k}\cdot(\tau_j - \tau_i)}$$

其中附加的相位因子 $e^{i\mathbf{k}\cdot(\tau_j - \tau_i)}$ 修正了轨道位置相位。

---

##### 1.3.6 `finalize_wann(ham, all)`

**功能**: 释放 Hamiltonian 数组

**接口**:
```fortran
SUBROUTINE finalize_wann(ham, all)
    TYPE(wannham), intent(inout) :: ham
    logical, intent(in) :: all
```

**实现方法**:
- 如果 allocated(hr)，deallocate
- 如果 all 为 true，同时 deallocate weight, rvec, tau

---

##### 1.3.7 `write_ham(ham, seed)`

**功能**: 将 Hamiltonian 写回文件

**接口**:
```fortran
SUBROUTINE write_ham(ham, seed)
    TYPE(wannham), intent(in) :: ham
    character(*), intent(in) :: seed
```

**实现方法**:
- 在 master 进程写入 `{seed}_hr.dat` 格式

---

### 1.4 linalgwrap — BLAS/LAPACK 线性代数封装

**文件**: [`modules/linalgwrap.f90`](wannchi/modules/linalgwrap.f90) (156 行)

**依赖**: `constants`

#### 接口定义

```fortran
interface invmat
    module procedure dinvmat, zinvmat
end interface

interface eigen
    module procedure heigen, geigen
end interface
```

#### 子程序详细说明

##### 1.4.1 `dinvmat(xmat, ndim)`

**功能**: 实矩阵求逆

**接口**:
```fortran
subroutine dinvmat(xmat, ndim)
    integer, intent(in) :: ndim
    real(dp), dimension(ndim, ndim) :: xmat  ! 输入输出矩阵
```

**实现方法**: 使用 LAPACK 的 `dgetrf` + `dgetri`

**算法**:
```
call dgetrf(ndim, ndim, xmat, ndim, ipiv, info)
call dgetri(ndim, xmat, ndim, ipiv, work, ndim, info)
```

**公式**: 通过 LU 分解求矩阵逆

---

##### 1.4.2 `zinvmat(xmat, ndim)`

**功能**: 复矩阵求逆

**接口**:
```fortran
subroutine zinvmat(xmat, ndim)
    integer, intent(in) :: ndim
    complex(dp), dimension(ndim, ndim) :: xmat  ! 输入输出矩阵
```

**实现方法**: 使用 LAPACK 的 `zgetrf` + `zgetri`

**算法**:
```
call zgetrf(ndim, ndim, xmat, ndim, ipiv, info)
call zgetri(ndim, xmat, ndim, ipiv, work, ndim, info)
```

---

##### 1.4.3 `heigen(eig, xmat, ndim)`

**功能**: Hermitian 矩阵本征问题

**接口**:
```fortran
subroutine heigen(eig, xmat, ndim)
    integer, intent(in) :: ndim
    complex(dp), dimension(ndim, ndim) :: xmat  ! 输入矩阵，输出本征向量
    real(dp), dimension(ndim), intent(out) :: eig  ! 本征值
```

**实现方法**: 使用 LAPACK 的 `zheev`

**算法**:
```
call zheev('V', 'U', ndim, xmat, ndim, eig, work, 2*ndim, rwork, info)
```

**公式**: 求解 $H\psi = \lambda\psi$，其中 $H = H^\dagger$

---

##### 1.4.4 `geigen(eig, xmat, ndim)`

**功能**: 通用复矩阵本征问题 (右本征向量)

**接口**:
```fortran
subroutine geigen(eig, xmat, ndim)
    integer, intent(in) :: ndim
    complex(dp), dimension(ndim, ndim) :: xmat  ! 输入矩阵，输出右本征向量
    complex(dp), dimension(ndim), intent(out) :: eig  ! 本征值
```

**实现方法**: 使用 LAPACK 的 `zgeev`

**算法**:
```
call zgeev('N', 'V', ndim, xmat, ndim, eig, vl, 1, vr, ndim, work, 2*ndim, rwork, info)
xmat = vr  ! 用右本征向量覆盖输入矩阵
```

---

##### 1.4.5 `sparsemulmat(zmat, xmat_cp, ymat, idxcp, ndim, nidxcp, alpha, beta)`

**功能**: 稀疏矩阵乘法 $z = \alpha \cdot x_{\text{cp}} \cdot y + \beta \cdot z$

**接口**:
```fortran
subroutine sparsemulmat(zmat, xmat_cp, ymat, idxcp, ndim, nidxcp, alpha, beta)
    integer :: ndim
    integer :: nidxcp
    integer, dimension(2, nidxcp) :: idxcp
    complex(dp), dimension(ndim, ndim) :: ymat, zmat
    real(dp), dimension(nidxcp) :: xmat_cp
    real(dp) :: alpha, beta
```

**实现方法**: 
- xmat_cp 是压缩形式的矩阵 (非零元素)
- idxcp 存储非零元素在满矩阵中的 (i,j) 索引
- 直接遍历计算

**算法** (伪代码):
```
z = beta * z
do ii = 1 to nidxcp:
    i1 = idxcp(1, ii)
    i2 = idxcp(2, ii)
    do jj = 1 to ndim:
        z(i1, jj) += alpha * xmat_cp(ii) * ymat(i2, jj)
```

**公式**:
$$Z_{i_1,j} = \beta Z_{i_1,j} + \alpha \sum_{k} X_{i_1,i_2}^{(cp)} Y_{i_2,j}$$

---

##### 1.4.6 `matmulsparse(zmat, xmat, ymat_cp, idxcp, ndim, nidxcp, alpha, beta)`

**功能**: 稀疏矩阵乘法 $z = \alpha \cdot x \cdot y_{\text{cp}} + \beta \cdot z$

**接口**:
```fortran
subroutine matmulsparse(zmat, xmat, ymat_cp, idxcp, ndim, nidxcp, alpha, beta)
    integer :: ndim
    integer :: nidxcp
    integer, dimension(2, nidxcp) :: idxcp
    complex(dp), dimension(ndim, ndim) :: xmat, zmat
    real(dp), dimension(nidxcp) :: ymat_cp
    real(dp) :: alpha, beta
```

**算法** (伪代码):
```
z = beta * z
do ii = 1 to ndim:
    do jj = 1 to nidxcp:
        j1 = idxcp(1, jj)
        j2 = idxcp(2, jj)
        z(ii, j2) += alpha * xmat(ii, j1) * ymat_cp(jj)
```

---

### 1.5 symmetry_module — 角动量与对称性操作

**文件**: [`modules/symmetry.f90`](wannchi/modules/symmetry.f90) (506 行)

**依赖**: `constants`, `linalgwrap`

#### 类型定义

```fortran
TYPE symmetry
    real(dp), dimension(3,3) :: rot    ! 直接格矢空间的旋转矩阵
    real(dp), dimension(3) :: tau       ! 平移矢量
    real(dp), dimension(3) :: axis      ! 旋转轴（笛卡尔坐标）
    real(dp) :: theta                   ! 旋转角
    logical :: inv                      ! 是否含反演
END TYPE
```

#### 子程序详细说明

##### 1.5.1 `generate_Smatrix(Sx, Sy, Sz)`

**功能**: 生成 Pauli 自旋矩阵

**接口**:
```fortran
SUBROUTINE generate_Smatrix(Sx, Sy, Sz)
    complex(dp), dimension(2, 2), intent(out) :: Sx, Sy, Sz
```

**实现方法**: 直接构造 Pauli 矩阵

**公式**:
$$S_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad 
S_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad 
S_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

---

##### 1.5.2 `generate_Lmatrix(Lx, Ly, Lz, l)`

**功能**: 生成角动量算符 $L_x, L_y, L_z$ 在 Ylm 基底的矩阵元

**接口**:
```fortran
SUBROUTINE generate_Lmatrix(Lx, Ly, Lz, l)
    integer, intent(in) :: l
    complex(dp), dimension(2*l+1, 2*l+1), intent(out) :: Lx, Ly, Lz
```

**实现方法**: 
- 先构造 $L_+$ 和 $L_-$ 矩阵
- 通过 $L_x = (L_+ + L_-)/2$, $L_y = (L_+ - L_-)/(2i)$ 计算

**算法**:
```
do m = -l to l:
    Lz(m+l+1, m+l+1) = m
    if (m < l):
        Lp(m+l+2, m+l+1) = sqrt((l-m)*(l+m+1))
        Lm(m+l+1, m+l+2) = sqrt((l+m+1)*(l-m))
Lx = (Lp + Lm) / 2
Ly = (Lp - Lm) / (2*i)
```

**公式**:
$$L_z |l,m\rangle = m |l,m\rangle$$
$$L_+ |l,m\rangle = \sqrt{(l-m)(l+m+1)} |l,m+1\rangle$$
$$L_- |l,m\rangle = \sqrt{(l+m)(l-m+1)} |l,m-1\rangle$$

---

##### 1.5.3 `generate_Ylm2C(Umat, l)`

**功能**: 生成 Ylm → Cubic Harmonics 变换矩阵

**接口**:
```fortran
SUBROUTINE generate_Ylm2C(Umat, l)
    integer, intent(in) :: l
    complex(dp), dimension(2*l+1, 2*l+1), intent(out) :: Umat
```

**实现方法**: 根据立方 Harmonics 与球谐函数的变换关系构造矩阵

**公式**: 将球谐函数 $Y_{lm}$ 线性组合成立方对称的基函数

---

##### 1.5.4 `rotate_Ylm(rot, l, symm)`

**功能**: Ylm 基底旋转 $\exp(-i\theta \hat{L}\cdot\hat{n})$

**接口**:
```fortran
SUBROUTINE rotate_Ylm(rot, l, symm)
    complex(dp), dimension(2*l+1, 2*l+1), intent(out) :: rot
    integer, intent(in) :: l
    TYPE(symmetry), intent(in) :: symm
```

**实现方法**:
- 如果旋转角很小 (theta < eps4)，返回单位矩阵
- 构造 $L \cdot \hat{n} = L_x n_x + L_y n_y + L_z n_z$
- 求本征值问题: $U^{-1} (L\cdot\hat{n}) U = \Lambda$
- 计算旋转矩阵: $R = U e^{-i\theta\Lambda} U^{-1}$

**算法**:
```
if (theta < eps4 or l == 0):
    rot = identity
else:
    U = n_x*Lx + n_y*Ly + n_z*Lz
    call eigen(eig, U, 2*l+1)
    lambda = exp(-i * theta * eig)
    rot = U * diag(lambda) * U^dagger
```

**公式**:
$$R_{Ylm} = \exp(-i\theta \hat{L} \cdot \hat{n})$$

---

##### 1.5.5 `rotate_spinor(rot, symm)`

**功能**: Spinor 旋转 $\exp(-i\theta \hat{\sigma}\cdot\hat{n}/2)$

**接口**:
```fortran
SUBROUTINE rotate_spinor(rot, symm)
    complex(dp), dimension(2, 2), intent(out) :: rot
    TYPE(symmetry), intent(in) :: symm
```

**实现方法**: 类似于 rotate_Ylm，但使用 Pauli 矩阵

**公式**:
$$R_{\text{spin}} = \exp(-i\theta \hat{\sigma} \cdot \hat{n}/2)$$

---

##### 1.5.6 `rotate_cubic(rot, l, symm)`

**功能**: Cubic Harmonics 基底旋转

**接口**:
```fortran
SUBROUTINE rotate_cubic(rot, l, symm)
    complex(dp), dimension(2*l+1, 2*l+1), intent(out) :: rot
    integer, intent(in) :: l
    TYPE(symmetry), intent(in) :: symm
```

**实现方法**: 通过 Ylm 旋转和 Ylm2C 变换组合

**公式**:
$$R_{\text{cubic}} = (Ylm2C)^{-1} \cdot R_{Ylm} \cdot Ylm2C$$

---

##### 1.5.7 `rotate_Ylms(rot, l, symm)`

**功能**: Ylm + Spin 联合基底旋转

**接口**:
```fortran
SUBROUTINE rotate_Ylms(rot, l, symm)
    integer, intent(in) :: l
    TYPE(symmetry), intent(in) :: symm
    complex(dp), dimension(4*l+2, 4*l+2), intent(out) :: rot
```

**实现方法**: 构造直积旋转矩阵

**公式**:
$$R_{Ylm\otimes S} = R_{Ylm} \otimes R_{\text{spin}}$$

---

##### 1.5.8 `init_symm_matrix(symm, r, t, avec)`

**功能**: 从旋转矩阵 R 和平移 t 初始化对称操作

**接口**:
```fortran
SUBROUTINE init_symm_matrix(symm, r, t, avec)
    type(symmetry), intent(inout) :: symm
    real(dp), dimension(3, 3), intent(in) :: r
    real(dp), dimension(3), intent(in) :: t
    real(dp), dimension(3, 3), intent(in), optional :: avec
```

**实现方法**:
- 如果提供了晶格矢量 avec，进行坐标变换
- 计算行列式判断是否含反演
- 从旋转矩阵提取旋转轴和角度

---

##### 1.5.9 `init_symm_axis_angle_inv(symm, axis, theta, inv, t, avec)`

**功能**: 从轴-角表示初始化对称操作

**接口**:
```fortran
SUBROUTINE init_symm_axis_angle_inv(symm, axis, theta, inv, t, avec)
    type(symmetry), intent(inout) :: symm
    real(dp), dimension(3), intent(in) :: axis
    real(dp), intent(in) :: theta
    logical, intent(in) :: inv
    real(dp), dimension(3), intent(in) :: t
    real(dp), dimension(3, 3), intent(in), optional :: avec
```

**实现方法**: 使用罗德里格斯公式构造旋转矩阵

**公式** (罗德里格斯公式):
$$R = I + \sin\theta \cdot K + (1-\cos\theta) \cdot K^2$$

其中 $K$ 是叉积矩阵:
$$K = \begin{pmatrix} 0 & -n_z & n_y \\ n_z & 0 & -n_x \\ -n_y & n_x & 0 \end{pmatrix}$$

---

##### 1.5.10 `inverse_symm(symm)`

**功能**: 求对称操作的逆

**接口**:
```fortran
SUBROUTINE inverse_symm(symm)
    TYPE(symmetry), intent(inout) :: symm
```

**公式**:
- $R \rightarrow R^{-1}$
- $\tau \rightarrow -R^{-1}\tau$
- $\theta \rightarrow -\theta$

---

##### 1.5.11 `generate_Ylms2JBasis(rot, l)`

**功能**: 生成 Ylm×Spin → 总角动量 J 的变换矩阵 ($j = l \pm 1/2$)

**接口**:
```fortran
SUBROUTINE generate_Ylms2JBasis(rot, l)
    complex(dp), dimension(4*l+2, 4*l+2), intent(out) :: rot
    integer, intent(in) :: l
```

**公式** (Clebsch-Gordan 系数):
- 对于 $j = l - 1/2$: $m_j = m + 1/2$
- 对于 $j = l + 1/2$: $m_j = m - 1/2$

---

### 1.6 simp_module — Impurity 简化与自能打包

**文件**: [`modules/simp.f90`](wannchi/modules/simp.f90) (144 行)

**依赖**: `constants`

#### 类型定义

```fortran
TYPE simp
    integer :: ndim              ! 该 impurity 的维度
    integer :: lang              ! 角动量量子数 l
    integer, dimension(:), allocatable :: gidx  ! 全局轨道索引
    complex(dp), dimension(:,:), allocatable :: Utrans  ! DMFT基底→Cubic Harmonics 变换矩阵
    integer, dimension(:,:), allocatable :: sigidx  ! 自能映射: Sigma(i,j) = sigpack(sigidx(i,j))
END TYPE
```

#### 子程序详细说明

##### 1.6.1 `init_simp(imp, ndim, l)`

**功能**: 初始化 impurity 结构

**接口**:
```fortran
subroutine init_simp(imp, ndim, l)
    TYPE(simp), intent(out) :: imp
    integer, intent(in) :: ndim, l
```

**实现方法**:
- 设置 ndim 和 lang
- 分配 gidx(ndim), Utrans(ndim, ndim), sigidx(ndim, ndim)

---

##### 1.6.2 `finalize_simp(imp)`

**功能**: 释放 impurity 结构

**接口**:
```fortran
subroutine finalize_simp(imp)
    TYPE(simp), intent(inout) :: imp
```

**实现方法**: deallocate 所有分配数组

---

##### 1.6.3 `matrix2pack(sigpack, sigmat, imp)`

**功能**: 将满空间自能矩阵压缩到 impurity 打包形式

**接口**:
```fortran
subroutine matrix2pack(sigpack, sigmat, imp)
    TYPE(simp), intent(in) :: imp
    complex(dp), dimension(imp%ndim, imp%ndim), intent(in) :: sigmat
    complex(dp), dimension(:), intent(out) :: sigpack
```

**实现方法**:
- 先进行基变换: $s_{\text{tmp1}} = U_{\text{trans}}^\dagger \cdot \Sigma \cdot U_{\text{trans}}$
- 然后根据 sigidx 映射填充 sigpack

**算法**:
```
! 基变换
call zgemm('N', 'C', ndim, ndim, ndim, 1, sigmat, ndim, Utrans, ndim, 0, stmp1, ndim)
call zgemm('N', 'N', ndim, ndim, ndim, 1, Utrans, ndim, stmp1, ndim, 0, stmp2, ndim)

! 填充打包形式
do ii = 1 to ndim:
    do jj = 1 to ndim:
        if (imp%sigidx(ii, jj) > 0):
            sigpack(imp%sigidx(ii, jj)) += sigmat(ii, jj)
```

**公式**:
$$\Sigma_{\text{packed}}(k) = \sum_{ij} \Sigma_{ij} \cdot U_{ik} \cdot U_{jk}^*$$

---

##### 1.6.4 `restore_matrix(sigmat, sigpack, imp)`

**功能**: 从打包形式恢复满空间自能矩阵

**接口**:
```fortran
subroutine restore_matrix(sigmat, sigpack, imp)
    TYPE(simp), intent(in) :: imp
    complex(dp), dimension(imp%ndim, imp%ndim), intent(out) :: sigmat
    complex(dp), dimension(:), intent(in) :: sigpack
```

**实现方法**: 与 matrix2pack 相反的过程

**算法**:
```
! 从打包形式恢复
sigmat = 0
do ii = 1 to ndim:
    do jj = 1 to ndim:
        if (imp%sigidx(ii, jj) > 0):
            sigmat(ii, jj) = sigpack(imp%sigidx(ii, jj))

! 基变换
call zgemm('N', 'N', ndim, ndim, ndim, 1, sigmat, ndim, Utrans, ndim, 0, stmp, ndim)
call zgemm('C', 'N', ndim, ndim, ndim, 1, Utrans, ndim, stmp, ndim, 0, sigmat, ndim)
```

---

### 1.7 lattice — 晶格结构、k-mesh 与自能插值

**文件**: [`modules/lattice.f90`](wannchi/modules/lattice.f90) (1073 行)

**依赖**: `constants`, `wanndata`, `simp_module`, `para`, `linalgwrap`, `symmetry_module`

#### 全局变量

```fortran
real(dp), dimension(3,3) :: avec   ! 晶格矢量 (a1=avec(:,1), a2=avec(:,2), a3=avec(:,3))
real(dp), dimension(3,3) :: bvec   ! 倒格子矢量 (b1=bvec(1,:), b2=bvec(2,:), b3=bvec(3,:))
logical :: spinor                  ! 是否为自旋极化（含 SOC）基底
integer :: nsite                   ! 原子位置数
integer, allocatable :: zat(:)     ! 原子序数 (nsite)
real(dp), allocatable :: xat(:,:)  ! 原子分数坐标 (3, nsite)
integer, allocatable :: nbasis(:)  ! 各原子位置上的 Wannier 轨道数 (nsite)
real(dp) :: nelec                  ! 价电子总数

TYPE(wannham) :: ham              ! Wannier Hamiltonian
integer :: nimp                   ! Impurity 个数
TYPE(simp), allocatable :: imp(:) ! 各 impurity 的定义
integer :: nbath                  ! 自能打包形式的 bath 维度（= ndimf，所有 impurity 轨道总数）
real(dp) :: beta                   ! 逆温度（由 input 模块传入 lattice）
integer :: nw                      ! 自能频率点数量

integer :: ndimf, ndimc           ! F/C 子空间维度
integer, allocatable :: f2g_idx(:), c2g_idx(:)  ! F/C → 全局索引
integer, allocatable :: g2f_idx(:), g2c_idx(:)  ! 全局 → F/C 索引
integer, allocatable :: partition(:)  ! 轨道所属 partition（0=bath/C, >0=impurity编号）

integer :: nk1, nk2, nk3          ! BZ 网格密度
integer :: nkirr                   ! 不可约 k 点数
real(dp), allocatable :: kvec(:,:) ! k 点坐标 (3, nkirr)，分数坐标
real(dp), allocatable :: kwt(:)   ! k 点权重

complex(dp), allocatable :: omega(:) ! 频率网格
complex(dp), allocatable :: sinf(:)  ! Σ(∞)，高频自能极限
complex(dp), allocatable :: sigpack(:,:)  ! Σ(ω) 打包形式 (nbath, nw)
```

#### 子程序详细说明

##### 1.7.1 `read_posfile(seed)`

**功能**: 读取 `.pos` 文件（类 POSCAR 格式）

**接口**:
```fortran
SUBROUTINE read_posfile(seed)
    character(*), intent(in) :: seed
```

**文件格式**:
```
Kagome                  ! 注释行
1.0                     ! 缩放因子
4.6669313530311989   -2.6944540729627522    0.0    ! a1
4.6669313530311989    2.6944540729627526    0.0    ! a2
0.0    0.0    9.8872122322434954           ! a3
3  0                      ! nsite, soc (0/1)
1  0.5  0.0  0.0  1      ! Zat, x, y, z, nbasis
1  0.0  0.5  0.0  1
1  0.5  0.5  0.0  1
```

**实现方法**:
- 解析晶格矢量
- 解析原子位置和轨道数
- 计算倒格子矢量

**公式**:
$$b_i \cdot a_j = 2\pi \delta_{ij}$$

---

##### 1.7.2 `read_kmesh(filename)`

**功能**: 读取 IBZKPT 文件

**接口**:
```fortran
SUBROUTINE read_kmesh(filename)
    character(*), intent(in) :: filename
```

**文件格式**:
```
Automatically generated mesh
0           ! switch (0 = 自动 gamma 网格)
Reciprocal lattice
nk1 nk2 nk3  ! k-mesh 密度
```

**实现方法**:
- 读取网格密度 nk1, nk2, nk3
- 生成以 Gamma 为中心的均匀网格
- 通过对称性找到不可约 k 点

---

##### 1.7.3 `read_impfile(seed)`

**功能**: 读取 impurity 定义文件

**接口**:
```fortran
SUBROUTINE read_impfile(seed)
    character(*), intent(in) :: seed
```

**文件格式** (参见 understand.md §8.7):
```
IMP-1
5  2                    ! ndim, l
BasisMap
1  2  3  4  5           ! gidx
Sigidx
1  2  3  4  5
2  6  7  8  9
...
LocRot
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
```

---

##### 1.7.4 `setup_mapping()`

**功能**: 构建 F/C 分区映射

**接口**:
```fortran
SUBROUTINE setup_mapping()
```

**实现方法**:
- 根据 imp(ii)%gidx 设置 partition
- 计算 ndimf 和 ndimc
- 构建 f2g_idx, c2g_idx, g2f_idx, g2c_idx

**算法**:
```
partition = 0
do ii = 1 to nimp:
    partition(imp(ii)%gidx(:)) = ii
    
ndimf = sum(imp(ii)%ndim)
ndimc = norb - ndimf

allocate(f2g_idx(ndimf), c2g_idx(ndimc), g2f_idx(norb), g2c_idx(norb))

jj = 1
kk = 1
do ii = 1 to norb:
    if (partition(ii) > 0):
        f2g_idx(jj) = ii
        g2f_idx(ii) = jj
        jj = jj + 1
    else:
        c2g_idx(kk) = ii
        g2c_idx(ii) = kk
        kk = kk + 1
```

---

##### 1.7.5 `fix_sigma_static()`

**功能**: 将 Σ(∞) 加到 on-site Hamiltonian

**接口**:
```fortran
SUBROUTINE fix_sigma_static()
```

**实现方法**:
- 对于每个 k 点，将 sinf 加到 Hamiltonian 对角元

---

##### 1.7.6 `get_sigma_matrix(sigfull, z)`

**功能**: 在复频率 z 处获取完整自能矩阵

**接口**:
```fortran
SUBROUTINE get_sigma_matrix(sigfull, z)
    complex(dp), dimension(norb, norb), intent(out) :: sigfull
    complex(dp), intent(in) :: z
```

**实现方法**:
- 调用 interpolate_single_sigma 获取打包形式的自能
- 通过 restore_matrix 恢复满空间形式

---

##### 1.7.7 `interpolate_single_sigma(sigval, w)`

**功能**: 自能插值（单频率点）

**接口**:
```fortran
SUBROUTINE interpolate_single_sigma(sigval, w)
    complex(dp), intent(out) :: sigval
    complex(dp), intent(in) :: w
```

**实现方法**:
- 如果 w 在网格内，直接返回值
- 如果 w 在网格间，线性插值
- 如果 w 在高频尾区，使用渐近展开

**公式**:
- **Matsubara 频率**: $\omega_n = i\frac{2\pi(n-1)}{\beta}$
- **高频尾**: $\Sigma(\omega) \approx \Sigma_\infty + \frac{A}{\omega} + \frac{B}{\omega^2}$

---

### 1.8 IntRPA — RPA FF/CC 块结构

**文件**: [`modules/intRPA.f90`](wannchi/modules/intRPA.f90) (242 行)

**依赖**: `constants`, `lattice`, `para`, `linalgwrap`

#### 全局变量

```fortran
integer :: nFFidx     ! FF block 总维度 = Σ blkdim²
integer :: nCCidx     ! CC block 维度 = norb - Σ blkdim
integer, dimension(2, nFFidx) :: FFidx  ! FF 轨道对 (i1=FFidx(1,:), i2=FFidx(2,:))
integer, dimension(nCCidx) :: CCidx     ! CC 全局轨道索引
integer :: nUcp       ! U 矩阵非零元个数
real(dp), dimension(nUcp) :: Uint_cp    ! U 矩阵元（压缩形式）
integer, dimension(2, nUcp) :: idxUcp  ! 非零元在满矩阵中的 (i,j) 索引
```

#### 子程序详细说明

##### 1.8.1 `read_RPA()`

**功能**: 读取 RPA.inp，构建 FFidx、CCidx、Uint_cp

**接口**:
```fortran
SUBROUTINE read_RPA()
```

**实现方法**:
1. 读取 nffblk 和 blkdim
2. 计算 nFFidx 和 nCCidx
3. 构建 FFidx (所有 i-j 对)
4. 构建 CCidx (排除 FF 块轨道)
5. 读取 U 矩阵元

**算法**:
```
read nffblk
read blkdim(1:nffblk)

nFFidx = sum(blkdim(i)^2)
nCCidx = norb - sum(blkdim(i))

mapping = 0
do ii = 1 to nffblk:
    read blkidx(1:blkdim(ii))
    mapping(blkidx) = ii
    
    ! 构建 FFidx
    jj = 1
    do j1 = 1 to blkdim(ii):
        do j2 = 1 to blkdim(ii):
            FFidx(1, jj) = blkidx(j1)
            FFidx(2, jj) = blkidx(j2)
            jj = jj + 1

! 构建 CCidx
jj = 1
do ii = 1 to norb:
    if (mapping(ii) == 0):
        CCidx(jj) = ii
        jj = jj + 1

! 读取 U 矩阵
read nUcp
do ii = 1 to nUcp:
    read i1, i2, j1, j2, Uij
    idxUcp(1, ii) = i1
    idxUcp(2, ii) = i2
    Uint_cp(ii) = Uij
```

---

##### 1.8.2 `find_ffidx(ii, i1, i2)`

**功能**: 给定 (i1, i2) 轨道对，在 FFidx 中找索引

**接口**:
```fortran
SUBROUTINE find_ffidx(ii, i1, i2)
    integer, intent(out) :: ii
    integer, intent(in) :: i1, i2
```

**算法**: 线性搜索 FFidx 数组

---

##### 1.8.3 `calc_chiRPA(chiff, chicc, chifc, chicf, chi0ff, chi0cc, chi0fc, chi0cf, ff_only, nw)`

**功能**: 核心 RPA 方程求解

**接口**:
```fortran
SUBROUTINE calc_chiRPA(chiff, chicc, chifc, chicf, chi0ff, chi0cc, chi0fc, chi0cf, ff_only, nw)
    complex(dp), dimension(nFFidx, nFFidx, nw), intent(out) :: chiff
    complex(dp), dimension(nCCidx, nCCidx, nw), intent(out) :: chicc
    complex(dp), dimension(nFFidx, nCCidx, nw), intent(out) :: chifc
    complex(dp), dimension(nCCidx, nFFidx, nw), intent(out) :: chicf
    complex(dp), dimension(nFFidx, nFFidx, nw), intent(in) :: chi0ff
    complex(dp), dimension(nCCidx, nCCidx, nw), intent(in) :: chi0cc
    complex(dp), dimension(nFFidx, nCCidx, nw), intent(in) :: chi0fc
    complex(dp), dimension(nCCidx, nFFidx, nw), intent(in) :: chi0cf
    logical, intent(in) :: ff_only
    integer, intent(in) :: nw
```

**实现方法**:
1. 计算 Dyson 因子 $D_{FF} = (1 - \chi^0_{FF} \cdot U_{FF})^{-1}$
2. 计算 $V_{FF} = U_{FF} \cdot D_{FF}$
3. 计算 RPA 响应函数块

**算法** (伪代码):
```
! 计算 Dff = (1 - chi0ff * Uff)^(-1)
call matmulsparse(Dff, chi0ff, Uint_cp, idxUcp, nFFidx, nUcp, -1.d0, 1.d0)
call invmat(Dff, nFFidx)

! 计算 Vff = Uff * Dff
call sparsemulmat(Vff, Uint_cp, Dff, idxUcp, nFFidx, nUcp, 1.d0, 0.d0)

! 计算响应函数块
! chiFF = DFF * chi0FF
call zgemm('N', 'N', nFFidx, nFFidx, nFFidx, 1, Dff, nFFidx, chi0ff, nFFidx, 0, chiff, nFFidx)

! 如果不是 ff_only:
if (.not. ff_only):
    ! tmpCF = chi0CF * VFF
    call zgemm('N', 'N', nCCidx, nFFidx, nFFidx, 1, chi0cf, nCCidx, Vff, nFFidx, 0, tmpcf, nCCidx)
    
    ! chiFC = DFF * chi0FC
    call zgemm('N', 'N', nFFidx, nCCidx, nFFidx, 1, Dff, nFFidx, chi0fc, nFFidx, 0, chifc, nFFidx)
    
    ! chiCF = chi0CF + tmpCF * chi0FF
    call zgemm('N', 'N', nCCidx, nFFidx, nFFidx, 1, tmpcf, nCCidx, chi0ff, nFFidx, 1, chicf, nCCidx)
    
    ! chiCC = chi0CC + tmpCF * chi0FC
    call zgemm('N', 'N', nCCidx, nCCidx, nFFidx, 1, tmpcf, nCCidx, chi0fc, nFFidx, 1, chicc, nCCidx)
```

**公式**:
$$\chi = \chi^0 + \chi^0 U \chi$$

即:
$$\chi = (1 - \chi^0 U)^{-1} \chi^0$$

写成块矩阵形式:
$$\begin{pmatrix} \chi_{FF} & \chi_{FC} \\ \chi_{CF} & \chi_{CC} \end{pmatrix} = \begin{pmatrix} D_{FF} & 0 \\ -\chi_{CF}^0 U_{FC} D_{FF} & 1 \end{pmatrix} \begin{pmatrix} \chi_{FF}^0 & \chi_{FC}^0 \\ \chi_{CF}^0 & \chi_{CC}^0 \end{pmatrix}$$

其中 $D_{FF} = (1 - \chi_{FF}^0 U_{FF})^{-1}$

---

##### 1.8.4 `finalize_RPA()`

**功能**: 释放 RPA 数组

**接口**:
```fortran
SUBROUTINE finalize_RPA()
```

---

### 1.9 pade — Padé 求和

**文件**: [`modules/pade.f90`](wannchi/modules/pade.f90) (112 行)

**依赖**: `constants`

#### 全局变量

```fortran
integer :: npole      ! 极点个数
real(dp), allocatable :: zp(:)   ! 极点位置
real(dp), allocatable :: eta(:)  ! 极点强度
```

#### 子程序详细说明

##### 1.9.1 `init_pade(np_, ispade)`

**功能**: 初始化 Padé 极点

**接口**:
```fortran
SUBROUTINE init_pade(np_, ispade)
    integer, intent(in) :: np_
    logical, intent(in) :: ispade
```

**实现方法**:
- 如果 ispade=true，使用优化的正交极点
- 如果 ispade=false，使用等间距 Matsubara 频率

**优化极点算法** (ispade=true):
- 构造 $2N_{\text{pole}} \times 2N_{\text{pole}}$ Jacobi 型三对角矩阵
- $B_{ii+1} = -0.5$, $D_{ii} = \sqrt{2i-1}$
- 通过矩阵乘积 $D \cdot B^{-1} \cdot D$ 的本征值问题生成极点

**等间距模式** (ispade=false):
- 极点取 $(n-0.5) \cdot 2\pi$
- 权重为 1

**公式**:
$$\omega_n = i\frac{2\pi(n-1)}{\beta}$$

---

##### 1.9.2 `print_pade()`

**功能**: 打印极点信息

**接口**:
```fortran
SUBROUTINE print_pade()
```

---

##### 1.9.3 `finalize_pade()`

**功能**: 释放数组

**接口**:
```fortran
SUBROUTINE finalize_pade()
```

---

### 1.10 transp_calc — 输运性质计算

**文件**: [`modules/transp_calc.f90`](wannchi/modules/transp_calc.f90) (349 行)

**依赖**: `constants`, `wanndata`, `linalgwrap`

**功能**: 从 Wannier Hamiltonian 计算输运性质：
- 反常霍尔电导率 (AHC, σ_xy) 通过贝里曲率
- 纵向直流电导率 (σ_xx) 通过 Kubo-Greenwood 公式
- 速度矩阵（解析地从 HR 导出，无数值 k-导数）

#### 子程序详细说明

##### 1.10.1 `calc_velocity(v_alpha, ham, kvec, alpha)`

**功能**: 解析计算速度矩阵 $v_\alpha(\mathbf{k})$

**接口**:
```fortran
SUBROUTINE calc_velocity(v_alpha, ham, kvec, alpha)
  TYPE(wannham), intent(in) :: ham
  real(dp), dimension(3), intent(in) :: kvec
  integer, intent(in) :: alpha   ! 1=x, 2=y, 3=z
  complex(dp), dimension(ham%norb, ham%norb), intent(out) :: v_alpha
```

**公式**:
$$v_\alpha(\mathbf{k}) = \sum_{\mathbf{R}} i \cdot 2\pi \cdot \tilde{R}_\alpha \cdot \frac{e^{i\mathbf{k}\cdot\mathbf{R}}}{w(\mathbf{R})} \cdot e^{i\mathbf{k}\cdot(\tau_j - \tau_i)} \cdot H_{ij}(\mathbf{R})$$

其中 $\tilde{R}_\alpha = R_\alpha + \tau_{j,\alpha} - \tau_{i,\alpha}$

**算法**:
1. 计算轨道相位因子: $\text{phase}(io) = \exp(i \cdot 2\pi \cdot \mathbf{k} \cdot \tau_{io})$
2. 对每个 R 格子计算: $\text{fact} = \exp(i \cdot 2\pi \cdot \mathbf{k} \cdot \mathbf{R}) / w(\mathbf{R})$
3. 对每个轨道对 $(io, jo)$: $v_\alpha(io,jo) += i \cdot 2\pi \cdot \tilde{R}_\alpha \cdot \text{fact} \cdot \text{conj}(\text{phase}(io)) \cdot \text{phase}(jo) \cdot H_{ij}(\mathbf{R})$

---

##### 1.10.2 `calc_berry_curvature(omega_n, eigvec, vx, vy, eig, norb)`

**功能**: 从本征态和速度矩阵计算贝里曲率

**公式**:
$$\Omega_n^{xy}(\mathbf{k}) = -2 \Im \sum_{m \neq n} \frac{Vx_{nm} \cdot Vy_{mn}}{(E_n - E_m)^2}$$

其中 $Vx_{nm} = \langle n|v_x|m \rangle$ 在本征基底下计算。

**算法**:
1. 通过 ZGEMM 变换到本征基底: $vx\_band = eigvec^H \cdot vx\_orb \cdot eigvec$
2. 对每对能带 $(n, m)$: 计算贡献，跳过简并情况 $|E_n - E_m| < \epsilon$

---

##### 1.10.3 `fermi_func(f, eig, mu, temperature, norb)`

**功能**: 计算费米-狄拉克分布

**公式**:
$$f(E) = \frac{1}{\exp((E - \mu) / T) + 1}$$

**边界处理**: 当 $T < 10^{-6}$ 时使用阶跃函数；溢出时使用极限值。

---

##### 1.10.4 `calc_sigma_xy(sigma_xy, ham, kvec_all, kwt_all, nk, mu_chem, temperature)`

**功能**: 计算反常霍尔电导率 σ_xy

**公式**:
$$\sigma_{xy} = -\frac{e^2}{h} \cdot \frac{1}{N_k} \sum_{\mathbf{k}} \sum_n f_n(\mathbf{k}) \Omega_n^{xy}(\mathbf{k})$$

**算法**:
1. 对每个 k 点: 计算 $H(\mathbf{k})$ → 对角化 → 计算速度 → 计算贝里曲率 → 累加
2. 对所有 k 点求和，使用归一化权重

**单位**: 对于 2D 系统 (nk3=1)，结果以 $e^2/h$ 为单位。

---

##### 1.10.5 `calc_sigma_xx(sigma_xx, ham, kvec_all, kwt_all, nk, mu_chem, temperature, broadening)`

**功能**: 通过 Kubo-Greenwood 公式计算纵向直流电导率

**公式**:
$$\sigma_{xx} = \frac{1}{N_k} \sum_{\mathbf{k}} \text{Tr}[v_x \cdot G(\mathbf{k}, \mu+i\eta) \cdot v_x \cdot G(\mathbf{k}, \mu+i\eta)]$$

其中 $G(\mathbf{k}, z) = (z - H(\mathbf{k}))^{-1}$ 是推迟格林函数。

**算法**:
1. 计算复频率 $w = \mu + i \cdot \eta$
2. 对每个 k 点: $G = (w \cdot I - H(\mathbf{k}))^{-1}$ → 计算 $v_x$ → 计算 Kubo 泡
3. $\sigma_{xx} = -\text{Im}(\sum_k \text{Tr}[v_x G v_x G]) / (2\pi)^2$

**物理意义**: $\eta = \hbar/(2\tau)$ 对应散射时间。Kubo 公式自然包含 Drude（带内）和带间贡献。

---

### 1.11 gp_bo — 高斯过程贝叶斯优化

**文件**: [`modules/gp_bo.f90`](wannchi/modules/gp_bo.f90) (704 行)

**依赖**: `constants`, `linalgwrap`

**功能**: 使用高斯过程和期望改进 (EI) 获取函数的贝叶斯优化

**适用性**:
- 低至中维问题 (n_params ≤ 20-30)
- 需要样本效率的场景
- 光滑连续目标函数

**不适用于**:
- 高维问题 (n_params > 30) — 维度灾难
- 非光滑、不连续目标函数

#### 类型定义

```fortran
TYPE gp_model
  integer :: n_train           ! 观测数量
  integer :: n_params          ! 参数空间维度
  real(dp), allocatable :: x_train(:,:)  ! (n_params, n_train)
  real(dp), allocatable :: y_train(:)    ! (n_train) 目标值
  real(dp) :: length_scale     ! RBF 核长度尺度 l
  real(dp) :: signal_var       ! σ_f² 信号方差
  real(dp) :: noise_var        ! σ_n² 噪声/jitter
  real(dp), allocatable :: K_inv(:,:)  ! (K+σ_n²I)^{-1}
  real(dp), allocatable :: alpha(:)    ! K^{-1} * y_train
END TYPE
```

#### 子程序详细说明

##### 1.11.1 `gp_init(gp, n_params, ls, sv, nv)`

**功能**: 初始化 GP 模型

**接口**:
```fortran
SUBROUTINE gp_init(gp, n_params, ls, sv, nv)
  TYPE(gp_model), intent(out) :: gp
  integer, intent(in) :: n_params
  real(dp), intent(in) :: ls, sv, nv
```

**实现**: 简单赋值，不分配 x_train 等（由 gp_update 懒分配）。

---

##### 1.11.2 `rbf_kernel(x1, x2, n, ls, sv)`

**功能**: 平方指数（RBF）核函数

**公式**:
$$k(\mathbf{x}_1, \mathbf{x}_2) = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x}_1 - \mathbf{x}_2\|^2}{2l^2}\right)$$

---

##### 1.11.3 `gp_update(gp, x_new, y_new)`

**功能**: 添加新观测到 GP 模型

**算法**:
1. 重新分配 x_train, y_train 扩展到 n+1
2. 构建新的 (n+1)×(n+1) 协方差矩阵 K
3. 添加对角 jitter: $K_{ii} += \sigma_n^2$
4. 计算 $K^{-1}$ 通过 invmat（LU 分解）
5. 计算 $\alpha = K^{-1} \cdot \mathbf{y}$

**NaN 检查**: invmat 后检查 NaN，发现则 revert n_train 保持旧状态。

---

##### 1.11.4 `gp_predict(gp, x_test, mu_out, sigma_out)`

**功能**: 在未测点预测均值和方差

**公式**:
$$\mu(\mathbf{x}^*) = \mathbf{k}^* \cdot \boldsymbol{\alpha}$$
$$\sigma^2(\mathbf{x}^*) = k(\mathbf{x}^*,\mathbf{x}^*) - \mathbf{k}^* \cdot K^{-1} \cdot \mathbf{k}^*$$

---

##### 1.11.5 `expected_improvement(mu, sigma, y_best)`

**功能**: 期望改进获取函数

**公式**:
$$EI = (y_{best} - \mu) \cdot \Phi(z) + \sigma \cdot \phi(z)$$
其中 $z = (y_{best} - \mu) / \sigma$

**边界处理**: NaN 输入返回 0；σ < 10⁻¹⁰ 返回 max(y_best - μ, 0)；|z| > 50 使用极限近似。

---

##### 1.11.6 `latin_hypercube(samples, n_samples, n_params, bounds)`

**功能**: 生成拉丁超立方样本用于初始 GP 训练

**算法**: 每个维度独立 shuffle，确保每行每列恰好一个样本。

---

##### 1.11.7 `gp_optimize_ls(gp, bounds, n_params)`

**功能**: 通过黄金分割搜索优化长度尺度

**方法**: 在对数边际似然上执行 15 步黄金分割搜索。

---

##### 1.11.8 `bayesian_optimize(objective_func, bounds, n_params, result, n_iter)`

**功能**: 主贝叶斯优化入口

**算法**:
1. **阶段1**: Latin Hypercube 初始化 (n_init = max(5, n_params))
2. **阶段2**: 迭代 (n_iter 次):
   - 采样 n_cand = max(2000, 50·n_params) 随机候选点
   - 计算每候选点的 EI
   - 选择 top-K (K=5) 通过梯度上升 refinement EI
   - 评估目标函数，更新 GP
   - 每 10 步（n_params > 10）重新优化长度尺度

---

### 1.12 cma_es — CMA-ES 优化器

**文件**: [`modules/cma_es.f90`](wannchi/modules/cma_es.f90) (233 行)

**功能**: 协方差矩阵自适应进化策略

**适用性**:
- 高维问题 (n_params ≥ 20)
- 非凸、多峰目标函数
- 黑盒优化（无梯度）

**不适用于**:
- 低维问题 (n_params < 10)
- 离散/分类参数

#### 子程序详细说明

##### 1.12.1 `cmaes_optimize(objective_func, bounds, n_params, result, n_iter)`

**功能**: CMA-ES 优化器入口

**算法**:

**初始化**:
- λ（种群大小）: n_params ≤ 20 → max(20, 4+3·log(n_params))；n_params > 20 → max(50, n_params/2)
- μ = λ/2（父代数量）
- 均值: bounds 中心
- σ: 参数范围的 30%

**CMA-ES 参数**:
```fortran
cc   = 4.0 / (n_params + 4.0)
c1   = 2.0 / ((n_params + 1.3)^2 + mu)
cmu  = min(1 - c1, 2*(mu-2+1/mu) / ((n_params+2)^2 + mu))
damps = 1 + 2*max(0, sqrt(mu-1) - 1) + cc
chiN = sqrt(n_params) * (1 - 1/(4n_params) + 1/(21n_params²))
```

**主循环**:
1. 采样: $x_k = \mathbf{m} + \sigma \cdot N(0, I)$
2. 评估: 计算所有个体适应度
3. 排序: 取 top μ 个
4. 更新均值、进化路径 pc, ps
5. 更新协方差矩阵
6. 更新步长: $\sigma = \sigma \cdot \exp((||p_s|| - \chi_N) / (\sqrt{n} \cdot d_{amp}))$
7. 检查收敛: f_best < 10⁻⁸ 则退出

---

### 1.13 classical_mc — 经典海森堡蒙特卡洛

**文件**: [`modules/classical_mc.f90`](wannchi/modules/classical_mc.f90) (438 行)

**依赖**: `constants`

**功能**: f-site 子格子上经典海森堡自旋模拟，使用 Metropolis 算法

#### 类型定义

```fortran
TYPE mc_lattice
  integer :: n_sites           ! 超胞中自旋总数
  integer :: n_neighbors_max  ! 每位点最大邻居数（默认12）
  integer, allocatable :: neighbor_list(:,:)  ! (n_neighbors_max, n_sites)
  integer, allocatable :: n_nn(:)             ! 每位点实际邻居数
  real(dp), allocatable :: spin(:,:)          ! (3, n_sites) 单位自旋矢量
  real(dp) :: J_mc            ! 交换耦合 (>0: 铁磁)
  real(dp) :: S_mag           ! 自旋大小 |S|
END TYPE
```

#### 子程序详细说明

##### 1.13.1 `mc_init(mc, n_sites, J_mc_in, S_mag_in)`

**功能**: 初始化自旋晶格，随机化自旋方向

---

##### 1.13.2 `mc_random_spin(spin)`

**功能**: 通过 Marsaglia 方法生成单位球面上均匀随机单位矢量

**算法**:
1. 选取均匀分布的 (u,v) 满足 s = u² + v² < 1
2. 计算: $\mathbf{s} = (2u\sqrt{1-s}, 2v\sqrt{1-s}, 1-2s)$

---

##### 1.13.3 `mc_build_neighbors(mc, frac_pos, n_uc_sites, avec, nx, ny, nz, cutoff)`

**功能**: 为超胞构建周期性邻居列表

**算法**:
1. 位点索引: $idx = ((ic \cdot ny + ib) \cdot nx + ia) \cdot n_{uc} + isite$
2. 对每个位点，搜索 ±2 壳层内的邻居
3. PBC 包装: $mod(ic+dc+2nz, nz)$

---

##### 1.13.4 `mc_sweep(mc, temperature, n_accepted)`

**功能**: 一次 MC sweep = n_sites 次单自旋更新尝试

**算法**:
1. 随机选择一个位点
2. 生成新自旋方向（Marsaglia）
3. 计算能量变化: $dE = -J_{mc} S^2 \sum_{j \in NN} (\mathbf{S}_{new} - \mathbf{S}_{old}) \cdot \mathbf{S}_j$
4. Metropolis 接受准则

---

##### 1.13.5 `mc_measure_magnetization(mc, mvec_out)`

**功能**: 测量磁化强度

**公式**:
$$\mathbf{m} = \frac{1}{N} \sum_{i=1}^N \mathbf{S}_i$$

输出为分数磁化矢量（|mvec| ∈ [0,1]）。

---

##### 1.13.6 `classical_mc_run(J_mc_in, S_mag_in, frac_pos, n_f_sites, avec, T_start, T_step, T_end, mvec_vs_T, n_temps, mc_supercell_in)`

**功能**: 温度扫描主循环

**超胞自动检测**:
- 若 |a_i| > 2·min(|a_j|, |a_k|): N_i = 1（真空方向）
- 否则: N_i = 10（标准热力学极限）

**参数**: N_THERM = 5000（热化步）, N_MEAS = 10000（测量步）, MEAS_EVERY = 10

---

### 1.14 wannlog — 计时与日志

**文件**: [`modules/wannlog.f90`](wannchi/modules/wannlog.f90) (186 行)

**功能**: 轻量级 wall-clock 计时和消息日志工具

#### 子程序详细说明

##### 1.14.1 `log_init()`

**功能**: 初始化计时器，记录程序开始时间

**操作**:
1. 重置 wl_n_timers = 0, wl_n_messages = 0
2. 记录 cpu_time(wl_wall_start)
3. 打印初始化横幅

---

##### 1.14.2 `log_start(label)`

**功能**: 启动（或重启）命名计时器

**算法**: 查找现有计时器或创建新条目，重置开始时间。

---

##### 1.14.3 `log_stop(label)`

**功能**: 停止计时器并累加时间

**算法**: 找到匹配计时器，累加 elapsed += t_now - start_time。

---

##### 1.14.4 `log_msg(msg)`

**功能**: 添加消息到日志缓冲区并立即回显

---

##### 1.14.5 `log_print_summary()`

**功能**: 打印计时表格和消息到 stdout

**输出格式**:
```
============================================================
  wannlog: timing summary
============================================================
  Total CPU time:  XXX.XXX s

  Stage                                    Calls     CPU (s)  %Total
  ---------------------------------------------------------------
  downfolding                                   1     XX.XXX   XX.X
  bayesian_optimize                             1    XXX.XXX   XX.X
  ---------------------------------------------------------------
```

---

### 1.15 wanneff_JS — J-S Kondo 耦合拟合工作流

**文件**: [`src/wanneff_JS.f90`](wannchi/src/wanneff_JS.f90) (1176 行)

**依赖**: `constants`, `wanndata`, `linalgwrap`, `gp_bo`, `cma_es`, `classical_mc`, `transp_calc`, `wannlog`

**功能**: 端到端流程：downfolding → J·S 优化 → MC → 输运

#### 物理模型

```
H_seed = [ H_CC  H_CF ]    (含 f 电子的完整系统)
        [ H_FC  H_FF ]

H_eff_CC = H_CC - H_CF · H_FF^{-1} · H_FC   (舒尔补)
H_bare   = seedbare 的 Wannier90 HR（纯导带）
```

#### 关键子程序

##### 1.15.1 `downfold_rspace(ham_eff_cc, ham_full, cc_idx, ff_idx, n_cc, n_ff)`

**功能**: R 空间舒尔补 downfolding

**公式**:
$$H_{eff}^{CC}(\mathbf{R}) = H_{CC}(\mathbf{R}) - H_{CF}(\mathbf{R}) \cdot H_{FF}(\mathbf{R})^{-1} \cdot H_{FC}(\mathbf{R})$$

**算法**:
1. 对每个 R 格子提取 4 个块: H_CC, H_CF, H_FC, H_FF
2. 若 H_FF ≈ 0（无 FF 跳跃）→ H_eff = H_CC
3. 否则: 求逆并计算舒尔补项

**⚠️ 已知限制**: 符号取决于 FF 态是在费米能以下（填充）还是以上（空）。当前实现始终使用负号。FF 态在费米能以上时需要正号。

---

##### 1.15.2 `js_objective(params, n_params, val)`

**功能**: 贝叶斯优化的目标函数

**目标函数**:
$$L(J, S) = \frac{1}{N_k} \sum_{\mathbf{k}} \left\| \text{sort}(\lambda(H_{bare}(\mathbf{k}) + H_{JS}(\mathbf{k}))) - \text{sort}(\lambda(H_{eff}^{CC}(\mathbf{k}))) \right\|^2$$

**算法**:
1. 根据 eff_mode 解包参数
2. 对每个 k 点: 计算 $H_{bare}(\mathbf{k})$ → 添加 $H_{JS}$ → 对角化
3. 累加排序后本征值的 L2 差异

---

## 2. 源文件详细子程序分析

### 2.1 input — 输入文件解析

**文件**: [`src/input.f90`](wannchi/src/input.f90) (436 行)

#### 全局变量

```fortran
character(len=80) seed          ! SeedName
real(dp) mu, beta               ! 费米能和温度
integer nqpt                   ! q点数
real(dp), dimension(:, :), allocatable :: qvec  ! q 向量
integer nnu                    ! 频率点数
real(dp) emin, emax            ! 实频范围
real(dp) eps                   ! 虚部展宽
complex(dp), dimension(:), allocatable :: nu  ! 频率点
logical spectra_calc           ! 是否实频计算
logical trace_only             ! 是否只算 trace
logical ff_only                ! 是否只算 FF 块
logical use_lehman             ! 是否用 Lehman 算法
logical fast_calc              ! 是否用快速算法
integer npade                  ! Padé 极点数
```

#### 子程序详细说明

##### 2.1.1 `read_input(codename)`

**功能**: 读取输入文件

**接口**:
```fortran
SUBROUTINE read_input(codename)
    character(*), intent(in) :: codename
```

**实现方法**:
- 解析 Fortran namelist 文件
- 设置默认参数
- 计算频率网格

**输入文件格式** (wannchi.inp):
```fortran
&SYSTEM
    seed='wannier90'
    beta=2000.d0
    mu=0.d0
    spectra_calc=.false.
/

&CONTROL
    use_lehman=.false.
    trace_only=.false.
    ff_only=.true.
    fast_calc=.true.
    npade=80
    nnu=1
    eps=0.001
/
```

---

##### 2.1.2 `read_qpoints()`

**功能**: 读取 QPOINTS 文件

**接口**:
```fortran
SUBROUTINE read_qpoints()
```

**文件格式**:
- mode=0: 单点，`qx qy qz`
- mode=1: 线模式，`nseg ninterpolate`，每段 `x1 y1 z1 x2 y2 z2`
- mode=2: 平面网格，`nint1 nint2`，然后顶点、两方向
- mode=3: 完整 BZ 网格，`nk1 nk2 nk3`

---

##### 2.1.3 `finalize_input()`

**功能**: 释放输入数组

**接口**:
```fortran
SUBROUTINE finalize_input()
```

---

### 2.2 green.f90 — 格林函数计算

**文件**: [`src/green.f90`](wannchi/modules/green.f90) (67 行)

#### 子程序详细说明

##### 2.2.1 `calc_g0(gf, hk, w, ndim, inv)`

**功能**: 计算无相互作用格林函数 $G^0 = (w - H)^{-1}$

**接口**:
```fortran
subroutine calc_g0(gf, hk, w, ndim, inv)
    complex(dp), dimension(ndim, ndim), intent(out) :: gf
    complex(dp), dimension(ndim, ndim), intent(in) :: hk
    complex(dp), intent(in) :: w
    integer, intent(in) :: ndim
    logical, intent(in) :: inv
```

**实现方法**:
- 构建 $w \cdot I - H$
- 如果 inv=true，求逆得到格林函数
- 否则，返回矩阵本身

**公式**:
$$G^0(\mathbf{k}, \omega) = [\omega - H(\mathbf{k})]^{-1}$$

---

##### 2.2.2 `calc_corrFF_gf(gf, hff, sigmat, Ecc_diag, Vfc, w, ndimf, ndimc, inv)`

**功能**: 计算含杂化的关联格林函数

**接口**:
```fortran
subroutine calc_corrFF_gf(gf, hff, sigmat, Ecc_diag, Vfc, w, ndimf, ndimc, inv)
    complex(dp), dimension(ndimf, ndimf), intent(out) :: gf
    complex(dp), dimension(ndimf, ndimf), intent(in) :: hff
    complex(dp), dimension(ndimf, ndimf), intent(in) :: sigmat
    real(dp), dimension(ndimc), intent(in) :: Ecc_diag
    complex(dp), dimension(ndimf, ndimc), intent(in) :: Vfc
    complex(dp), intent(in) :: w
    integer, intent(in) :: ndimf, ndimc
    logical, intent(in) :: inv
```

**实现方法**:
1. 计算 bath 格林函数: $G_{CC} = \text{diag}(1/(w - E_{CC}))$
2. 计算杂化函数: $\Delta = -V_{FC} \cdot G_{CC} \cdot V_{FC}^\dagger$
3. 计算 $G_{FF}^{-1} = w - H_{FF} - \Sigma_{FF} - \Delta$
4. 求逆得到 $G_{FF}$

**算法**:
```
! Bath 格林函数 (对角)
gcc = 1 / (w - Ecc_diag)

! 杂化函数
Delta = - Vfc · diag(gcc) · Vfc^dagger

! FF 格林函数
Gff_inv = w * I - Hff - sigmat - Delta

if (inv):
    gf = inv(Gff_inv)
else:
    gf = Gff_inv
```

**公式**:
$$\Delta(\omega) = -V_{FC} \cdot G_{CC}(\omega) \cdot V_{FC}^\dagger$$

$$G_{FF}^{-1}(\omega) = \omega - H_{FF} - \Sigma_{FF} - \Delta$$

其中 bath 格林函数:
$$G_{CC}(\omega) = \frac{1}{\omega - E_{CC}}$$

---

### 2.3 output.f90 / output_chi.f90 — 输出工具

#### 子程序详细说明

##### 2.3.1 `output_header(...)` (output.f90)

**功能**: 写谱函数头部

**接口**:
```fortran
SUBROUTINE output_header(fout, seed, nk1, nk2, nk3)
```

---

##### 2.3.2 `output_spectral(...)` (output.f90)

**功能**: 写谱函数数据

**接口**:
```fortran
SUBROUTINE output_spectral(fout, akw, nk, nw)
```

**公式**:
$$A(\mathbf{k}, \omega) = -\frac{1}{\pi} \text{Im} \, G(\mathbf{k}, \omega)$$

---

##### 2.3.3 `output_chi(...)` (output_chi.f90)

**功能**: 写 χ(q,ω) 数据

**接口**:
```fortran
SUBROUTINE output_chi(fout, chi, ndim, nqpt, nw, qvec, omega)
```

---

### 2.4 compute_chi.f90 — 响应函数核心计算

**文件**: [`src/compute_chi.f90`](wannchi/src/compute_chi.f90) (~1500 行)

**重要模块**: `MODULE chi_internal` (第 1-280 行)

#### chi_internal 模块变量

**快速算法共享**:
```fortran
integer, allocatable :: kq_idx(:)  ! k → k+q 的索引映射
```

**Lehman 表象相关**:
```fortran
real(dp), allocatable :: ek(:), ekq(:)        ! k 和 k+q 点的能带能量
real(dp), allocatable :: occ_k(:), occ_kq(:)  ! 费米占据率
complex(dp), allocatable :: hk(:,:), hkq(:,:)  ! Hamiltonian / 本征向量
real(dp), allocatable :: Skq(:,:)  ! 结构因子 |U_k^† U_{k+q}|²
```

**Lehman 快速算法**:
```fortran
real(dp), allocatable :: eig_full(:,:), occ_full(:,:)  ! (norb, nkirr) 全本征值
complex(dp), allocatable :: Uk_full(:,:,:)  ! (norb, norb, nkirr) 全本征向量
```

**G*G 算法（关联 case）**:
```fortran
complex(dp), allocatable :: Hff_k(:,:), Hff_kq(:,:)  ! F 块 Hamiltonian
complex(dp), allocatable :: Vfc_k(:,:), Vfc_kq(:,:)  ! F-C 耦合
real(dp), allocatable :: Ecc_k(:), Ecc_kq(:)        ! C 块能量（对角）
```

#### 子程序详细说明

##### 2.4.1 `prepare_lehman(fast_calc, step)`

**功能**: 初始化 Lehman 计算

**接口**:
```fortran
SUBROUTINE prepare_lehman(fast_calc, step)
    logical, intent(in) :: fast_calc
    integer, intent(in) :: step
```

**实现方法**:
- 分布式对角化 k-mesh
- 预计算本征值/向量

---

##### 2.4.2 `calc_chi_bare_matrix_lehman_kernel(w, nw)`

**功能**: 核心核 - 对单个 k 点累加 FF/FC/CF/CC 块贡献

**接口**:
```fortran
SUBROUTINE calc_chi_bare_matrix_lehman_kernel(w, nw)
    complex(dp), intent(in) :: w
    integer, intent(in) :: nw
```

**算法**:
```
do ik = first_idx to last_idx:
    do iq = 1 to nqpt:
        k = kvec(ik)
        q = qvec(iq)
        kq = k + q
        
        ! 对角化 H(k) 和 H(k+q)
        call calc_hk(hk, ham, k)
        call eigen(eig_k, hk, norb)
        call calc_hk(hkq, ham, kq)
        call eigen(eig_kq, hkq, norb)
        
        ! 计算响应函数
        do ibnd = 1 to norb:
            do jbnd = 1 to norb:
                fact = (occ_k(ibnd) - occ_kq(jbnd)) / (w + eig_kq(jbnd) - eig_k(ibnd) + i*eps)
                chi += fact * |S_ij|^2
```

**公式**:
$$\chi^0_{ij}(\mathbf{q}, \omega) = \frac{1}{N_k} \sum_{\mathbf{k}} \sum_{mn} \frac{(f_m - f_n) \cdot |S_{ij}^{mn}(\mathbf{k},\mathbf{q})|^2}{\omega + \epsilon_n(\mathbf{k+q}) - \epsilon_m(\mathbf{k}) + i\eta}$$

其中 $S_{ij}^{mn} = \langle u_m(\mathbf{k}) | \phi_i \rangle \langle \phi_j | u_n(\mathbf{k+q}) \rangle$

---

##### 2.4.3 `calc_chi_bare_matrix_lehman(chi0, w, nw, qv)`

**功能**: 慢速版 - 逐 k 循环

**接口**:
```fortran
SUBROUTINE calc_chi_bare_matrix_lehman(chi0, w, nw, qv)
    complex(dp), intent(out) :: chi0(*)
    complex(dp), intent(in) :: w
    integer, intent(in) :: nw
    real(dp), intent(in) :: qv(3)
```

---

##### 2.4.4 `calc_chi_bare_matrix_lehman_fast(chi0, w, nw, qv)`

**功能**: 快速版 - 预计算全量本征值/向量

**接口**:
```fortran
SUBROUTINE calc_chi_bare_matrix_lehman_fast(chi0, w, nw, qv)
    complex(dp), intent(out) :: chi0(*)
    complex(dp), intent(in) :: w
    integer, intent(in) :: nw
    real(dp), intent(in) :: qv(3)
```

---

##### 2.4.5 `prepare_GG(fast_calc, step)`

**功能**: 初始化 G*G 计算

**接口**:
```fortran
SUBROUTINE prepare_GG(fast_calc, step)
    logical, intent(in) :: fast_calc
    integer, intent(in) :: step
```

**实现方法**:
- 初始化 Padé 极点
- 求解 k-mesh 本征值/向量

---

##### 2.4.6 `calc_chi_bare_matrix_GG_kernel(w, nw)`

**功能**: 核心核 - 用 Padé 极点和 Green 函数乘积累加 χ

**接口**:
```fortran
SUBROUTINE calc_chi_bare_matrix_GG_kernel(w, nw)
    complex(dp), intent(in) :: w
    integer, intent(in) :: nw
```

**算法**:
```
! 使用 Padé 求和近似频率求和
do ipole = 1 to npole:
    z_pos = i * zp(ipole) / beta
    z_neg = -i * zp(ipole) / beta
    
    ! 正极点贡献
    do ik = first_idx to last_idx:
        Gk_pos = 1 / (z_pos - eig_k)
        Gkq_pos = 1 / (z_pos + w - eig_kq)
        chi += eta(ipole) * Gk_pos * Gkq_pos * |S|^2
    
    ! 负极点贡献 (c.c.)
    do ik = first_idx to last_idx:
        Gk_neg = 1 / (z_neg - eig_k)
        Gkq_neg = 1 / (z_neg + w - eig_kq)
        chi += eta(ipole) * Gk_neg * Gkq_neg * |S|^2
```

**公式**:
$$\frac{1}{\beta}\sum_{\omega_n} f(i\omega_n) \approx \sum_{p=1}^{N_{\text{pole}}} \eta_p \left[f(iz_p/\beta) + f(-iz_p/\beta)\right]$$

$$\chi^0_{ij}(\mathbf{q},\omega) = \frac{1}{N_k}\sum_k \sum_{mn}\left\{\sum_{p=1}^{N_{\text{pole}}}\eta_p\left[\frac{1}{iz_p/\beta - \epsilon_{mk}}\cdot\frac{1}{iz_p/\beta+\omega-\epsilon_{nkq}} + \text{c.c.}\right]\right\}\cdot U^*_{i,m}(k)\,U_{j,n}(kq)\,U^*_{j,n}(kq)\,U_{i,m}(k)$$

---

##### 2.4.7 `calc_chi_trace_from_matrixFF(chi0, nw)`

**功能**: 从 FF 块提取 trace

**接口**:
```fortran
SUBROUTINE calc_chi_trace_from_matrixFF(chi0, nw)
    complex(dp), intent(in) :: chi0(*)
    integer, intent(in) :: nw
```

---

### 2.5 wannchi.f90 — 主程序 (bare susceptibility)

**文件**: [`src/wannchi.f90`](wannchi/src/wannchi.f90) (137 行)

#### 程序流程

```
1. init_para → MPI 初始化
2. read_input('wannchi') → 读取 wannchi.inp
3. read_ham(ham, seed) → 读取 Wannier Hamiltonian
4. wannham_shift_ef(ham, mu) → 费米能级移动
5. read_posfile → 读取原子位置
6. read_kmesh → 读取 k-mesh
7. read_qpoints → 读取 q-points
8. if (.not.trace_only) read_RPA → 可选读取 RPA 块定义
9. 选择算法:
   - use_lehman=true → prepare_lehman()
   - else → prepare_GG() + print_pade()
10. loop iq = 1 to nqpt:
        if trace_only:
            calc_chi_bare_trace_lehman/GG_*
        else:
            calc_chi_bare_matrix_lehman/GG_*
        输出文本文件 chi0tr.dat (trace)
        输出二进制文件 chiff.dat (FF 块)
        若 ff_only=false: 还有 chicc.dat, chifc.dat, chicf.dat
11. finalize → 释放所有资源
```

---

### 2.6 wannchiRPA.f90 — 主程序 (RPA-dressed susceptibility)

**文件**: [`src/wannchiRPA.f90`](wannchi/src/wannchiRPA.f90) (69 行)

#### 程序流程

```
1. init_para → MPI 初始化
2. read_input → 读取参数
3. read_ham_dim → 读取 Hamiltonian 维度
4. read_posfile → 读取原子位置
5. read_qpoints → 读取 q-points
6. read_RPA → 读取 FF 块结构和 U 矩阵
7. init_chi_matrix_RPA(nnu) → 分配 χ 和 χ⁰ 矩阵
8. loop iq = 1 to nqpt:
        read_chi_matrix_RPA((iq-1)*nnu, nnu, ff_only)
        calc_chiRPA(chiff, chicc, chifc, chicf, ...)
        save_chi_matrix(iq*nnu, nnu, ff_only)
        calc_chi_trace_from_matrixFF(chi, nnu)
        输出文本文件 chiRPAtr.dat
9. All Done
```

---

### 2.7 postchi.f90 — 响应函数后处理

**文件**: [`src/postchi.f90`](wannchi/src/postchi.f90) (~150 行)

#### 功能

- 读取 `{seed}.impdef` 并 `setup_mapping` 构建 F/C 分区
- 读取 `chiff.dat` 加载 FF 块 χ 矩阵
- 若 SOC 检测到 (`imp%ndim == 2*(lang+1)`)，将 spin-off-diagonal 矩阵元置零
- 对每个频率点做本征分解: `eigen(chiEig, chiff, nFFidx)`
- 输出 `postchi.dat` (top-5 本征值及对应本征向量)

---

### 2.8 wannband.f90 — 谱函数计算

**文件**: [`src/wannband.f90`](wannchi/src/wannband.f90) (~200 行)

#### 功能

**输入文件**: `wannband.inp`

计算谱函数:
$$A(\mathbf{k}, \omega) = -\frac{1}{\pi} \text{Im} \, G(\mathbf{k}, \omega)$$

- 若 `.impdef` 不存在：计算无自能的裸格林函数谱函数
- 若 `.impdef` 存在：读取 `.sig`，计算含自能的关联谱函数

---

### 2.9 wanneff_JS.f90 — J-S Kondo 交换耦合拟合

**文件**: [`src/wanneff_JS.f90`](wannchi/src/wanneff_JS.f90) (703 行)

#### 类型和变量

```fortran
! 输入参数
character(len=80) :: seed      ! 含 f 电子的完整系统 seed
character(len=80) :: seedbare  ! 仅导带的系统 seed
logical :: eff_js = .true.     ! 是否进行 J-S 拟合
logical :: eff_mc = .false.    ! 是否进行蒙特卡洛温度扫描
logical :: J_TENSOR = .false.  ! .false.=标量 J; .true.=张量 J(R)
integer :: bayes_niter = 200   ! 贝叶斯优化迭代次数
real(dp) :: tol_Jeff = 1e-4   ! 裁剪小 J(R) 的阈值
real(dp) :: J_bounds(2) = (/0.0, 0.5/)  ! J 边界
real(dp) :: S_bounds(2) = (/0.1, 5.0/)  ! S 边界
```

#### 关键子程序

##### 2.9.1 `downfold_to_cc(hk_cc, hk_full, norb_full, norb_cc, cc_idx)`

**功能**: 静态 downfolding：提取有效 CC Hamiltonian

**接口**:
```fortran
subroutine downfold_to_cc(hk_cc, hk_full, norb_full, norb_cc, cc_idx)
    complex(dp), intent(out) :: hk_cc(norb_cc, norb_cc)
    complex(dp), intent(in) :: hk_full(norb_full, norb_full)
    integer, intent(in) :: norb_full, norb_cc
    integer, intent(in) :: cc_idx(norb_cc)
```

**公式**:
$$H_{\text{eff}}^{CC}(\mathbf{k}) = -[G_{CC}(\mathbf{k}, 0)]^{-1}$$

其中 $G_{\text{full}}(\mathbf{k}, 0) = (0 - H_{\text{seed}}(\mathbf{k}))^{-1}$

---

##### 2.9.2 `determine_cc_indices(cc_idx, norb_full, norb_bare, nbasis_full, nbasis_bare, nsite_full, nsite_bare)`

**功能**: 确定 CC 轨道索引

**接口**:
```fortran
subroutine determine_cc_indices(cc_idx, norb_full, norb_bare, nbasis_full, nbasis_bare, nsite_full, nsite_bare)
    integer, intent(out) :: cc_idx(*)
    integer, intent(in) :: norb_full, norb_bare
    integer, intent(in) :: nbasis_full(nsite_full), nbasis_bare(nsite_bare)
    integer, intent(in) :: nsite_full, nsite_bare
```

**实现方法**:
- 对比 seed 和 seedbare 中每站点的轨道数
- seed 中比 seedbare 多出的轨道为 FF 轨道
- seedbare 中的轨道为 CC 轨道

---

##### 2.9.3 `add_js_coupling(ham_out, norb_bare, J_val, Svec, irpt)`

**功能**: 添加 J·(S·σ)/2 交换耦合

**接口**:
```fortran
subroutine add_js_coupling(ham_out, norb_bare, J_val, Svec, irpt)
    complex(dp), intent(out) :: ham_out(norb_bare, norb_bare)
    integer, intent(in) :: norb_bare
    real(dp), intent(in) :: J_val, Svec(3)
    integer, intent(in) :: irpt
```

**公式**:
$$H_{JS}(\mathbf{R}) = J(\mathbf{R}) \cdot \frac{\mathbf{S} \cdot \boldsymbol{\sigma}}{2}$$

---

##### 2.9.4 `js_objective(params, n_params, val)`

**功能**: 贝叶斯优化的目标函数

**接口**:
```fortran
function js_objective(params, n_params) result(val)
    real(dp), intent(in) :: params(n_params)
    integer, intent(in) :: n_params
    real(dp) :: val
```

**目标函数**:
$$L(J, S) = \frac{1}{N_k} \sum_{\mathbf{k}} \left\| \text{sort}(\lambda(H_{\text{bare}} + H_{JS})) - \text{sort}(\lambda(H_{\text{eff}}^{CC})) \right\|^2$$

---

## 3. Downfolding 架构说明

### 3.1 物理模型

```
H_seed = [ H_CC  H_CF ]
         [ H_FC  H_FF ]    (含 f 电子的完整系统)

H_eff_CC = H_CC - H_CF * H_FF^{-1} * H_FC   (Schur 补)
H_bare   = seedbare 的 Wannier90 HR (仅导带)
```

**关键假设**：FF 轨道与 CC 轨道位于不同原子位点，因此 CC-FF 耦合 `H_CF` 是跳跃型的。

### 3.2 R-space 与 K-space Downfolding

**R-space（此处使用）**：
- 通过 Schur 补直接计算 `H_eff_CC(R)`
- 逆傅里叶变换得到 `H_eff_CC(k)` 用于本征值比较
- 对局域化 Wannier 函数数值更稳定

**K-space（已废弃）**：
- 计算 `G_full(k,0) = (0 - H_seed(k))^{-1}`
- 提取 CC 块：`H_eff_CC(k) = -[G_CC(k,0)]^{-1}`
- 当 FF 能量接近费米面时容易产生鬼态

### 3.3 ⚠️ 已知限制：能量窗口符号依赖

Schur 补的符号取决于能量窗口：

| FF 态 | 符号 | 物理含义 |
|--------|------|---------|
| 费米能以下（填充） | `-H_CF * H_FF^{-1} * H_FC` | 电子虚跳跃到空的 FF 态 |
| 费米能以上（空） | `+H_CF * H_FF^{-1} * H_FC` | 从填充 FF 态的虚跳跃 |

**当前行为**：始终使用负号，与能量窗口无关。

**检测方法**：检查 R=0 处 `H_FF` 对角元（on-site 能量）。若对角元为正（高于费米），则翻转符号。

**测试用例**：`kagome_f_spinor_test.py` 将 f 轨道置于 `EF + 0.4 eV` → 需要正号。

---

## 4. 测试文档

### 4.1 kagome_f_spinor_test.py (Test 6)

**位置**：`tests/kagome_f_spinor_test.py`
**用途**：端到端流水线测试 — Python 仅做 I/O，所有计算在 Fortran 中

**晶格**：
- a1 = (1, 0, 0), a2 = (0.5, √3/2, 0), a3 = (0, 0, 10)
- 4 个 kagome 位点 + 位点 4 上的 f 轨道装饰
- seed：10 个自旋轨道（4 位点 × [1,1,1,2] 基底）
- seedbare：8 个自旋轨道（4 位点 × [1,1,1,1] 基底）

**测试变体**：
1. `full_pipeline_e2e`：标量 J，完整工作流
2. `cubic_pipeline`：简单立方晶格变体
3. `tensor_pipeline_*`：张量 J 模式
4. `kagome3_pipeline`：Heisenberg 反铁磁体
5. `js_tensor_pipeline_*`：J_S_TENSOR 模式

**用法**：
```bash
cd tests
source /Users/ykxu/Projects/hrJS/hrJS/bin/activate
python3 kagome_f_spinor_test.py
```

**输出**：测试结果在 `test_tsf0.5_output/` 目录，包含：
- `test6_report.md`：所有测试结果汇总
- 能带比较图
- 输运性质随温度变化图

### 4.2 algo_test.py (Tests 1-5)

**位置**：`tests/algo_test.py`
**用途**：纯 Python 算法验证 — 无需 Fortran

| 测试 | 名称 | 验证内容 |
|------|------|---------|
| 1 | Kagome reference | Wannier 插值 + 能带结构 |
| 2 | Bayesian 8D | GP-BO 在 8D Ackley 函数上 |
| 3 | MC square lattice | Heisenberg MC 磁化强度 |
| 4 | Berry curvature | AHC 计算 |
| 5 | Fortran pipeline | （无 Fortran 时跳过）|

---

## 5. 构建说明

### 5.1 make.sys.laptop (gfortran + Accelerate)

**位置**：`build_laptop/Makefile`
**配置**：`make.sys.laptop`

```makefile
F90 = gfortran
F90FLAGS = -O2 -framework Accelerate -fpp
LAPACKLIBS = -framework Accelerate
```

**关键特性**：
- 无 MPI（`para_serial.f90` 提供桩）
- gfortran 使用 `-fpp`（Fortran 预处理器）
- Accelerate 框架提供 LAPACK/BLAS

### 5.2 para_serial.f90

**位置**：`modules/para_serial.f90`
**用途**：MPI 并行工具的串行桩

提供空操作版本：
- `init_para`, `finalize_para`
- `distribute_calc`（设置 `first_idx=1, last_idx=n`）
- `para_merge_cmplx`, `para_sync_logical`, `para_sync0`

### 5.3 构建命令

```bash
# Laptop 构建
cd build_laptop
make wanneff_js.x wannband.x
cp wanneff_js.x wannband.x ../src/

# Fortran 单元测试
cd tests
make test_all

# 模块编译
cd modules
make mod.a
```

---

## 6. 问答

### Q1: n_jrpt 是什么？

`n_jrpt` 是张量模式（eff_mode=2 或 3）中用于 J(R) 张量的 R 矢量（实空间格矢）数量。

- 标量模式（eff_mode=1）：`n_jrpt = 0`，仅使用 J(R=0)
- 张量模式：`n_jrpt` 等于 Wannier Hamiltonian 中的 R 矢量数（若指定了 `J_R_range` 则为自定义网格）

每个 R 矢量有一个关联的 J(R) 值（在 mode 3 中还有一个关联的 S(R) 矢量）。

### Q2: Wigner-Seitz 晶胞划分是如何工作的？

Wigner-Seitz（WS）晶胞划分决定了哪些格点 `R` 属于哪个相邻晶胞的周期像。

**`find_ws_rvectors` 中的算法**：
1. 在盒子 `[-nr1:nr1] × [-nr2:nr2] × [-nr3:nr3]` 中遍历候选 R 矢量
2. 对每个候选 `R`，计算到所有 125 个周期像 `R + n1*a1 + n2*a2 + n3*a3`（`ni ∈ {-2,-1,0,1,2}`）的距离
3. 找到最小距离
4. 若原始 R 到自身的距离最小（即"主"晶胞），则包含它，权重 = 等价 R 矢量数

**在 wanndata.f90 中的使用**：
- `rvec`：WS 晶胞中 R 矢量的分数坐标
- `weight`：等价 R 矢量数（用于傅里叶变换的归一化）

---

## 7. 附录：Fortran 语法参考

### 7.1 TYPE 定义

```fortran
TYPE :: type_name
  integer :: field1
  real(dp), allocatable :: array(:,:)
END TYPE type_name
```

### 7.2 Interface 块（回调）

```fortran
interface
  function objective_func(params, n) result(val)
    use constants, only : dp
    integer, intent(in) :: n
    real(dp), dimension(n), intent(in) :: params
    real(dp) :: val
  end function
end interface
```

### 7.3 Intent 修饰符

| Intent | 含义 |
|--------|------|
| `intent(in)` | 输入 — 不被修改 |
| `intent(out)` | 输出 — 将被写入 |
| `intent(inout)` | 两者 — 可能会被修改 |

### 7.4 数组切片

```fortran
real(dp), dimension(10) :: a
a(1:5)        ! 前 5 个元素
a(2:10:2)    ! 元素 2,4,6,8,10（步长 2）
a(:)         ! 所有元素
```

### 7.5 Module 与 Program

- **MODULE**：类型、常量、过程的集合；`CONTAINS` 标记过程定义
- **PROGRAM**：可执行单元；`CALL` 过程，`USE` 模块
- **内部过程**：在模块/程序内部 `CONTAINS` 之后定义的过程；可访问模块级变量

### 7.6 ZGEMM（复数矩阵乘法）

```fortran
call zgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
! C = alpha * op(A) * op(B) + beta * C
! op(X) = X（无转置），X^T（转置），X^H（共轭转置）
```

**常见用法**：
```fortran
! vx_band = eigvec^H . vx_orb . eigvec
call zgemm('N', 'N', norb, norb, norb, zone, vx, norb, eigvec, norb, zzero, tmp, norb)
call zgemm('C', 'N', norb, norb, norb, zone, eigvec, norb, tmp, norb, zzero, vx_band, norb)
```

### 7.7 随机数生成

```fortran
real(dp) :: u
call random_number(u)  ! u ∈ [0, 1)
u = 2.0_dp * u - 1.0_dp  ! 映射到 [-1, 1)
```

### 7.8 数组构造

```fortran
real(dp), dimension(3) :: vec
vec = [1.0_dp, 2.0_dp, 3.0_dp]  ! 1D 数组构造器
! 或显式界：
real(dp), dimension(0:2) :: vec
```

### 7.9 带检查的释放

```fortran
if (allocated(arr)) deallocate(arr)
if (associated(ptr)) nullify(ptr)
```

### 7.10 Merge（条件赋值）

```fortran
! 若条件为真，使用 tval；否则使用 fval
NX = merge(1, N_DEFAULT, len_a1 > 2.0_dp * len_min)
```

---

## 总结

本文档提供了 WannChi 项目中所有子程序的详细分析，包括：

1. **功能描述**: 每个子程序的具体作用
2. **接口**: 输入输出参数类型和含义
3. **实现方法**: 算法实现细节
4. **算法**: 计算流程和逻辑
5. **对应公式**: 物理和数学公式

所有子程序均按照模块和源文件进行组织，便于代码理解和维护。