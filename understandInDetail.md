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
8. [MPI 并行化状态分析](#8-mpi-并行化状态分析)
9. [已知错误与设计缺陷](#9-已知错误与设计缺陷)

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

**源代码**:
```fortran
#if defined __MPI
  integer ierr
  !
  CALL mpi_init(ierr)
  !
  CALL mpi_comm_rank(mpi_comm_world, inode, ierr)
  CALL mpi_comm_size(mpi_comm_world, nnode, ierr)
  !
  if (inode.eq.0) write(stdout, *) trim(codename)//" running on ", nnode, " nodes..."
  !
  allocate(map(nnode, 2))
  !
#else
  inode=0
  nnode=1
  write(stdout, *) trim(codename)//" serial ..."
#endif
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

**源代码**:
```fortran
#if defined __MPI
  integer ierr
  CALL mpi_finalize(ierr)
#endif
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

**源代码**:
```fortran
#if defined __MPI
  !
  map(:, :)=0
  first_idx=inode*nidx/nnode+1
  last_idx=(inode+1)*nidx/nnode
  map(inode+1, 1)=first_idx-1
  map(inode+1, 2)=last_idx-first_idx+1
  !
  call para_merge_int(map, 2*nnode)
  !
#else
  first_idx=1
  last_idx=nidx
#endif
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

**源代码**:
```fortran
#if defined __MPI
  !
  ! We need to use mpi_type
  integer ierr, blk_cmplx
  !
  call mpi_type_contiguous(blk_size, MPI_DOUBLE_COMPLEX, blk_cmplx, ierr)
  ! new type object created
  call mpi_type_commit(blk_cmplx, ierr)
  ! now type1 can be used for communication
  call mpi_gatherv(dat, map(inode+1, 2), blk_cmplx, fulldat, map(:, 2), map(:, 1), blk_cmplx, 0, mpi_comm_world, ierr)
  call mpi_type_free(blk_cmplx, ierr)
  !
#endif
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

**源代码**:
```fortran
  if (inode.eq.0) then
    write(stdout, *) " # Reading file "//trim(seed)//"_hr.dat"
    !
    open(unit=fin, file=trim(seed)//"_hr.dat")
    !
    read(fin, *)
    read(fin, *) tt(1) ! norb
    read(fin, *) tt(2) ! nrpt
    !
    write(stdout, *) " #  Dimensions:"
  endif
  !
  CALL para_sync_int(tt, 2)
  ham%norb=tt(1)
  ham%nrpt=tt(2)
  !
  allocate(ham%hr(ham%norb, ham%norb, ham%nrpt))
  allocate(ham%weight(ham%nrpt))
  allocate(ham%rvec(3, ham%nrpt))
  allocate(ham%tau(3, ham%norb))
  !
  ham%tau(:,:)=0.d0
  !
  if (inode.eq.0) then
    write(stdout, *) "    # of orbitals:", ham%norb
    write(stdout, *) "    # of real-space grid:", ham%nrpt
    allocate(wt(1:ham%nrpt))
    read(fin, '(15I5)') (wt(irpt),irpt=1,ham%nrpt)
    ham%weight(:)=wt(:)
    deallocate(wt)
    !
    do irpt=1, ham%nrpt
      do iorb=1, ham%norb
        do jorb=1, ham%norb
          read(fin, *) tt, a, b
          if ((jorb.eq.1).and.(iorb.eq.1)) then
            ham%rvec(:, irpt)=tt(1:3)
            if (tt(1)**2+tt(2)**2+tt(3)**2.eq.0) then
              ham%r000=irpt
            endif
          endif
          ham%hr(jorb, iorb, irpt)=CMPLX(a, b, KIND=dp)
        enddo
      enddo
    enddo
    !
    close(unit=fin)
    write(stdout, *) " # Done."
  endif
  !
  CALL para_sync_cmplx(ham%hr, ham%norb * ham%norb * ham%nrpt)
  CALL para_sync_real(ham%weight, ham%nrpt)
  CALL para_sync_real(ham%rvec, 3*ham%nrpt)
  CALL para_sync0(ham%r000)
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

**源代码**:
```fortran
  hk(:,:)=cmplx_0
  !
  do io=1, ham%norb
    ktau=sum(kvec(:)*ham%tau(:, io))*twopi
    phase(io)=cmplx(cos(ktau), sin(ktau), KIND=dp)
  enddo
  !
  do ir=1, ham%nrpt
    rdotk=sum(kvec(:)*ham%rvec(:, ir))*twopi
    fact=cmplx(cos(rdotk), sin(rdotk), KIND=dp)/ham%weight(ir)
    !
    do io=1, ham%norb
      do jo=1, ham%norb
        hk(io, jo)=hk(io, jo)+fact*conjg(phase(io))*phase(jo)*ham%hr(io, jo, ir)
      enddo
    enddo
  enddo
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

**源代码**:
```fortran
  real(dp), dimension(ndim) :: work
  integer, dimension(ndim) :: ipiv
  integer :: info
  !
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

**源代码**:
```fortran
  complex(dp), dimension(ndim) :: work
  integer, dimension(ndim) :: ipiv
  integer :: info
  !
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

**源代码**:
```fortran
  integer info
  complex(dp), dimension(2*ndim) :: work
  complex(dp), dimension(3*ndim) :: rwork
  !
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

**源代码**:
```fortran
  integer info
  complex(dp), dimension(2*ndim) :: work
  complex(dp), dimension(2*ndim) :: rwork
  complex(dp), dimension(1, 1)   :: vl
  complex(dp), dimension(ndim, ndim) :: vr
  !
  call zgeev('N', 'V', ndim, xmat, ndim, eig, vl, 1, vr, ndim, work, 2*ndim, rwork, info)
  xmat(:, :)=vr(:, :)
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

**源代码**:
```fortran
  zmat(:, :)=beta*zmat(:, :)
  !
  do ii=1, nidxcp
    i1=idxcp(1, ii)
    i2=idxcp(2, ii)
    do jj=1, ndim
      zmat(i1, jj)=zmat(i1, jj)+alpha*xmat_cp(ii)*ymat(i2, jj)
    enddo
  enddo
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

**源代码**:
```fortran
  zmat(:, :)=beta*zmat(:, :)
  do ii=1, ndim
    do jj=1, nidxcp
      j1=idxcp(1, jj)
      j2=idxcp(2, jj)
      zmat(ii, j2)=zmat(ii, j2)+alpha*xmat(ii, j1)*ymat_cp(jj)
    enddo
  enddo
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

**源代码**:
```fortran
  imp%ndim=ndim
  imp%lang=l
  !
  allocate(imp%gidx(ndim))
  allocate(imp%Utrans(ndim, ndim))
  allocate(imp%sigidx(ndim, ndim))
```

---

##### 1.6.2 `finalize_simp(imp)`

**功能**: 释放 impurity 结构

**接口**:
```fortran
subroutine finalize_simp(imp)
    TYPE(simp), intent(inout) :: imp
```

**源代码**:
```fortran
  if (allocated(imp%gidx))   deallocate(imp%gidx)
  if (allocated(imp%Utrans)) deallocate(imp%Utrans)
  if (allocated(imp%sigidx)) deallocate(imp%sigidx)
```

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

**源代码**:
```fortran
  complex(dp), dimension(imp%ndim, imp%ndim) :: stmp1, stmp2
  integer ii, jj
  !
  call zgemm('N', 'C', imp%ndim, imp%ndim, imp%ndim, &
              cmplx_1, sigmat, imp%ndim, &
              imp%Utrans, imp%ndim, &
              cmplx_0, stmp1, imp%ndim)
  !
  call zgemm('N', 'N', imp%ndim, imp%ndim, imp%ndim, &
              cmplx_1, imp%Utrans, imp%ndim, &
              stmp1, imp%ndim, &
              cmplx_0, stmp2, imp%ndim)
  !
  do ii=1, imp%ndim
    do jj=1, imp%ndim
      !
      if (imp%sigidx(ii, jj)>0) then
        sigpack(imp%sigidx(ii, jj))=sigpack(imp%sigidx(ii, jj))+sigmat(ii, jj)
      endif
      !
    enddo
  enddo
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

**源代码**:
```fortran
  sigmat=cmplx_0
  !
  do ii=1, imp%ndim
    do jj=1, imp%ndim
      !
      if (imp%sigidx(ii, jj)>0) then
        sigmat(ii, jj)=sigpack(imp%sigidx(ii, jj))
      endif
      !
    enddo
  enddo
  !
  call zgemm('N', 'N', imp%ndim, imp%ndim, imp%ndim, &
              cmplx_1, sigmat, imp%ndim, &
              imp%Utrans, imp%ndim, &
              cmplx_0, stmp, imp%ndim)
  !
  call zgemm('C', 'N', imp%ndim, imp%ndim, imp%ndim, &
              cmplx_1, imp%Utrans, imp%ndim, &
              stmp, imp%ndim, &
              cmplx_0, sigmat, imp%ndim)
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

**源代码**:
```fortran
subroutine read_posfile(fn)
  character(*), intent(in) :: fn
  real(dp)  :: alat
  real(dp), dimension(3)  :: xx
  integer   :: ii, jj, nn, ispin
  if (inode.eq.0) then
    open(unit=fin, file=trim(fn))
    read(fin, *) ! First line is comment
    read(fin, *) alat
    do ii=1, 3
      read(fin, *) xx
      avec(:, ii)=xx(:)*alat
    enddo
    bvec(:, :)= avec(:, :)
    call invmat(bvec, 3)
    read(fin, *) nsite, ispin
  endif
  call para_sync_real(avec, 9)
  call para_sync_real(bvec, 9)
  call para_sync_int0(nsite)
  call para_sync_int0(ispin)
  allocate(xat(3, nsite))
  allocate(zat(nsite))
  allocate(nbasis(nsite))
  spinor=(ispin>0)
  if (inode.eq.0) then
    do ii=1, nsite
      read(fin, *) zat(ii), xat(:, ii), nbasis(ii)
    enddo
    close(unit=fin)
  endif
  call para_sync_int(zat, nsite)
  call para_sync_int(nbasis, nsite)
  call para_sync_real(xat, 3*nsite)
  nn=1
  do ii=1, nsite
    do jj=1, nbasis(ii)
      ham%tau(:, nn)=xat(:, ii)
      nn=nn+1
    enddo
  enddo
  if (spinor) then
    do ii=1, nsite
      do jj=1, nbasis(ii)
        ham%tau(:, nn)=xat(:, ii)
        nn=nn+1
      enddo
    enddo
  endif
end subroutine
```

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

**源代码**:
```fortran
subroutine read_kmesh(fn)
  integer ik1, ik2, ik3, iik
  integer, dimension(4) :: tt
  !
  if (inode.eq.0) then
    open(unit=fin, file=trim(fn))
    read(fin, *)      ! COMMENT
    read(fin, *) iik  ! SWITCH
    !
    if (iik.eq.0) then
      ! AUTOMATIC K_MESH
      read(fin, *)    ! Always use Gamma-centered
      read(fin, *) nk1, nk2, nk3
      nkirr=nk1*nk2*nk3
    else
      read(fin, *)
      nkirr=iik
    endif
    !
    tt(1)=nk1; tt(2)=nk2; tt(3)=nk3; tt(4)=nkirr
    !
  endif
  !
  CALL para_sync_int(tt, 4)
  nk1=tt(1); nk2=tt(2); nk3=tt(3); nkirr=tt(4)
  !
  allocate(kwt(nkirr), kvec(3, nkirr))
  !
  if (inode.eq.0) then
    if (iik.eq.0) then
      do ik1=0, nk1-1
        do ik2=0, nk2-1
          do ik3=0, nk3-1
            iik=ik1*nk2*nk3+ik2*nk3+ik3+1
            kwt(iik)=1.d0
            kvec(1, iik)=ik1*1.d0/nk1
            kvec(2, iik)=ik2*1.d0/nk2
            kvec(3, iik)=ik3*1.d0/nk3
          enddo
        enddo
      enddo
    else
      ! IBZKPT form
      do iik=1, nkirr
        read(fin, *) kvec(:, iik), ik1
        kwt(iik)=ik1*1.d0
      enddo
    endif
    close(unit=fin)
  endif
  !
  call para_sync_real(kwt, nkirr)
  call para_sync_real(kvec, nkirr*3)
  kwt(:)=kwt(:)/SUM(kwt(:))
end subroutine
```

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

**源代码**:
```fortran
subroutine read_impfile(fn)
  integer, dimension(2) :: tt
  integer ii, jj, kk
  real(dp), dimension(:), allocatable :: aa
  character(len=256)    :: key
  real(dp), dimension(3, 3) :: locrot
  !
  if (inode.eq.0) then
    open(unit=fin, file=trim(fn))
    read(fin, *) nimp
  endif
  !
  call para_sync0(nimp)
  allocate(imp(nimp))
  !
  do ii=1, nimp
    if (inode.eq.0) then
      read(fin, *) key       !  IMP idx
      read(fin, *) tt        !  ndim,  l
    endif
    call para_sync_int(tt, 2)
    call init_simp(imp(ii), tt(1), tt(2))
    !
    if (inode.eq.0) then
      read(fin, *) key       ! BasisMap
      if (trim(key)/='BasisMap') stop
      read(fin, *) imp(ii)%gidx(:)
      read(fin, *) key       ! Sigidx
      if (trim(key)/='Sigidx') stop
      do jj=1, imp(ii)%ndim
        read(fin, *) imp(ii)%sigidx(jj, :)
      enddo
      read(fin, *) key       ! Transformation Matrix or Local Rotation
      if (trim(key)=='Transformation') then
        allocate(aa(2*imp(ii)%ndim))
        do jj=1, imp(ii)%ndim
          read(fin, *) aa
          do kk=1, imp(ii)%ndim
            imp(ii)%Utrans(jj, kk)=aa(kk*2-1)+aa(kk*2)*cmplx_i
          enddo
        enddo
        deallocate(aa)
      else
        do jj=1, 3
          read(fin, *) locrot(jj, :)
        enddo
        call find_Utrans_from_locrot(imp(ii)%Utrans, imp(ii)%lang, &
                                      imp(ii)%ndim, locrot)
      endif
    endif
    !
    call para_sync_int(imp(ii)%gidx, imp(ii)%ndim)
    call para_sync_int(imp(ii)%sigidx, imp(ii)%ndim*imp(ii)%ndim)
    call para_sync_cmplx(imp(ii)%Utrans, imp(ii)%ndim*imp(ii)%ndim)
    !
  enddo
  if (inode.eq.0) close(unit=fin)
end subroutine
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

**源代码**:
```fortran
  allocate(partition(ham%norb))
  !
  ndimf=0
  ndimc=0
  !
  partition(:)=0
  !
  do ii=1, nimp
    ndimf=ndimf+imp(ii)%ndim
    partition(imp(ii)%gidx(:))=ii
  enddo
  ndimc=ham%norb-ndimf
  !
  allocate(f2g_idx(ndimf))
  allocate(c2g_idx(ndimc))
  !
  allocate(g2f_idx(ham%norb))
  allocate(g2c_idx(ham%norb))
  !
  g2f_idx=0
  g2c_idx=0
  !
  jj=1
  kk=1
  do ii=1, ham%norb
    !
    if (partition(ii)>0) then
      f2g_idx(jj)=ii
      g2f_idx(ii)=jj
      jj=jj+1
    else
      c2g_idx(kk)=ii
      g2c_idx(ii)=kk
      kk=kk+1
    endif
    !
  enddo
  !
  if (jj.ne.(ndimf+1) .or. kk.ne.(ndimc+1)) then
    write(*, *) "!!! FATAL: Incorrect F/C partition!"
    stop
  endif
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

**源代码**:
```fortran
subroutine fix_sigma_static
  !
  complex(dp), dimension(ham%norb, ham%norb) :: sigmat
  !
  call restore_lattice(sigmat, sinf)
  ham%hr(:, :, ham%r000)=ham%hr(:, :, ham%r000)+sigmat
  !
  if (inode.eq.0) then
    write(stdout, *) " Sinf matrix expands to:"
    call print_impurity(sigmat)
  endif
  !
end subroutine
```

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

**源代码**:
```fortran
subroutine get_sigma_matrix(sigfull, z)
  !
  complex(dp), dimension(ham%norb, ham%norb) :: sigfull
  complex(dp)  :: z
  !
  complex(dp), dimension(nbath)  :: sigma
  !
  call interpolate_single_sigma(sigma, z)
  call restore_lattice(sigfull, sigma)
  !
end subroutine
```

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

**源代码**:
```fortran
subroutine interpolate_single_sigma(sig, z)
  !
  complex(dp), dimension(nbath) :: sig
  complex(dp) :: z
  !
  complex(dp), dimension(nbath) :: sigtmp
  real(dp) :: w  ! The pure imaginary part of z
  !
  real(dp) :: ff
  real(dp) :: a, b, c, d
  integer ii
  !
  w=aimag(z)
  beta=(2*nw-1.d0)*twopi/(2.d0*aimag(omega(nw)))
  ff=(abs(w)*beta/twopi+0.5d0)
  ii=nint(ff)
  if (abs(ff-ii)<eps6 .and. ii<=nw) then
    ! Exactly on the mesh
    sigtmp(:)=sigpack(:, ii)
  else if (ff>nw) then
    ! Out of the mesh - use high-frequency tail
    do ii=1, nbath
      w1=aimag(omega(nw-10)); w2=aimag(omega(nw))
      s11=real(sigpack(ii, nw-10)); s12=aimag(sigpack(ii, nw-10))
      s21=real(sigpack(ii, nw)); s22=aimag(sigpack(ii, nw))
      a=(s21-s11)/(1.d0/(w2*w2)-1.d0/(w1*w1))
      d=(s21*w2*w2-s11*w1*w1)/(w2*w2-w1*w1)
      if (abs(s22)<eps6 .or. abs(s12)<eps6) then
        b=0.d0; c=0.d0
      else
        b=-(w2*s22-w1*s12)/(s22/w2-s12/w1)
        c=(w2*w2-w1*w1)/(w2/s22-w1/s12)
      endif
      sigtmp(ii)=d+a/(w*w)+cmplx_i*c*abs(w)/(w*w+b)
    enddo
  else if (ff>ii .and. ii<nw) then
    ! Between ii and ii+1 - linear interpolation
    sigtmp(:)=(ii+1-ff)*sigpack(:, ii)+(ff-ii)*sigpack(:, ii+1)
  else if (ff<ii .and. ii>0) then
    ! Between ii and ii-1 - linear interpolation
    sigtmp(:)=(ii-ff)*sigpack(:, ii-1)+(ff-ii+1)*sigpack(:, ii)
  else
    write(*, *)  '!!! FATAL: Incorrect matsubara frequency! Are you sure?'
    stop
  endif
  !
  if (w>0) then
    sig(:)=sigtmp(:)
  else
    sig(:)=conjg(sigtmp(:))
  endif
  !
end subroutine
```

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

**源代码**:
```fortran
  if (inode.eq.0) then
    !
    open(unit=fin, file="RPA.inp")
    read(fin, *) nffblk
    allocate(blkdim(nffblk))
    read(fin, *) blkdim
    !
    nFFidx=0
    nCCidx=ham%norb
    !
    do ii=1, nffblk
      nFFidx=nFFidx+blkdim(ii)*blkdim(ii)
      nCCidx=nCCidx-blkdim(ii)
    enddo
    !
  endif
  !
  mapping=0
  !
  call para_sync_int0(nFFidx)
  call para_sync_int0(nCCidx)
  !
  if (nCCidx<0) then
    write(*, *) " !!! Incorrect RPA division!"
    stop
  endif
  !
  allocate(FFidx(2, nFFidx))
  if (nCCidx>0) allocate(CCidx(nCCidx))
  !
  if (inode.eq.0) then
    !
    jj=1
    do ii=1, nffblk
      !
      allocate(blkidx(blkdim(ii)))
      !
      read(fin, *) blkidx(:)
      mapping(blkidx(:))=ii
      !
      do j1=1, blkdim(ii)
        do j2=1, blkdim(ii)
          FFidx(1, jj)=blkidx(j1)
          FFidx(2, jj)=blkidx(j2)
          jj=jj+1
        enddo
      enddo
      !
      deallocate(blkidx)
      !
    enddo
    !
    jj=1
    do ii=1, ham%norb
      !
      if (mapping(ii).eq.0) then
        CCidx(jj)=ii
        jj=jj+1
      endif
      !
    enddo
    !
    read(fin, *) nUcp
    !
  endif
  !
  call para_sync_int(FFidx, nFFidx*2)
  if (nCCidx>0) call para_sync_int(CCidx, nCCidx)
  call para_sync_int0(nUcp)
  !
  if (nUcp.ne.0) then
    !
    allocate(Uint_cp(nUcp), idxUcp(2, nUcp))
    !
    if (inode.eq.0) then
      !
      do ii=1, nUcp
        !
        read(fin, *) i1, i2, j1, j2, Uint_cp(ii)
        call find_ffidx(jj, i1, i2)
        idxUcp(1, ii)=jj
        call find_ffidx(jj, j1, j2)
        idxUcp(2, ii)=jj
        !
      enddo
      !
    endif
    !
    call para_sync_int(idxUcp, nUcp*2)
    call para_sync_real(Uint_cp, nUcp)
    !
  endif
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

**源代码** (来自 `modules/intRPA.f90:193-226`):
```fortran
do iw=1, nw
  !
  Dff=cmplx_0
  !
  do ii=1, nFFidx
    Dff(ii, ii)=cmplx_1
  enddo
  !
  ! D_{FF}=(1-\chi^0_{FF}*U_{FF})^{-1}
  ! Vff=Uff*Dff
  !
  call matmulsparse(Dff, chi0ff(:, :, iw), Uint_cp, idxUcp, nFFidx, nUcp, -1.d0, 1.d0)
  call invmat(Dff, nFFidx)
  call sparsemulmat(Vff, Uint_cp,           Dff,     idxUcp, nFFidx, nUcp,  1.d0, 0.d0)
  !
  ! chiff=Dff*chi0ff
  !
  call zgemm('N', 'N', nFFidx, nFFidx, nFFidx, cmplx_1, Dff, nFFidx, chi0ff(:, :, iw), nFFidx, cmplx_0, chiff(:, :, iw), nFFidx)
  !
  if ((.not. ff_only).and.(nCCidx>0)) then
    !
    ! tmpcf=chi0cf*Vff
    ! chifc=Dff*chi0fc
    ! chicf=chi0cf+tmpcf*chi0ff
    ! chicc=chi0cc+tmpcf*chi0fc
    !
    call zgemm('N', 'N', nCCidx, nFFidx, nFFidx, cmplx_1, chi0cf(:, :, iw), nCCidx, Vff,              nFFidx, cmplx_0, tmpcf,           nCCidx)
    call zgemm('N', 'N', nFFidx, nCCidx, nFFidx, cmplx_1, Dff,              nFFidx, chi0fc(:, :, iw), nFFidx, cmplx_0, chifc(:, :, iw), nFFidx)
    call zgemm('N', 'N', nCCidx, nFFidx, nFFidx, cmplx_1, tmpcf,            nCCidx, chi0ff(:, :, iw), nFFidx, cmplx_1, chicf(:, :, iw), nCCidx)
    call zgemm('N', 'N', nCCidx, nCCidx, nFFidx, cmplx_1, tmpcf,            nCCidx, chi0fc(:, :, iw), nFFidx, cmplx_1, chicc(:, :, iw), nCCidx)
    !
  endif
  !
enddo ! iw
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

**源代码**:
```fortran
SUBROUTINE calc_velocity(v_alpha, ham, kvec, alpha)
  TYPE(wannham), intent(in) :: ham
  real(dp), dimension(3), intent(in) :: kvec
  integer, intent(in) :: alpha
  complex(dp), dimension(ham%norb, ham%norb), intent(out) :: v_alpha
  integer :: ir, io, jo
  real(dp) :: rdotk, ktau, rtilde_alpha
  complex(dp) :: fact, orbfac
  complex(dp), dimension(ham%norb) :: phase
  !
  do io = 1, ham%norb
    ktau = sum(kvec(:) * ham%tau(:, io)) * twopi
    phase(io) = cmplx(cos(ktau), sin(ktau), KIND=dp)
  enddo
  !
  v_alpha(:,:) = cmplx_0
  !
  do ir = 1, ham%nrpt
    rdotk = sum(kvec(:) * ham%rvec(:, ir)) * twopi
    fact  = cmplx(cos(rdotk), sin(rdotk), KIND=dp) / ham%weight(ir)
    do io = 1, ham%norb
      do jo = 1, ham%norb
        rtilde_alpha = ham%rvec(alpha, ir) + ham%tau(alpha, jo) - ham%tau(alpha, io)
        orbfac = cmplx_i * twopi * rtilde_alpha * fact * &
                 conjg(phase(io)) * phase(jo)
        v_alpha(io, jo) = v_alpha(io, jo) + orbfac * ham%hr(io, jo, ir)
      enddo
    enddo
  enddo
END SUBROUTINE
```

---

##### 1.10.2 `calc_berry_curvature(omega_n, eigvec, vx, vy, eig, norb)`

**功能**: 从本征态和速度矩阵计算贝里曲率

**公式**:
$$\Omega_n^{xy}(\mathbf{k}) = -2 \Im \sum_{m \neq n} \frac{Vx_{nm} \cdot Vy_{mn}}{(E_n - E_m)^2}$$

其中 $Vx_{nm} = \langle n|v_x|m \rangle$ 在本征基底下计算。

**源代码**:
```fortran
SUBROUTINE calc_berry_curvature(omega_n, eigvec, vx, vy, eig, norb)
  integer, intent(in) :: norb
  complex(dp), dimension(norb, norb), intent(in) :: eigvec, vx, vy
  real(dp), dimension(norb), intent(in) :: eig
  real(dp), dimension(norb), intent(out) :: omega_n
  complex(dp), dimension(norb, norb) :: vx_band, vy_band, tmp
  complex(dp) :: numer
  real(dp) :: dE2
  integer :: n, m
  complex(dp), parameter :: zone  = cmplx(1.0_dp, 0.0_dp, KIND=dp)
  complex(dp), parameter :: zzero = cmplx(0.0_dp, 0.0_dp, KIND=dp)
  !
  call zgemm('N', 'N', norb, norb, norb, zone, vx, norb, eigvec, norb, zzero, tmp, norb)
  call zgemm('C', 'N', norb, norb, norb, zone, eigvec, norb, tmp, norb, zzero, vx_band, norb)
  call zgemm('N', 'N', norb, norb, norb, zone, vy, norb, eigvec, norb, zzero, tmp, norb)
  call zgemm('C', 'N', norb, norb, norb, zone, eigvec, norb, tmp, norb, zzero, vy_band, norb)
  !
  omega_n(:) = 0.0_dp
  do n = 1, norb
    do m = 1, norb
      if (m == n) cycle
      dE2 = (eig(n) - eig(m))**2
      if (dE2 < eps6*eps6) cycle
      numer = vx_band(n, m) * vy_band(m, n)
      omega_n(n) = omega_n(n) - 2.0_dp * aimag(numer) / dE2
    enddo
  enddo
END SUBROUTINE
```

---

##### 1.10.3 `fermi_func(f, eig, mu, temperature, norb)`

**功能**: 计算费米-狄拉克分布

**公式**:
$$f(E) = \frac{1}{\exp((E - \mu) / T) + 1}$$

**源代码**:
```fortran
SUBROUTINE fermi_func(f, eig, mu, temperature, norb)
  integer, intent(in) :: norb
  real(dp), dimension(norb), intent(in) :: eig
  real(dp), intent(in) :: mu, temperature
  real(dp), dimension(norb), intent(out) :: f
  integer :: ii
  real(dp) :: x
  do ii = 1, norb
    if (temperature < eps6) then
      f(ii) = merge(1.0_dp, 0.0_dp, eig(ii) <= mu)
    else
      x = (eig(ii) - mu) / temperature
      if (x > 500.0_dp) then
        f(ii) = 0.0_dp
      elseif (x < -500.0_dp) then
        f(ii) = 1.0_dp
      else
        f(ii) = 1.0_dp / (exp(x) + 1.0_dp)
      endif
    endif
  enddo
END SUBROUTINE
```

---

##### 1.10.4 `calc_sigma_xy(sigma_xy, ham, kvec_all, kwt_all, nk, mu_chem, temperature)`

**功能**: 计算反常霍尔电导率 σ_xy

**公式**:
$$\sigma_{xy} = -\frac{e^2}{h} \cdot \frac{1}{N_k} \sum_{\mathbf{k}} \sum_n f_n(\mathbf{k}) \Omega_n^{xy}(\mathbf{k})$$

**源代码** (k-loop core):
```fortran
do ik = first_idx, last_idx
  call calc_hk(hk, ham, kvec_all(:, ik))
  call eigen(eig, hk, norb)
  call calc_velocity(vx, ham, kvec_all(:, ik), 1)
  call calc_velocity(vy, ham, kvec_all(:, ik), 2)
  call calc_berry_curvature(omega_n, hk, vx, vy, eig, norb)
  call fermi_func(f_occ, eig, mu_chem, temperature, norb)
  sigma_acc = sigma_acc + sum(f_occ * omega_n) * kwt_all(ik)
enddo
!
call para_merge_real0(sigma_acc)
sigma_xy = -sigma_acc / (twopi * twopi)
```

---

##### 1.10.5 `calc_sigma_xx(sigma_xx, ham, kvec_all, kwt_all, nk, mu_chem, temperature, broadening)`

**功能**: 通过 Kubo-Greenwood 公式计算纵向直流电导率

**公式**:
$$\sigma_{xx} = \frac{1}{N_k} \sum_{\mathbf{k}} \text{Tr}[v_x \cdot G(\mathbf{k}, \mu+i\eta) \cdot v_x \cdot G(\mathbf{k}, \mu+i\eta)]$$

其中 $G(\mathbf{k}, z) = (z - H(\mathbf{k}))^{-1}$ 是推迟格林函数。

**源代码** (k-loop core):
```fortran
w_cmplx = cmplx(mu_chem, broadening, KIND=dp)
sigma_acc = cmplx_0
!
do ik = first_idx, last_idx
  call calc_hk(hk, ham, kvec_all(:, ik))
  call calc_g0(gf, hk, w_cmplx, norb, .false.)
  call calc_velocity(vx, ham, kvec_all(:, ik), 1)
  call zgemm('N', 'N', norb, norb, norb, zone, vx, norb, gf, norb, zzero, tmp1, norb)
  call zgemm('N', 'N', norb, norb, norb, zone, tmp1, norb, vx, norb, zzero, tmp2, norb)
  call zgemm('N', 'N', norb, norb, norb, zone, tmp2, norb, gf, norb, zzero, tmp1, norb)
  sigma_acc = sigma_acc + sum([(tmp1(ii, ii), ii=1,norb)]) * kwt_all(ik)
enddo
!
call para_merge_cmplx0(sigma_acc)
sigma_xx = -aimag(sigma_acc) / (twopi * twopi)
```

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

**源代码**:
```fortran
SUBROUTINE gp_update(gp, x_new, y_new)
  TYPE(gp_model), intent(inout) :: gp
  real(dp), dimension(gp%n_params), intent(in) :: x_new
  real(dp), intent(in) :: y_new
  !
  integer :: n_train_new, ii, jj
  real(dp), dimension(gp%n_train+1) :: k_star
  real(dp), dimension(gp%n_train+1, gp%n_train+1) :: K_new
  !
  n_train_new = gp%n_train + 1
  !
  ! Extend training set
  gp%x_train(:, n_train_new) = x_new
  gp%y_train(n_train_new) = y_new
  !
  ! Compute new kernel matrix
  do ii = 1, n_train_new
    do jj = 1, n_train_new
      K_new(ii, jj) = gp%kernel(gp%x_train(:, ii), gp%x_train(:, jj), &
                               gp%ls, gp%sv)
    enddo
  enddo
  !
  ! Add noise to diagonal
  K_new(n_train_new, n_train_new) = K_new(n_train_new, n_train_new) + gp%nv
  !
  ! Invert new matrix
  call invmat(K_new, n_train_new)
  !
  gp%K_inv = K_new
  gp%alpha = matmul(K_new, gp%y_train(1:n_train_new))
  gp%n_train = n_train_new
  !
END SUBROUTINE
```

---

##### 1.11.4 `gp_predict(gp, x_test, mu_out, sigma_out)`

**功能**: 在未测点预测均值和方差

**源代码**:
```fortran
SUBROUTINE gp_predict(gp, x_test, mu_out, sigma_out)
  TYPE(gp_model), intent(inout) :: gp
  real(dp), dimension(gp%n_params), intent(in) :: x_test
  real(dp), intent(out) :: mu_out, sigma_out
  !
  real(dp), dimension(gp%n_train) :: k_star
  real(dp) :: k_ss, var_temp
  integer :: ii
  !
  ! k*(x_test) = [k(x_test, x1), k(x_test, x2), ..., k(x_test, x_n)]
  do ii = 1, gp%n_train
    k_star(ii) = gp%kernel(x_test, gp%x_train(:, ii), gp%ls, gp%sv)
  enddo
  !
  ! Predictive mean: mu = k* . K^{-1} . y
  mu_out = dot_product(k_star, gp%alpha(1:gp%n_train))
  !
  ! Predictive variance: sigma^2 = k(x_test, x_test) - k* . K^{-1} . k*
  k_ss = gp%kernel(x_test, x_test, gp%ls, gp%sv)
  var_temp = dot_product(k_star, matmul(gp%K_inv(1:gp%n_train, 1:gp%n_train), k_star))
  sigma_out = max(k_ss - var_temp, 0.0_dp)
  !
END SUBROUTINE
```

---

##### 1.11.5 `expected_improvement(mu, sigma, y_best, xi, ei)`

**功能**: 期望改进获取函数

**源代码**:
```fortran
SUBROUTINE expected_improvement(mu, sigma, y_best, xi, ei)
  real(dp), intent(in) :: mu, sigma, y_best, xi
  real(dp), intent(out) :: ei
  !
  real(dp) :: diff, z, norm_pdf, norm_cdf
  !
  diff = y_best - mu - xi
  !
  if (sigma < eps6) then
    ei = 0.0_dp
    return
  endif
  !
  z = diff / sigma
  norm_pdf = exp(-0.5_dp * z * z) / sqrt(2.0_dp * 3.14159265359_dp)
  norm_cdf = 0.5_dp * (1.0_dp + erf(z / sqrt(2.0_dp)))
  !
  ei = diff * norm_cdf + sigma * norm_pdf
  ei = max(ei, 0.0_dp)
  !
END SUBROUTINE
```

---

##### 1.11.6 `latin_hypercube(samples, n_samples, n_params, bounds)`

**功能**: 生成拉丁超立方样本用于初始 GP 训练

**源代码**:
```fortran
SUBROUTINE latin_hypercube(samples, n_samples, n_params, bounds)
  real(dp), intent(out) :: samples(n_samples, n_params)
  integer, intent(in) :: n_samples, n_params
  real(dp), intent(in) :: bounds(n_params, 2)
  !
  integer :: i, j
  real(dp) :: u, span
  !
  do i = 1, n_params
    span = bounds(i, 2) - bounds(i, 1)
    do j = 1, n_samples
      u = (j - 1.0_dp + rand()) / n_samples
      samples(j, i) = bounds(i, 1) + u * span
    enddo
  enddo
  !
END SUBROUTINE
```

---

##### 1.11.7 `gp_optimize_ls(gp, bounds, n_params)`

**功能**: 通过梯度下降优化长度尺度

**源代码**:
```fortran
SUBROUTINE gp_optimize_ls(gp, bounds, n_params)
  TYPE(gp_model), intent(inout) :: gp
  real(dp), intent(in) :: bounds(n_params, 2)
  integer, intent(in) :: n_params
  !
  integer :: iter, max_iter_ls
  real(dp) :: lr, diff, ls_old
  max_iter_ls = 50
  lr = 0.1_dp
  !
  ls_old = gp%ls
  do iter = 1, max_iter_ls
    call gp_log_marginal_likelihood(gp, n_params, gp%lml)
    ls_old = gp%ls
    gp%ls = gp%ls * (1.0_dp + lr * (0.5_dp - gp%lml))
    gp%ls = max(gp%ls, bounds(1, 1))
    gp%ls = min(gp%ls, bounds(1, 2))
    diff = abs(gp%ls - ls_old)
    if (diff < 1e-4_dp) exit
  enddo
  !
END SUBROUTINE
```

---

##### 1.11.8 `bayesian_optimize(result, n_params, bounds, n_iter, n_init)`

**功能**: 主贝叶斯优化入口

**源代码**:
```fortran
SUBROUTINE bayesian_optimize(result, n_params, bounds, n_iter, n_init)
  real(dp), intent(out) :: result(n_params)
  integer, intent(in) :: n_params, n_iter, n_init
  real(dp), intent(in) :: bounds(n_params, 2)
  !
  TYPE(gp_model) :: gp
  real(dp), dimension(n_params) :: x_new, x_best
  real(dp), dimension(n_iter + n_init) :: y_all
  real(dp) :: y_best, f_new, mu, sigma, ei
  integer :: iter, ii
  !
  call gp_init(gp, n_params, 1.0_dp, 1.0_dp, 1e-2_dp)
  !
  ! Latin Hypercube initialization
  call latin_hypercube(gp%x_train, n_init, n_params, bounds)
  do ii = 1, n_init
    call js_objective(gp%x_train(ii, :), n_params, y_all(ii))
  enddo
  gp%y_train(1:n_init) = y_all(1:n_init)
  gp%n_train = n_init
  !
  y_best = minval(y_all(1:n_init))
  x_best = gp%x_train(minloc(y_all(1:n_init), dim=1), :)
  !
  ! Main Bayesian optimization loop
  do iter = 1, n_iter
    call gp_optimize_ls(gp, bounds, n_params)
    !
    ! Find best EI candidate
    ei = -1.0_dp
    do ii = 1, n_cand
      call gp_predict(gp, cand(ii, :), mu, sigma)
      call expected_improvement(mu, sigma, y_best, xi, f_new)
      if (f_new > ei) then
        ei = f_new
        x_new = cand(ii, :)
      endif
    enddo
    !
    ! Evaluate objective at new point
    call js_objective(x_new, n_params, f_new)
    !
    ! Update GP
    call gp_update(gp, x_new, f_new)
    !
    if (f_new < y_best) then
      y_best = f_new
      x_best = x_new
    endif
    !
    if (inode == 0) write(stdout, '(A,I4,A,ES12.4)') &
      "BO iter ", iter, " f_best= ", y_best
  enddo
  !
  result = x_best
  !
END SUBROUTINE
```

---

### 1.12 cma_es — CMA-ES 优化器

**文件**: [`modules/cma_es.f90`](wannchi/modules/cma_es.f90) (344 行)

**功能**: 标准 CMA-ES — Hansen & Ostermeier, Evolutionary Computation 9(2), 159 (2001)

**核心算法**:
1. 采样: `x = m + σ * B * D * z`, z~N(0,I) via Box-Muller
2. 协方差更新: `(1-c1-cμ)*C + c1*pc*pcᵀ + cμ*Σᵢ wᵢ*yᵢ*yᵢᵀ`
3. 特征分解: `C = B * D² * Bᵀ` via LAPACK `dsyev`
4. 步长更新: `σ ← σ·exp((||p_s||/χN - 1) · c_s/damps)`

**适用性**:
- 高维问题 (n_params ≥ 20)
- 非凸、多峰目标函数
- 黑盒优化（无梯度）

**不适用于**:
- 低维问题 (n_params < 10)
- 离散/分类参数

**关键实现细节**:
- sigma 是标量（不是轴平行向量）
- B 矩阵存储特征向量，特征值 D = sqrt(eigenvalues)
- sigma_arg 限制在 ±2.0 以防爆炸
- best_individual 单独保存（不是通过 best_idx）
- `xmean` 代替 `mean` 避免与 Fortran intrinsic 冲突

#### 子程序详细说明

##### 1.12.1 `cmaes_optimize(objective_func, bounds, n_params, result, n_iter)`

**功能**: CMA-ES 优化器入口

**接口**: `subroutine cmaes_optimize(objective_func, bounds, n_params, result, n_iter)`
- `objective_func(params, n) -> real(dp)`: 外部目标函数（最小化）
- `bounds(2, n_params)`: 参数边界 [lower, upper]
- `n_iter`: 最大迭代次数
- `result(n_params)`: 最优解

**实现**: 见源文件 [`modules/cma_es.f90`](wannchi/modules/cma_es.f90)，主要流程:

```fortran
! 初始化
lambda = max(..., 4 + 3*log(n))   ! 群体大小
mu = lambda / 2                    ! 父代数量
cc = (4+mu_eff)/(n+4+2*mu_eff)    ! 协方差路径学习率
cs = (mu_eff+2)/(n+mu_eff+5)       ! 步长路径学习率
c1 = 2/((n+1.3)²+mu_eff)           ! rank-1 学习率
cmu = min(1-c1, ...)               ! rank-mu 学习率

! 主循环
do gen = 1, n_iter
  ! 采样: x = m + sigma * B * D * z
  call sample_population(x_pop, xmean, sigma, B, D, n_params, lambda)

  ! 分布式评估
  call distribute_calc(lambda)

  ! 排序 & 选择 top-mu
  call sort_by_fitness(x_pop, fitness, lambda)

  ! 计算 centered y_sel = (x_i - m) / sigma

  ! 加权均值 y_w = Σ w_i * y_sel_i

  ! 更新 pc (协方差路径)
  pc = (1-cc)*pc + sqrt(cc*(2-cc)*mu_eff) * y_w

  ! 更新 ps (步长路径)
  ps = (1-cs)*ps + sqrt(cs*(2-cs)*mu_eff) * B * D⁻¹ * y_w

  ! 更新协方差 C = (1-c1-cmu)*C + c1*pc*pcᵀ + cmu*Σ w_i*y_i*y_iᵀ
  ! via dger rank-1 updates

  ! 特征分解: C -> B, D (dsyev)
  call dsyev('V', 'U', n, C, n, D, work, lwork, info)

  ! 更新步长 sigma (overflow-protected)
  sigma_arg = ((norm(ps)/chiN - 1) * cs / damps)
  sigma = sigma * exp(clamp(sigma_arg, -2, +2))

  ! 更新均值
  xmean = xmean + sigma * pc

  ! 收敛检查
  if (f_best < TOL) exit
enddo
```

**关键参数** (Hansen 2019):
- `mu_eff = mu` (等权), `chiN = sqrt(n)*(1 - 1/(4n) + 1/(21n²))`
- `damps = 1 + max(0, sqrt(mu_eff)-1) + cc`

**已知修复** (2026-04-07):
- sigma 初始值 = 0.02（防止爆炸）
- sigma_arg 限制 ±2.0（sigma 最多变化 exp(2)≈7 倍/代）
- best_individual 单独数组保存
- `xmean` 避免与 Fortran intrinsic 冲突
- dsyev 需要 work/lwork 参数（macOS Accelerate 必需）

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

**源代码**:
```fortran
SUBROUTINE mc_init(mc, n_sites, J_mc_in, S_mag_in)
  TYPE(mc_lattice), intent(inout) :: mc
  integer, intent(in) :: n_sites
  real(dp), intent(in) :: J_mc_in, S_mag_in
  !
  integer :: ii
  !
  mc%n_sites         = n_sites
  mc%n_neighbors_max = 12  ! enough for FCC/HCP; will be trimmed by n_nn
  mc%J_mc            = J_mc_in
  mc%S_mag           = S_mag_in
  !
  allocate(mc%spin(3, n_sites))
  allocate(mc%neighbor_list(mc%n_neighbors_max, n_sites))
  allocate(mc%n_nn(n_sites))
  !
  mc%neighbor_list = 0
  mc%n_nn          = 0
  !
  ! Random initial spins on unit sphere
  do ii = 1, n_sites
    call mc_random_spin(mc%spin(:, ii))
  enddo
  !
END SUBROUTINE
```

##### 1.13.2 `mc_random_spin(spin)`

**功能**: 通过 Marsaglia 方法生成单位球面上均匀随机单位矢量

**源代码**:
```fortran
SUBROUTINE mc_random_spin(spin)
  !
  ! Generate a uniformly random unit vector on S^2 using the Marsaglia method.
  ! Marsaglia (1972): pick (u,v) uniform in unit disk, then
  !   s = u^2 + v^2
  !   spin = (2u*sqrt(1-s), 2v*sqrt(1-s), 1-2s)
  !
  real(dp), dimension(3), intent(out) :: spin
  !
  real(dp) :: u, v, s
  !
  do
    call random_number(u); u = 2.0_dp*u - 1.0_dp
    call random_number(v); v = 2.0_dp*v - 1.0_dp
    s = u*u + v*v
    if (s < 1.0_dp) exit
  enddo
  !
  spin(1) = 2.0_dp * u * sqrt(1.0_dp - s)
  spin(2) = 2.0_dp * v * sqrt(1.0_dp - s)
  spin(3) = 1.0_dp - 2.0_dp * s
  !
END SUBROUTINE
```

##### 1.13.3 `mc_build_neighbors(mc, frac_pos, n_uc_sites, avec, nx, ny, nz, cutoff)`

**功能**: 为超胞构建周期性邻居列表

**源代码**:
```fortran
SUBROUTINE mc_build_neighbors(mc, frac_pos, n_uc_sites, avec, nx, ny, nz, cutoff)
  !
  ! Build periodic neighbor list for a supercell of nx x ny x nz unit cells,
  ! each containing n_uc_sites f-sites at fractional positions frac_pos(3, n_uc_sites).
  !
  ! Site indexing: site (ix, iy, iz, isite) -> global index
  !   idx = (iz*ny*nx + iy*nx + ix) * n_uc_sites + isite
  !
  TYPE(mc_lattice), intent(inout) :: mc
  integer, intent(in)  :: n_uc_sites, nx, ny, nz
  real(dp), dimension(3, n_uc_sites), intent(in) :: frac_pos
  real(dp), dimension(3, 3), intent(in)  :: avec  ! lattice vectors (columns)
  real(dp), intent(in) :: cutoff  ! distance cutoff in Angstrom
  !
  integer :: ia, ib, ic, is, ja, jb, jc, js, idx_i, idx_j
  integer :: da, db, dc
  real(dp), dimension(3) :: ri_cart, rj_cart, diff
  real(dp) :: dist
  !
  mc%n_nn(:) = 0
  !
  do ic = 0, nz-1
    do ib = 0, ny-1
      do ia = 0, nx-1
        do is = 1, n_uc_sites
          !
          idx_i = ((ic*ny + ib)*nx + ia) * n_uc_sites + is
          ! Cartesian position of site i
          ri_cart = matmul(avec, frac_pos(:,is) + [real(ia,dp), real(ib,dp), real(ic,dp)])
          !
          ! Loop over neighboring unit cells (±2 shell in each direction)
          do dc = -2, 2
            do db = -2, 2
              do da = -2, 2
                do js = 1, n_uc_sites
                  !
                  ! PBC unit cell indices
                  jc = mod(ic + dc + 2*nz, nz)
                  jb = mod(ib + db + 2*ny, ny)
                  ja = mod(ia + da + 2*nx, nx)
                  !
                  idx_j = ((jc*ny + jb)*nx + ja) * n_uc_sites + js
                  !
                  if (idx_j == idx_i) cycle  ! skip self
                  !
                  rj_cart = matmul(avec, frac_pos(:,js) + &
                            [real(ja,dp), real(jb,dp), real(jc,dp)])
                  diff = ri_cart - rj_cart
                  dist = sqrt(sum(diff**2))
                  !
                  if (dist < cutoff + 1.0d-6) then
                    if (mc%n_nn(idx_i) < mc%n_neighbors_max) then
                      mc%n_nn(idx_i) = mc%n_nn(idx_i) + 1
                      mc%neighbor_list(mc%n_nn(idx_i), idx_i) = idx_j
                    endif
                  endif
                  !
                enddo
              enddo
            enddo
          enddo
          !
        enddo
      enddo
    enddo
  enddo
  !
END SUBROUTINE
```

##### 1.13.4 `mc_sweep(mc, temperature, n_accepted)`

**功能**: 一次 MC sweep = n_sites 次单自旋更新尝试

**源代码**:
```fortran
SUBROUTINE mc_sweep(mc, temperature, n_accepted)
  !
  ! Single-spin Metropolis sweep: n_sites attempted updates.
  ! temperature in eV.
  !
  TYPE(mc_lattice), intent(inout) :: mc
  real(dp), intent(in)  :: temperature
  integer,  intent(out) :: n_accepted
  !
  integer :: ii, isite, jj, jsite
  real(dp) :: dE, u
  real(dp), dimension(3) :: spin_old, spin_new
  real(dp) :: sdot_old, sdot_new
  !
  n_accepted = 0
  !
  do ii = 1, mc%n_sites
    !
    call random_number(u)
    isite = int(u * mc%n_sites) + 1
    if (isite > mc%n_sites) isite = mc%n_sites
    !
    spin_old = mc%spin(:, isite)
    call mc_random_spin(spin_new)
    !
    ! Compute dE from neighbor interactions only:
    ! dE = -J*S^2 * sum_{j in NN} (S_new - S_old) . S_j
    dE = 0.0_dp
    do jj = 1, mc%n_nn(isite)
      jsite = mc%neighbor_list(jj, isite)
      sdot_old = dot_product(spin_old, mc%spin(:, jsite))
      sdot_new = dot_product(spin_new, mc%spin(:, jsite))
      dE = dE - mc%J_mc * mc%S_mag * mc%S_mag * (sdot_new - sdot_old)
    enddo
    !
    ! Metropolis acceptance
    if (dE <= 0.0_dp) then
      mc%spin(:, isite) = spin_new
      n_accepted = n_accepted + 1
    else if (temperature > 1.0d-12) then
      call random_number(u)
      if (u < exp(-dE / temperature)) then
        mc%spin(:, isite) = spin_new
        n_accepted = n_accepted + 1
      endif
    endif
    !
  enddo
  !
END SUBROUTINE
```

##### 1.13.5 `mc_measure_magnetization(mc, mvec_out)`

**功能**: 测量磁化强度

**公式**:
$$\mathbf{m} = \frac{1}{N} \sum_{i=1}^N \mathbf{S}_i$$

输出为分数磁化矢量（|mvec| ∈ [0,1]）。

---

##### 1.13.6 `classical_mc_run(J_mc_in, S_mag_in, frac_pos, n_f_sites, avec, T_start, T_step, T_end, mvec_vs_T, n_temps, mc_supercell_in)`

**功能**: 温度扫描主循环

**源代码**:
```fortran
SUBROUTINE classical_mc_run(J_mc_in, S_mag_in, frac_pos, n_f_sites, avec, &
                             T_start, T_step, T_end, mvec_vs_T, n_temps, &
                             mc_supercell_in)
  !
  ! Main MC driver: temperature sweep from T_start to T_end with step T_step.
  !
  ! Inputs:
  !   J_mc_in    : exchange coupling between f-sites (eV)
  !   S_mag_in   : local spin magnitude |S|
  !   frac_pos   : (3, n_f_sites) fractional coordinates of f-sites in unit cell
  !   n_f_sites  : number of f-sites per unit cell
  !   avec       : (3,3) lattice vectors (columns = a1,a2,a3)
  !   T_start, T_step, T_end : temperature range in eV
  !   mc_supercell_in(3) : (NX, NY, NZ) supercell. 0 = auto-detect from avec:
  !     if |a_i| > 2 * min(|a_j|, |a_k|), it is assumed vacuum -> N_i = 1
  !     otherwise N_i = 10 (standard thermodynamic limit for 2D/3D).
  !
  ! Output:
  !   mvec_vs_T(3, n_temps) : fractional magnetization vector at each temperature
  !   n_temps               : number of temperature points
  !
  integer, parameter :: N_DEFAULT = 10  ! default supercell per active direction
  integer, parameter :: N_THERM = 5000, N_MEAS = 10000, MEAS_EVERY = 10
  !
  TYPE(mc_lattice) :: mc
  integer :: n_total_sites, iT, imeas, n_acc, ii
  integer :: NX, NY, NZ
  real(dp) :: T_now, cutoff
  real(dp), dimension(3) :: mvec_tmp, mvec_acc
  real(dp) :: dist_nn, len_a1, len_a2, len_a3, len_min, u
  real(dp), dimension(3) :: r1_cart, r2_cart
  real(dp), allocatable :: mvec_local(:,:)
  integer :: n_local_temps, iT_start, iT_end
  !
  ! Determine supercell dimensions: user override or auto-detect from lattice
  len_a1 = sqrt(sum(avec(:,1)**2))
  len_a2 = sqrt(sum(avec(:,2)**2))
  len_a3 = sqrt(sum(avec(:,3)**2))
  !
  if (mc_supercell_in(1) > 0) then
    NX = mc_supercell_in(1)
  else
    len_min = min(len_a2, len_a3)
    NX = merge(1, N_DEFAULT, len_a1 > 2.0_dp * len_min)
  endif
  if (mc_supercell_in(2) > 0) then
    NY = mc_supercell_in(2)
  else
    len_min = min(len_a1, len_a3)
    NY = merge(1, N_DEFAULT, len_a2 > 2.0_dp * len_min)
  endif
  if (mc_supercell_in(3) > 0) then
    NZ = mc_supercell_in(3)
  else
    len_min = min(len_a1, len_a2)
    NZ = merge(1, N_DEFAULT, len_a3 > 2.0_dp * len_min)
  endif
  !
  ! Distribute temperature points across MPI ranks
  call distribute_calc(n_temps)
  iT_start = first_idx
  iT_end = last_idx
  n_local_temps = last_idx - first_idx + 1
  !
  do iT = iT_start, iT_end
    !
    if (n_temps == 1) then
      T_now = T_start
    else
      T_now = T_start + real(iT-1, dp) * T_step
    endif
    !
    if (T_now < 1.0d-12) then
      ! T=0: perfect ferromagnetic order along z
      do ii = 1, n_total_sites
        mc%spin(:, ii) = [0.0_dp, 0.0_dp, 1.0_dp]
      enddo
      mvec_local(:, iT - iT_start + 1) = [0.0_dp, 0.0_dp, 1.0_dp]
      cycle
    endif
    !
    ! Thermalize
    call mc_thermalize(mc, T_now, N_THERM)
    !
    ! Measure
    mvec_acc = 0.0_dp
    do imeas = 1, N_MEAS
      call mc_sweep(mc, T_now, n_acc)
      if (mod(imeas, MEAS_EVERY) == 0) then
        call mc_measure_magnetization(mc, mvec_tmp)
        mvec_acc = mvec_acc + mvec_tmp
      endif
    enddo
    mvec_local(:, iT - iT_start + 1) = mvec_acc / real(N_MEAS/MEAS_EVERY, dp)
    !
  enddo
  !
  ! Gather results to all ranks via para_merge_real
  call para_merge_real(mvec_local, 3 * n_local_temps)
  !
END SUBROUTINE
```

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

**结构**: `MODULE wanneff_js_mod` (第 38-509 行) + `PROGRAM WannEffJS` (第 514-1175 行)

**依赖**: `constants`, `wanndata`, `linalgwrap`, `gp_bo`, `cma_es`, `classical_mc`, `transp_calc`, `wannlog`, `lattice`, `input`, `para`

**功能**: 端到端流程：R 空间 Schur 补 downfolding → J·S 优化 → MC → 输运

#### MODULE wanneff_js_mod — 模块级全局变量

模块级指针/变量用于将数据传递给贝叶斯回调函数（因为优化器回调接口固定为 `f(params, n) -> real`）：

```fortran
integer :: g_norb_bare, g_norb_cc, g_nkirr, g_n_jrpt
complex(dp), allocatable :: g_hk_eff_cc(:,:,:)   ! (norb_cc, norb_cc, nkirr)
TYPE(wannham), pointer :: g_ham_bare => null()
real(dp), allocatable :: g_kvec(:,:)              ! (3, nkirr)
real(dp), allocatable :: g_rvec_J(:,:)            ! (3, n_jrpt)
integer :: g_eff_mode  ! 1=标量 J, 2=J_TENSOR, 3=J_S_TENSOR
```

#### 10 个模块子程序概览

| 编号 | 子程序 | 功能 |
|------|--------|------|
| 1.15.1 | `downfold_rspace` | R 空间 Schur 补 downfolding |
| 1.15.2 | `normalize_spin_direction` | 旋量方向归一化 |
| 1.15.3 | `reconstruct_hr_from_hk` | H(k) → H(R) 逆傅里叶变换 |
| 1.15.4 | `validate_hr_reconstruction` | 验证 H(R) 重构精度 |
| 1.15.5 | `init_periodic_reconstruction_ham` | 构建周期性 R 网格 |
| 1.15.6 | `add_js_coupling` | R 空间添加 J·(S·σ)/2 耦合 |
| 1.15.7 | `js_objective` | 贝叶斯/CMA-ES 目标函数 |
| 1.15.8 | `add_js_coupling_kspace` | k 空间标量 JS 耦合 |
| 1.15.9 | `add_js_coupling_tensor_kspace` | k 空间张量 JS 耦合 |
| 1.15.10 | `js_objective_callback` | 优化器回调接口包装 |

详细源码和 PROGRAM WannEffJS 工作流见 [第 2.9 节](#29-wanneff_jsf90--j-s-kondo-交换耦合拟合)。

---

## 2. 源文件详细子程序分析

### 2.1 input — 输入文件解析

**文件**: [`src/input.f90`](wannchi/src/input.f90) (452 行)

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

##### 2.1.4 `read_effjs_input(codename)`

**功能**: 读取 `&EFFJS` namelist（用于 `wanneff_JS.x`）

**接口**:
```fortran
SUBROUTINE read_effjs_input(codename)
    character(*), intent(in) :: codename
```

**新增 &EFFJS 参数**（第 54-83 行）:

```fortran
character(len=80) :: seedbare = ''
logical  :: eff_js = .false.
logical  :: eff_mc = .false.
real(dp) :: mc_temperature(3) = (/0.0_dp, 0.5_dp, 300.0_dp/)
logical  :: mc_weiss_mean_field = .true.
real(dp) :: J_mc = 0.0_dp
integer  :: eff_mode = 1        ! 1=scalar, 2=J_TENSOR, 3=J_S_TENSOR
real(dp) :: tol_Jeff = 1.0d-2
integer  :: J_R_range(6) = 0   ! Rx_min,Rx_max,Ry_min,Ry_max,Rz_min,Rz_max
integer  :: bayes_niter = 50
real(dp) :: J_bounds(2) = (/0.0_dp, 10.0_dp/)
real(dp) :: S_bounds(2) = (/-5.0_dp,  5.0_dp/)
integer  :: mc_supercell(3) = (/0, 0, 0/)
real(dp) :: sigma_broadening = 0.05_dp
logical  :: berry_curvature_output = .false.
integer  :: n_ff_orbital_indices = 0
integer  :: ff_orbital_indices(100) = 0
```

**输入文件格式** (wanneff.inp):
```fortran
&SYSTEM
    seed='kagome_f',
    mu=0.d0,
    beta=2000.d0,
/

&CONTROL
    use_lehman=.false.,
    trace_only=.false.,
    ff_only=.true.,
    fast_calc=.true.,
/

&EFFJS
    seedbare='kagome_bare',
    eff_js=.true.,
    eff_mode=3,
    J_TENSOR=.false.,
    mc_temperature=0.0, 0.5, 300.0,
    mc_supercell=0, 0, 0,
    tol_Jeff=1.0d-2,
    J_bounds=0.0, 10.0,
    S_bounds=-5.0, 5.0,
    sigma_broadening=0.05,
    bayes_niter=50,
    mc_weiss_mean_field=.true.,
    n_ff_orbital_indices=2,
    ff_orbital_indices=9, 10,
/
```

**实现细节**:
- 同时读取 `&SYSTEM` 和 `&CONTROL`（覆盖 wannchi 的默认值）
- 通过 `para_sync_character` 广播 `seedbare`（字符 → ASCII 码数组 → MPI broadcast）
- `eff_mode=1` → 标量 J（GP-BO）；`eff_mode=2` → J_TENSOR（GP-BO 或 CMA-ES）；`eff_mode=3` → J_S_TENSOR（CMA-ES）

---

##### 2.1.5 `para_sync_character(str)`

**功能**: 通过 MPI 广播 80 字符字符串（Fortran 字符不能直接广播）

**实现**:
```fortran
if (inode .eq. 0) then
    do ii = 1, 80
        codes(ii) = ichar(str(ii:ii))  ! 字符 → ASCII 码
    enddo
endif
call para_sync_int(codes, 80)           ! MPI broadcast
if (inode .ne. 0) then
    do ii = 1, 80
        str(ii:ii) = char(codes(ii))    ! ASCII 码 → 字符
    enddo
endif
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

**源代码**:
```fortran
SUBROUTINE calc_chi_bare_matrix_lehman_kernel(w, nw)
  integer nw
  complex(dp), dimension(nw) :: w
  integer ibnd, jbnd, ii, iw, i1, i2
  real(dp) :: occ_diff
  complex(dp), dimension(nw) :: fact
  complex(dp), dimension(nFFidx) :: UUf
  complex(dp), dimension(nCCidx) :: UUc
  !
  do ibnd=1, ham%norb
    do jbnd=1, ham%norb
      occ_diff=occ_k(ibnd)-occ_kq(jbnd)
      if (abs(occ_diff)>eps12) then
        fact(:)=occ_diff/(w(:)+ekq(jbnd)-ek(ibnd))
        do ii=1, nFFidx
          i1=FFidx(1,ii); i2=FFidx(2,ii)
          UUf(ii)=conjg(hk(i1,ibnd))*hkq(i2,jbnd)
        enddo
        do ii=1, nCCidx
          i1=CCidx(ii)
          UUc(ii)=conjg(hk(i1,ibnd))*hkq(i1,jbnd)
        enddo
        do iw=1, nw
          call zgerc(nFFidx,nFFidx,fact(iw),UUf,1,UUf,1,chiff(:,:,iw),nFFidx)
          if (nCCidx>0) then
            call zgerc(nFFidx,nCCidx,fact(iw),UUf,1,UUc,1,chifc(:,:,iw),nFFidx)
            call zgerc(nCCidx,nFFidx,fact(iw),UUc,1,UUf,1,chicf(:,:,iw),nCCidx)
            call zgerc(nCCidx,nCCidx,fact(iw),UUc,1,UUc,1,chicc(:,:,iw),nCCidx)
          endif
        enddo
      endif
    enddo
  enddo
END SUBROUTINE
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

**源代码**:
```fortran
SUBROUTINE calc_chi_bare_matrix_GG_kernel(w, nw)
  integer nw
  complex(dp), dimension(nw) :: w
  integer ibnd, jbnd, ii, iw, i1, i2
  complex(dp), dimension(nw) :: fact
  complex(dp) :: z1, z2, g1, g2
  complex(dp), dimension(nFFidx) :: UUf
  complex(dp), dimension(nCCidx) :: UUc
  !
  do ibnd=1, ham%norb
    do jbnd=1, ham%norb
      fact(:)=cmplx_0
      do ii=1, npole
        z1=zp(ii)/beta*cmplx_i
        g1=cmplx_1/(z1-ek(ibnd))
        do iw=1, nw
          z2=z1+w(iw); g2=cmplx_1/(z2-ekq(jbnd))
          fact(iw)=fact(iw)+eta(ii)*g1*g2
        enddo
        z1=-zp(ii)/beta*cmplx_i
        g1=cmplx_1/(z1-ek(ibnd))
        do iw=1, nw
          z2=z1+w(iw); g2=cmplx_1/(z2-ekq(jbnd))
          fact(iw)=fact(iw)+eta(ii)*g1*g2
        enddo
      enddo
      fact(:)=-fact(:)/beta
      do ii=1, nFFidx
        i1=FFidx(1,ii); i2=FFidx(2,ii)
        UUf(ii)=conjg(hk(i1,ibnd))*hkq(i2,jbnd)
      enddo
      do ii=1, nCCidx
        i1=CCidx(ii)
        UUc(ii)=conjg(hk(i1,ibnd))*hkq(i1,ibnd)
      enddo
      do iw=1, nw
        call zgerc(nFFidx,nFFidx,fact(iw),UUf,1,UUf,1,chiff(:,:,iw),nFFidx)
        if (nCCidx>0) then
          call zgerc(nFFidx,nCCidx,fact(iw),UUf,1,UUc,1,chifc(:,:,iw),nFFidx)
          call zgerc(nCCidx,nFFidx,fact(iw),UUc,1,UUf,1,chicf(:,:,iw),nCCidx)
          call zgerc(nCCidx,nCCidx,fact(iw),UUc,1,UUc,1,chicc(:,:,iw),nCCidx)
        endif
      enddo
    enddo
  enddo
END SUBROUTINE
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

**文件**: [`src/wanneff_JS.f90`](wannchi/src/wanneff_JS.f90) (1176 行)

**结构**: `MODULE wanneff_js_mod` (第 38-509 行) + `PROGRAM WannEffJS` (第 514-1175 行)

**依赖**: `constants`, `wanndata`, `linalgwrap`, `gp_bo`, `cma_es`, `classical_mc`, `transp_calc`, `wannlog`, `lattice`, `input`, `para`

#### MODULE wanneff_js_mod — 模块级全局变量

模块级指针/变量用于将数据传递给贝叶斯回调函数 `js_objective`（因为 `bayesian_optimize` 和 `cmaes_optimize` 的回调接口固定为 `f(params, n) -> real`）：

```fortran
integer :: g_norb_bare, g_norb_cc, g_nkirr, g_n_jrpt
complex(dp), allocatable :: g_hk_eff_cc(:,:,:)   ! (norb_cc, norb_cc, nkirr) — downfold 后的有效 H(k)
TYPE(wannham), pointer :: g_ham_bare => null()     ! 指向 seedbare Hamiltonian
real(dp), allocatable :: g_kvec(:,:)              ! (3, nkirr) — 不可约 k 点
real(dp), allocatable :: g_rvec_J(:,:)            ! (3, n_jrpt) — J(R) 的 R 格矢
integer :: g_eff_mode  ! 1=标量 J, 2=J_TENSOR, 3=J_S_TENSOR
```

#### 模块子程序详细说明

##### 2.9.1 `downfold_rspace(ham_eff_cc, ham_full, cc_idx, ff_idx, n_cc, n_ff)`

**功能**: R 空间舒尔补 downfolding：计算有效 CC Hamiltonian

**公式**:
$$H_{eff}^{CC}(\mathbf{R}) = H_{CC}(\mathbf{R}) - H_{CF}(\mathbf{R}) \cdot H_{FF}(\mathbf{R})^{-1} \cdot H_{FC}(\mathbf{R})$$

**源代码** (第 57-151 行):

```fortran
SUBROUTINE downfold_rspace(ham_eff_cc, ham_full, cc_idx, ff_idx, n_cc, n_ff)
  TYPE(wannham), intent(inout) :: ham_eff_cc
  TYPE(wannham), intent(in) :: ham_full
  integer, dimension(n_cc), intent(in) :: cc_idx
  integer, dimension(n_ff), intent(in) :: ff_idx
  integer, intent(in) :: n_cc, n_ff
  !
  complex(dp), dimension(n_cc, n_cc) :: H_cc
  complex(dp), dimension(n_cc, n_ff) :: H_cf
  complex(dp), dimension(n_ff, n_cc) :: H_fc
  complex(dp), dimension(n_ff, n_ff) :: H_ff, H_ff_inv
  complex(dp), dimension(n_cc, n_ff) :: tmp
  integer :: ir, ii, jj
  ! Copy structure from ham_full to ham_eff_cc
  ham_eff_cc%norb = n_cc
  ham_eff_cc%nrpt = ham_full%nrpt
  ham_eff_cc%rvec = ham_full%rvec
  ham_eff_cc%weight = ham_full%weight
  allocate(ham_eff_cc%tau(3, n_cc))
  ham_eff_cc%tau = ham_full%tau(:, cc_idx)  ! Only CC orbitals
  ham_eff_cc%r000 = ham_full%r000
  allocate(ham_eff_cc%hr(n_cc, n_cc, ham_full%nrpt))
  !
  do ir = 1, ham_full%nrpt
    ! Extract blocks from full HR at this R
    do ii = 1, n_cc
      do jj = 1, n_cc
        H_cc(ii, jj) = ham_full%hr(cc_idx(ii), cc_idx(jj), ir)
      end do
    end do
    do ii = 1, n_cc
      do jj = 1, n_ff
        H_cf(ii, jj) = ham_full%hr(cc_idx(ii), ff_idx(jj), ir)
      end do
    end do
    do ii = 1, n_ff
      do jj = 1, n_cc
        H_fc(ii, jj) = ham_full%hr(ff_idx(ii), cc_idx(jj), ir)
      end do
    end do
    do ii = 1, n_ff
      do jj = 1, n_ff
        H_ff(ii, jj) = ham_full%hr(ff_idx(ii), ff_idx(jj), ir)
      end do
    end do
    !
    if (maxval(abs(H_ff)) < eps6) then
      ! H_ff is zero: no FF propagation at this R
    else
      H_ff_inv = H_ff
      call invmat(H_ff_inv, n_ff)
      ! tmp = H_cf * H_ff_inv
      call zgemm('N', 'N', n_cc, n_ff, n_ff, cmplx_1, H_cf, n_cc, &
                  H_ff_inv, n_ff, cmplx_0, tmp, n_cc)
      ! H_eff_cc = H_cc - tmp * H_fc
      call zgemm('N', 'N', n_cc, n_cc, n_ff, -cmplx_1, tmp, n_cc, &
                  H_fc, n_ff, cmplx_1, H_cc, n_cc)
    endif
    !
    ham_eff_cc%hr(:, :, ir) = H_cc
  end do
END SUBROUTINE downfold_rspace
```

**算法说明**:
1. 从 `ham_full` 复制结构（rvec, weight, r000），tau 仅取 CC 轨道
2. 对每个 R 格点：提取 4 个块矩阵 H_CC, H_CF, H_FC, H_FF
3. 若 H_FF ≈ 0（无 FF 跳跃）→ H_eff = H_CC（舒尔补项为零）
4. 否则：求逆 H_FF 并计算舒尔补项 H_cc -= H_CF * H_FF^{-1} * H_FC

---

##### 2.9.2 `normalize_spin_direction(raw_vec, unit_vec, raw_norm)`

**功能**: 将旋量方向归一化为单位向量，消除 J/|S| 标量冗余

**源代码** (第 154-169 行):

```fortran
SUBROUTINE normalize_spin_direction(raw_vec, unit_vec, raw_norm)
  real(dp), dimension(3), intent(in)  :: raw_vec
  real(dp), dimension(3), intent(out) :: unit_vec
  real(dp),               intent(out) :: raw_norm
  !
  raw_norm = sqrt(sum(raw_vec**2))
  if (raw_norm > eps6) then
    unit_vec = raw_vec / raw_norm
  else
    unit_vec = (/0.0_dp, 0.0_dp, 1.0_dp/)
  endif
END SUBROUTINE normalize_spin_direction
```

---

##### 2.9.3 `reconstruct_hr_from_hk(ham_hr, hk_k, nk, kmesh, kw)`

**功能**: 逆傅里叶变换，从 H(k) 重构 H(R)（与 `calc_hk` 的正变换一致）

**逆变换公式**:
$$H_{ij}(\mathbf{R}) = w_R \cdot \frac{1}{\sum_k w_k} \sum_{\mathbf{k}} w_k \cdot e^{-i\mathbf{k}\cdot\mathbf{R}} \cdot e^{+i\mathbf{k}\cdot\boldsymbol{\tau}_i} \cdot e^{-i\mathbf{k}\cdot\boldsymbol{\tau}_j} \cdot H_{ij}(\mathbf{k})$$

**源代码** (第 172-220 行):

```fortran
SUBROUTINE reconstruct_hr_from_hk(ham_hr, hk_k, nk, kmesh, kw)
  TYPE(wannham), intent(inout) :: ham_hr
  integer, intent(in) :: nk
  complex(dp), dimension(:, :, :), intent(in) :: hk_k
  real(dp),    dimension(:, :), intent(in) :: kmesh
  real(dp),    dimension(:),    intent(in) :: kw
  !
  integer :: ir, ik, io, jo
  real(dp) :: rdotk, ktau, sum_kwt
  complex(dp) :: rphase
  complex(dp), allocatable :: orb_phase(:)
  !
  sum_kwt = sum(kw(1:nk))
  allocate(orb_phase(ham_hr%norb))
  ham_hr%hr = cmplx_0
  !
  do ik = 1, nk
    do io = 1, ham_hr%norb
      ktau = sum(kmesh(:, ik) * ham_hr%tau(:, io)) * twopi
      orb_phase(io) = cmplx(cos(ktau), sin(ktau), KIND=dp)
    enddo
    !
    do ir = 1, ham_hr%nrpt
      rdotk = sum(ham_hr%rvec(:, ir) * kmesh(:, ik)) * twopi
      rphase = cmplx(cos(rdotk), -sin(rdotk), KIND=dp)
      do io = 1, ham_hr%norb
        do jo = 1, ham_hr%norb
          ham_hr%hr(io, jo, ir) = ham_hr%hr(io, jo, ir) + kw(ik) * ham_hr%weight(ir) * &
              rphase * orb_phase(io) * conjg(orb_phase(jo)) * hk_k(io, jo, ik)
        enddo
      enddo
    enddo
  enddo
  !
  ham_hr%hr = ham_hr%hr / sum_kwt
  deallocate(orb_phase)
END SUBROUTINE reconstruct_hr_from_hk
```

**归一化**: 使用 `sum_kwt = sum(kw(1:nk))`，而非 `nkirr`

---

##### 2.9.4 `validate_hr_reconstruction(ham_hr, hk_ref, nk, kmesh, max_err)`

**功能**: 验证重构的 H(R) 通过正变换 `calc_hk` 是否能还原目标 H(k)

**源代码** (第 223-246 行):

```fortran
SUBROUTINE validate_hr_reconstruction(ham_hr, hk_ref, nk, kmesh, max_err)
  TYPE(wannham), intent(in) :: ham_hr
  integer, intent(in) :: nk
  complex(dp), dimension(:, :, :), intent(in) :: hk_ref
  real(dp),    dimension(:, :), intent(in) :: kmesh
  real(dp), intent(out) :: max_err
  !
  integer :: ik
  complex(dp), allocatable :: hk_chk(:,:)
  !
  allocate(hk_chk(ham_hr%norb, ham_hr%norb))
  max_err = 0.0_dp
  !
  do ik = 1, nk
    call calc_hk(hk_chk, ham_hr, kmesh(:, ik))
    max_err = max(max_err, maxval(abs(hk_chk - hk_ref(:, :, ik))))
  enddo
  !
  deallocate(hk_chk)
END SUBROUTINE validate_hr_reconstruction
```

---

##### 2.9.5 `init_periodic_reconstruction_ham(ham_template, nk1, nk2, nk3, ham_hr)`

**功能**: 构建完整周期性 R 空间网格，匹配自动 k 网格 nk1 x nk2 x nk3

**源代码** (第 249-293 行):

```fortran
SUBROUTINE init_periodic_reconstruction_ham(ham_template, nk1, nk2, nk3, ham_hr)
  TYPE(wannham), intent(in)  :: ham_template
  TYPE(wannham), intent(out) :: ham_hr
  integer, intent(in) :: nk1, nk2, nk3
  !
  integer :: ir1, ir2, ir3, idx
  integer :: r1, r2, r3
  !
  ham_hr%norb = ham_template%norb
  ham_hr%nrpt = nk1 * nk2 * nk3
  allocate(ham_hr%hr(ham_hr%norb, ham_hr%norb, ham_hr%nrpt))
  allocate(ham_hr%weight(ham_hr%nrpt))
  allocate(ham_hr%rvec(3, ham_hr%nrpt))
  allocate(ham_hr%tau(3, ham_hr%norb))
  ham_hr%hr = cmplx_0
  ham_hr%tau = ham_template%tau
  ham_hr%weight = 1.0_dp
  ham_hr%r000 = -1
  !
  idx = 0
  do ir1 = 0, nk1 - 1
    r1 = ir1
    if (r1 > nk1 / 2) r1 = r1 - nk1
    do ir2 = 0, nk2 - 1
      r2 = ir2
      if (r2 > nk2 / 2) r2 = r2 - nk2
      do ir3 = 0, nk3 - 1
        r3 = ir3
        if (r3 > nk3 / 2) r3 = r3 - nk3
        idx = idx + 1
        ham_hr%rvec(:, idx) = real((/r1, r2, r3/), dp)
        if (r1 == 0 .and. r2 == 0 .and. r3 == 0) ham_hr%r000 = idx
      enddo
    enddo
  enddo
  !
  if (ham_hr%r000 < 1) then
    write(stdout, *) 'ERROR: periodic reconstruction grid did not include R=0'
    stop 1
  endif
END SUBROUTINE init_periodic_reconstruction_ham
```

**R 格矢范围**: 对每个分量 $i$，取 $r_i \in [-(n_i/2), n_i/2]$（若 > n_i/2 则折叠回负值）

---

##### 2.9.6 `add_js_coupling(ham_out, norb_bare, J_val, Svec, irpt)`

**功能**: 在 R 空间 ham_out%hr 的第 irpt 个 R 格点上添加 J*(S.sigma)/2 交换耦合

**Spinor 约定**: 对 $n_c = \text{norb\_bare}/2$ 个空间轨道：轨道 $1..n_c$ 为自旋向上，$n_c+1..2n_c$ 为自旋向下

**源代码** (第 296-332 行):

```fortran
SUBROUTINE add_js_coupling(ham_out, norb_bare, J_val, Svec, irpt)
  TYPE(wannham), intent(inout) :: ham_out
  integer,  intent(in) :: norb_bare, irpt
  real(dp), intent(in) :: J_val
  real(dp), dimension(3), intent(in) :: Svec  ! (S_x, S_y, S_z)
  !
  integer :: io, n_c
  complex(dp) :: Jsp, Jsm, Jsz
  !
  n_c  = norb_bare / 2
  !
  Jsz  = cmplx(J_val * Svec(3) / 2.0_dp, 0.0_dp, KIND=dp)
  Jsp  = cmplx(J_val * Svec(1) / 2.0_dp, -J_val * Svec(2) / 2.0_dp, KIND=dp)  ! S_x - i*S_y
  Jsm  = cmplx(J_val * Svec(1) / 2.0_dp,  J_val * Svec(2) / 2.0_dp, KIND=dp)  ! S_x + i*S_y
  !
  do io = 1, n_c
    ham_out%hr(io,     io,     irpt) = ham_out%hr(io,     io,     irpt) + Jsz
    ham_out%hr(io+n_c, io+n_c, irpt) = ham_out%hr(io+n_c, io+n_c, irpt) - Jsz
    ham_out%hr(io,     io+n_c, irpt) = ham_out%hr(io,     io+n_c, irpt) + Jsp
    ham_out%hr(io+n_c, io,     irpt) = ham_out%hr(io+n_c, io,     irpt) + Jsm
  enddo
END SUBROUTINE add_js_coupling
```

**交换矩阵** (对每个空间轨道 io):
- 对角: `+J*S_z/2` (上), `-J*S_z/2` (下)
- 非对角: `+J*(S_x-iS_y)/2` (上->下), `+J*(S_x+iS_y)/2` (下->上)

---

##### 2.9.7 `js_objective(params, n_params, val)`

**功能**: 贝叶斯优化目标函数 -- 计算 L2 本征值失配

**目标函数**:
$$L(J, S) = \frac{1}{N_k} \sum_{\mathbf{k}} \left\| \text{sort}(\lambda(H_{bare}(\mathbf{k}) + H_{JS}(\mathbf{k}))) - \text{sort}(\lambda(H_{eff}^{CC}(\mathbf{k}))) \right\|^2$$

**参数解包**:
- `eff_mode=1` (标量): `params = (J_0, S_x, S_y, S_z)` -- 4 个参数
- `eff_mode=2` (J_TENSOR): `params = (J_R(1:n), S_x, S_y, S_z)` -- (n+3) 个参数
- `eff_mode=3` (J_S_TENSOR): `params = (J_R(1:n), S_x(1:n), S_y(1:n), S_z(1:n))` -- (4n) 个参数

**源代码** (第 335-421 行):

```fortran
SUBROUTINE js_objective(params, n_params, val)
  integer,  intent(in) :: n_params
  real(dp), dimension(n_params), intent(in) :: params
  real(dp), intent(out) :: val
  !
  integer :: ik, n_jrpt_local
  real(dp) :: J_0
  real(dp), dimension(3) :: Svec
  real(dp), allocatable :: J_R(:), S_R(:,:)
  real(dp) :: sraw_norm
  complex(dp), allocatable :: hk_bare(:,:), hk_tmp1(:,:), hk_tmp2(:,:)
  real(dp),    allocatable :: eig_trial(:), eig_eff(:)
  !
  n_jrpt_local = g_n_jrpt
  allocate(J_R(n_jrpt_local), S_R(3, n_jrpt_local))
  !
  ! Unpack parameters based on eff_mode
  if (g_eff_mode == 1) then
    J_0 = params(1)
    call normalize_spin_direction(params(2:4), Svec, sraw_norm)
    J_R = 0.0_dp
    S_R = 0.0_dp
  else if (g_eff_mode == 2) then
    J_0 = 0.0_dp
    J_R(1:n_jrpt_local) = params(1:n_jrpt_local)
    Svec(1:3) = params(n_jrpt_local+1:n_jrpt_local+3)
    S_R = spread(Svec, 2, n_jrpt_local)
  else
    J_0 = 0.0_dp
    J_R(1:n_jrpt_local) = params(1:n_jrpt_local)
    S_R(1, 1:n_jrpt_local) = params(n_jrpt_local+1:2*n_jrpt_local)
    S_R(2, 1:n_jrpt_local) = params(2*n_jrpt_local+1:3*n_jrpt_local)
    S_R(3, 1:n_jrpt_local) = params(3*n_jrpt_local+1:4*n_jrpt_local)
  endif
  !
  allocate(hk_bare(g_norb_bare, g_norb_bare))
  allocate(hk_tmp1(g_norb_bare, g_norb_bare))
  allocate(hk_tmp2(g_norb_cc, g_norb_cc))
  allocate(eig_trial(g_norb_bare))
  allocate(eig_eff(g_norb_cc))
  !
  val = 0.0_dp
  !
  do ik = 1, g_nkirr
    call calc_hk(hk_bare, g_ham_bare, g_kvec(:, ik))
    hk_tmp1 = hk_bare
    !
    if (g_eff_mode == 1) then
      call add_js_coupling_kspace(hk_tmp1, g_norb_bare, J_0, Svec)
    else
      call add_js_coupling_tensor_kspace(hk_tmp1, g_norb_bare, J_R, S_R, &
                                         n_jrpt_local, g_rvec_J, g_kvec(:,ik))
    endif
    !
    call eigen(eig_trial, hk_tmp1, g_norb_bare)
    !
    hk_tmp2 = g_hk_eff_cc(:, :, ik)
    call eigen(eig_eff, hk_tmp2, g_norb_cc)
    !
    val = val + sum((eig_trial - eig_eff)**2)
  enddo
  !
  val = val / real(g_nkirr, dp)
  deallocate(hk_bare, hk_tmp1, hk_tmp2, eig_trial, eig_eff, J_R, S_R)
END SUBROUTINE js_objective
```

---

##### 2.9.8 `add_js_coupling_kspace(hk, norb, J_val, Svec)`

**功能**: 在 k 空间直接添加标量 J*(S.sigma)/2（仅 R=0 情况）

**源代码** (第 424-448 行):

```fortran
SUBROUTINE add_js_coupling_kspace(hk, norb, J_val, Svec)
  complex(dp), dimension(norb, norb), intent(inout) :: hk
  integer,  intent(in) :: norb
  real(dp), intent(in) :: J_val
  real(dp), dimension(3), intent(in) :: Svec
  !
  integer :: io, n_c
  complex(dp) :: Jsz, Jsp, Jsm
  !
  n_c = norb / 2
  Jsz = cmplx(J_val * Svec(3) / 2.0_dp, 0.0_dp, KIND=dp)
  Jsp = cmplx(J_val * Svec(1) / 2.0_dp, -J_val * Svec(2) / 2.0_dp, KIND=dp)
  Jsm = cmplx(J_val * Svec(1) / 2.0_dp,  J_val * Svec(2) / 2.0_dp, KIND=dp)
  !
  do io = 1, n_c
    hk(io,     io)     = hk(io,     io)     + Jsz
    hk(io+n_c, io+n_c) = hk(io+n_c, io+n_c) - Jsz
    hk(io,     io+n_c) = hk(io,     io+n_c) + Jsp
    hk(io+n_c, io)     = hk(io+n_c, io)     + Jsm
  enddo
END SUBROUTINE add_js_coupling_kspace
```

---

##### 2.9.9 `add_js_coupling_tensor_kspace(hk, norb, jeff_R, S_R, n_jrpt, rvec_J, kvec)`

**功能**: 添加张量 JS 耦合：$\sum_R J(R) e^{i\mathbf{k}\cdot\mathbf{R}} (\mathbf{S}(\mathbf{R})\cdot\boldsymbol{\sigma})/2$

**源代码** (第 451-497 行):

```fortran
SUBROUTINE add_js_coupling_tensor_kspace(hk, norb, jeff_R, S_R, n_jrpt, rvec_J, kvec)
  complex(dp), dimension(norb, norb), intent(inout) :: hk
  integer,  intent(in) :: norb, n_jrpt
  real(dp), dimension(n_jrpt), intent(in) :: jeff_R
  real(dp), dimension(3, n_jrpt), intent(in) :: S_R
  real(dp), dimension(3, n_jrpt), intent(in) :: rvec_J
  real(dp), dimension(3), intent(in) :: kvec
  !
  integer :: ir, io, n_c
  real(dp) :: rdotk
  complex(dp) :: Jk_R, Jsz_R, Jsp_R, Jsm_R
  complex(dp) :: Jsz, Jsp, Jsm
  !
  n_c = norb / 2
  Jsz = cmplx_0; Jsp = cmplx_0; Jsm = cmplx_0
  do ir = 1, n_jrpt
    rdotk = sum(kvec(:) * rvec_J(:, ir)) * twopi
    Jk_R = jeff_R(ir) * cmplx(cos(rdotk), sin(rdotk), KIND=dp)
    Jsz_R = Jk_R * cmplx(S_R(3, ir) / 2.0_dp, 0.0_dp, KIND=dp)
    Jsp_R = Jk_R * cmplx(S_R(1, ir) / 2.0_dp, -S_R(2, ir) / 2.0_dp, KIND=dp)
    Jsm_R = Jk_R * cmplx(S_R(1, ir) / 2.0_dp,  S_R(2, ir) / 2.0_dp, KIND=dp)
    Jsz = Jsz + Jsz_R
    Jsp = Jsp + Jsp_R
    Jsm = Jsm + Jsm_R
  enddo
  !
  do io = 1, n_c
    hk(io,     io)     = hk(io,     io)     + Jsz
    hk(io+n_c, io+n_c) = hk(io+n_c, io+n_c) - Jsz
    hk(io,     io+n_c) = hk(io,     io+n_c) + Jsp
    hk(io+n_c, io)     = hk(io+n_c, io)     + Jsm
  enddo
END SUBROUTINE add_js_coupling_tensor_kspace
```

**说明**: 对每个 R 求和 $J(\mathbf{R}) e^{i\mathbf{k}\cdot\mathbf{R}}$，乘以 $(\mathbf{S}(\mathbf{R})\cdot\boldsymbol{\sigma})/2$ 的三个分量，然后加到 H(k) 上。eff_mode=2 时 S_R 各列相同（均匀 S），eff_mode=3 时 S_R 按 R 变化。

---

##### 2.9.10 `js_objective_callback(params, n)`

**功能**: 包装函数，匹配 `bayesian_optimize` / `cmaes_optimize` 的回调接口

**源代码** (第 501-507 行):

```fortran
FUNCTION js_objective_callback(params, n) RESULT(val)
  integer,  intent(in) :: n
  real(dp), dimension(n), intent(in) :: params
  real(dp) :: val
  call js_objective(params, n, val)
END FUNCTION js_objective_callback
```

#### PROGRAM WannEffJS — 主程序工作流

**源代码**: 第 514-1175 行

##### 工作流概览

```
1. init_para + read_effjs_input         — MPI 初始化，读取 &EFFJS namelist
2. read_ham(ham, seed)                  — 读取完整系统 HR（存入 lattice 模块全局 ham）
3. read_posfile(seed.pos)               — 设置 tau, avec, nsite, nbasis
4. wannham_shift_ef(ham, mu)            — 移费米能
5. read_ham(ham_bare, seedbare)         — 读取纯导带 HR
6. 手动设置 ham_bare%tau                — 从 seedbare pos 文件赋值
7. read_kmesh('IBZKPT')                 — k 网格
8. 构建 CC/FF 索引                      — 从 ff_orbital_indices 显式指定
9. 设置 J(R) R 网格                     — eff_mode > 1 时
10. downfold_rspace                     — R 空间舒尔补 downfolding
11. 写 seed_downfold HR 文件
12. 设置模块全局变量（g_hk_eff_cc 等）
13. 贝叶斯/CMA-ES 优化                 — eff_mode 1/2/3
14. tol_Jeff 裁剪
15. 写 seed_JS.output
16. 构建 T=0 输出 Hamiltonian          — ham_bare + JS 耦合
17. 写 seedbare_hr_0K
18. 计算 AHC sigma_xy + sigma_xx (T=0)
19. 本征值比较（QPOINTS 路径）          — seed vs seed_downfold
20. Berry 曲率 k-map 输出（可选）
21. MC 温度扫描                         — classical_mc_run
22. 每个温度：重建 HR + 输运计算
23. Cleanup + finalize
```

##### 关键步骤详细说明

**CC/FF 索引构建** (第 652-704 行):

CC 索引由 `ff_orbital_indices` 显式指定。Python 在生成 HR 文件时明确写入 FF 轨道索引列表，Fortran 构建 `cc_idx` 为 seed 中所有不在 `ff_orbital_indices` 内的索引（升序排列）：

```fortran
allocate(cc_idx(norb_cc))
if (n_ff_orbital_indices > 0) then
  jj = 0
  do ii = 1, norb_full
    is_ff = any(ff_orbital_indices(1:n_ff_orbital_indices) == ii)
    if (.not. is_ff) then
      jj = jj + 1
      if (jj <= norb_cc) cc_idx(jj) = ii
    endif
  enddo
else
  ! 后向兼容：无 FF 指定时假设前 norb_bare 个为 CC
  do ii = 1, norb_cc
    cc_idx(ii) = ii
  enddo
endif
```

**J(R) R 网格** (第 706-732 行):

- 若 `J_R_range = 0`：使用 ham_bare 的 R 网格
- 否则：`find_ws` 构建 WS 网格

**优化器选择** (第 793-880 行):

| eff_mode | 参数数 | 优化器 | 参数布局 |
|----------|--------|--------|----------|
| 1 (标量) | 4 | GP-BO | `(J_0, S_x, S_y, S_z)` |
| 2 (J_TENSOR) | n+3 | GP-BO (n+3<=20) 或 CMA-ES (>20) | `(J_R(1:n), S_x, S_y, S_z)` |
| 3 (J_S_TENSOR) | 4n | CMA-ES | `(J_R, S_x(1:n), S_y(1:n), S_z(1:n))` |

**T=0 输出 Hamiltonian** (第 948-995 行):

深拷贝 `ham_bare` -> `ham_out`，然后根据 `eff_mode` 添加 JS 耦合：
- mode 1: 在 R=0 添加标量 J*S 耦合
- mode 2: 在每个 J(R) 对应的 R 点添加均匀 S 耦合
- mode 3: 在每个 J(R) 对应的 R 点添加 per-R S 耦合

**MC 温度扫描** (第 1053-1143 行):

1. 获取 f-site 位置（seed 中有但 seedbare 中没有的位点）
2. `classical_mc_run` 运行经典 Heisenberg 模型，得到 `mvec_vs_T(3, n_temps)`
3. 对每个温度：`S_eff_vec = S_mag_opt * mvec_vs_T(:, iT)`，重建 `ham_out`，计算输运

**输出文件**:
- `{seedbare}_hr_0K` -- T=0 有效 HR
- `{seedbare}_hr_{T}K` -- 温度依赖 HR（温度标签：`nint(T_eV * 11604.522)` K）
- `{seed}_JS.output` -- 拟合参数 J_opt, Svec_opt, S_mag, L2_best, J(R) 表
- `{seed}_JS_result.dat` -- 本征值比较（seed vs seed_downfold 沿 QPOINTS 路径）
- `{seed}_berry_curvature.dat` -- Berry 曲率 k-map（可选）
- `{seed}_transport_vs_T.dat` -- T(K)  sigma_xy  sigma_xx

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

## 8. MPI 并行化状态分析

### 8.1 para 模块架构

WannChi 项目使用 `para.f90`（MPI版本）和 `para_serial.f90`（串行版本）实现并行/串行切换：

- **MPI 版本** (`modules/para.f90`): 链接 `-lmpi`，使用 `__MPI` 预处理器宏
- **串行版本** (`modules/para_serial.f90`): 所有 MPI 调用设为空操作，`distribute_calc` 返回 `first_idx=1, last_idx=nidx`

**构建选择**：
- Server (Intel+MPI): `cd modules && make mod.a` → 链接 `para.f90`
- Laptop (gfortran): `cd build_laptop && make ...` → 链接 `para_serial.f90`

### 8.2 模块 MPI 使用状态

| 模块 | 使用 para | MPI 函数 | 说明 |
|------|---------|---------|------|
| `constants.f90` | 否 | - | 仅常量定义，无需并行 |
| `linalgwrap.f90` | 否 | - | BLAS/LAPACK 封装，已被库优化 |
| `symmetry.f90` | 否 | - | 小矩阵操作，非瓶颈 |
| `simp.f90` | 否 | - | 积分简化，非瓶颈 |
| `wannlog.f90` | 否 | - | 日志输出，无需并行 |
| **`gp_bo.f90`** | **是** | `distribute_calc`, `para_merge_real`, `inode` | 候选点评估并行（Latin Hypercube 初始化分布） |
| **`cma_es.f90`** | **是** | `distribute_calc`, `para_merge_real`, `inode`, `nnode` | 种群个体评估并行 |
| **`classical_mc.f90`** | **是** | `distribute_calc`, `para_merge_real` | 温度点并行（MC 温度扫描分布） |
| **`transp_calc.f90`** | **是** | `distribute_calc`, `para_merge_real0`, `para_merge_cmplx0`, `para_merge_real`, `inode` | k 点并行（sigma_xy, sigma_xx, berry_curvature_kmap） |
| `para.f90` | 自身 | MPI 模块 | 核心并行抽象 |
| `para_serial.f90` | 自身 | 串行桩 | 单进程回退 |
| `lattice.f90` | 是 | `inode`, `para_sync_int`, `para_sync_cmplx` | 晶格数据同步 |
| `wanndata.f90` | 是 | `para_sync_int`, `inode` | Hamiltonian 维度同步 |
| `pade.f90` | 是 | `inode` | 主进程 I/O 判断 |
| `intRPA.f90` | 是 | `inode`, `para_sync_int0`, `para_sync_int`, `para_sync_real` | RPA 块索引同步 |

### 8.3 源文件 MPI 使用状态

| 源文件 | 使用 para | 主要并行模式 |
|--------|---------|------------|
| `compute_chi.f90` | 是 | `distribute_calc(nq)` 分布 q 点循环 |
| `wannchi.f90` | 是 | `init_para`, `finalize_para` |
| `wannchiRPA.f90` | 是 | `init_para`, `distribute_calc` |
| `wannchi_bare.f90` | 是 | `init_para`, `distribute_calc` |
| `wannband.f90` | 是 | `distribute_calc`, `para_merge_real` |
| **`wanneff_JS.f90`** | 是 | `init_para`, `distribute_calc`, `para_merge_*` |
| `input.f90` | 是 | `para_sync_*` 广播输入参数 |
| `output_chi.f90` | 是 | `inode` 主进程输出 |
| `postchi.f90` | 是 | `init_para`, `finalize_para` |

### 8.4 并行化实现细节

以上四个模块的 MPI 并行化均已实现：

**1. `transp_calc.f90`** — k 点并行（✅ 已实现）

三个子程序均使用 `distribute_calc` 分布 k 点：
- `calc_sigma_xy` (第 180 行): `call distribute_calc(nk)` → `do ik = first_idx, last_idx`
- `calc_berry_curvature_kmap` (第 275 行): 同上
- `calc_sigma_xx` (第 335 行): 同上

合并结果: `para_merge_real0(sigma_acc)` / `para_merge_real(omega_kmap, nk)` / `para_merge_cmplx0(sigma_acc)`

**2. `classical_mc.f90`** — 温度点并行（✅ 已实现）

`classical_mc_run` (第 404 行):
```fortran
call distribute_calc(n_temps)
iT_start = first_idx
iT_end = last_idx
n_local_temps = last_idx - first_idx + 1
allocate(mvec_local(3, n_local_temps))
! ... local MC sweeps ...
call para_merge_real(mvec_local, 3 * n_local_temps)  ! 合并结果
```

**3. `gp_bo.f90`** — 候选点并行（✅ 已实现）

`latin_hypercube` (第 500 行) 和 `bayesian_optimize` 主循环中的候选评估通过 `distribute_calc` 分布（第 38-51 行声明 `use para, only : distribute_calc, first_idx, last_idx, para_merge_real, inode`）。

**4. `cma_es.f90`** — 种群并行（✅ 已实现）

`cmaes_optimize` (第 153 行):
```fortran
call distribute_calc(lambda)
! ... rank 0 samples lambda offspring ...
! 每 rank 评估自己分到的个体
call para_merge_real(fitness, lambda)  ! 合并适应度值
```

**注意**: 目前 MPI 仅在 Server 构建（Intel + MPI）中启用。Laptop 构建使用 `para_serial.f90`，所有 `distribute_calc` 返回 `first_idx=1, last_idx=n`，相当于串行执行。

### 8.5 验证方法

**Laptop 构建（gfortran + para_serial）**：
```bash
cd wannchi/build_laptop
make wanneff_js.x wannband.x
cp wanneff_js.x wannband.x ../src/
```

**完整流程测试**：
```bash
source /Users/ykxu/Projects/hrJS/hrJS/bin/activate
cd wannchi/tests
python3 kagome_f_spinor_test.py
```

预期结果：7-8 PASS，0-1 FAIL（已知数值问题导致间歇性失败）。

---

## 9. 已知错误与设计缺陷

本文档系统整理了源代码中发现的所有潜在问题。按影响程度分为三类：**严重**（运行时错误或结果错误）、**警告**（数值精度问题或逻辑缺陷）、**建议**（代码质量或可维护性问题）。

### 9.1 严重错误（Critical）

#### B1. `cma_es.f90` 第 134 行 — 格式字符串描述符数量不匹配

**位置**: `modules/cma_es.f90:134`

**问题**: `write(stdout, '(A,I6,A)')` 有 3 个描述符 `(A,I6,A)` 但插入了 4 个值：

```fortran
write(stdout, '(A,I6,A)') "  CMA-ES: n_params=", n_params, " lambda=", lambda
!                          A              I6          A      ...         I6 ...
```

**后果**: 运行时格式错误或错误的输出对齐。

**修复**:
```fortran
write(stdout, '(A,I6,A,I6)') "  CMA-ES: n_params=", n_params, " lambda=", lambda
```

#### B2. `gp_bo.f90` 第 393-398 行 — GP 对数边际似然的行列式计算错误

**位置**: `modules/gp_bo.f90:393-398`

**问题**: `gp_log_marginal_likelihood` 通过 `invmat` 获得 `K^{-1}`（LU 分解），然后从 `K^{-1}` 的对角元计算 `log|K|`：

```fortran
call invmat(Kmat, n)  ! Kmat 变成了 K^{-1}
log_det_K = 0.0_dp
do ii = 1, n
    log_det_K = log_det_K + log(Kmat(ii, ii))  ! 错：用的是 K^{-1} 的对角元
enddo
log_det_K = -2.0_dp * log_det_K
```

LU 分解给出 `K^{-1}` 而非 `K` 的 Cholesky 分解。从 `K^{-1}` 的对角元无法得到正确的 `log|K|`。

**后果**: LML 计算错误 → `gp_optimize_ls` 中的长度尺度优化不可靠 → GP-BO 收敛慢或不收敛。

**修复**: 正确计算 `log|K|` 的方法：
- 方法 1：使用 Cholesky 分解（`potrf`）后对对角元求和
- 方法 2：用 LAPACK `dgetrf` 的行列式例程

#### B3. `cma_es.f90` 第 144 行 — 种群采样使用均匀分布而非高斯分布

**位置**: `modules/cma_es.f90:144`

**问题**: CMA-ES 的核心是协方差矩阵自适应高斯采样，但当前代码使用均匀分布：

```fortran
x_pop((k-1)*n_params + jj) = mean(jj) + sigma(jj) * (u - 0.5_dp) * sqrt(12.0_dp)
```

这产生的是 **均匀分布** $U[-\sqrt{3}\sigma, +\sqrt{3}\sigma]$，不是 $N(0, \sigma^2)$。

**后果**: 协方差矩阵 `C` 的Adaptation（`eigen-decomposition` 更新）完全失效，因为采样根本没有用到多维高斯分布。优化行为退化为随机坐标方向上的均匀搜索。

**修复**:
```fortran
! 使用 Box-Muller 或 Ziggurat 方法生成标准高斯样本
call random_gaussian(u_gauss)  ! 需要实现高斯随机数生成
x_pop((k-1)*n_params + jj) = mean(jj) + sigma(jj) * u_gauss
```

---

### 9.2 警告（Design Issues）

#### B4. `wanndata.f90` — `r000` 未初始化时可能导致段错误

**位置**: `modules/wanndata.f90:196`

**问题**: `r000` 仅在 `read_ham` 中当且仅当 HR 文件中包含 `R=(0,0,0)` 格点时才被设置。如果文件中没有 R=0（罕见但可能），`ham%r000` 保持未初始化状态（Fortran 默认值 0 或随机值）。

之后 `wannham_shift_ef` 使用 `ham%r000` 访问数组索引：

```fortran
do ii=1, ham%norb
    ham%hr(ii, ii, ham%r000) = ham%hr(ii, ii, ham%r000)-mu  ! 若 r000=0 可能越界
enddo
```

**后果**: 若 `r000` 值为 0（默认整数），会错误地修改第一个 R 格点的对角元而非 R=0。若为其他随机值，则越界访问。

**建议**: 在 `read_ham` 末尾添加检查：
```fortran
if (ham%r000 < 1 .or. ham%r000 > ham%nrpt) then
    write(stdout, *) "ERROR: R=0 not found in HR file"
    stop 1
endif
```

#### B5. `linalgwrap.f90` — LAPACK `info` 返回码未被检查

**位置**: `modules/linalgwrap.f90:92-95, 108-111, 128-129, 149-150`

**问题**: 所有 LAPACK 调用（`dgetrf/dgetri`, `zgetrf/zgetri`, `zheev`, `zgeev`）都声明了 `info` 但从不检查其值：

```fortran
integer :: info
call dgetrf(ndim, ndim, xmat, ndim, ipiv, info)  ! info 未被检查
call dgetri(ndim, xmat, ndim, ipiv, work, ndim, info)
```

**后果**: 矩阵奇异、数值不稳定或维度错误时，程序静默返回错误结果而非报错退出。

**建议**: 添加 `info` 检查：
```fortran
if (info /= 0) then
    write(stdout, *) "FATAL: LAPACK error in dinvmat, info=", info
    stop 1
endif
```

#### B6. `transp_calc.f90` `calc_sigma_xx` — `temperature` 参数未使用

**位置**: `modules/transp_calc.f90:319`

**问题**: `calc_sigma_xx` 接受 `temperature` 参数但从未使用。调用处也传入了 `temperature=0.0_dp`：

```fortran
SUBROUTINE calc_sigma_xx(sigma_xx, ham, kvec_all, kwt_all, nk, &
                        mu_chem, temperature, broadening)  ! temperature 未被使用
```

Kubo-Greenwood 公式中的温度依赖应通过费米函数体现，但当前实现仅用 Lorentzian 展宽 `η`。

**后果**: 温度相关的输运性质计算不准确（始终是 T=0 的结果）。

#### B7. `wannlog.f90` — 计时使用 CPU 时间而非 wall-clock 时间

**位置**: `modules/wannlog.f90:44, 55, 74, 106`

**问题**: `wannlog` 使用 `cpu_time()` 但注释声称是 "wall-clock time"：

```fortran
real(dp)  :: wl_wall_start  ! absolute start time (seconds) -- 注释声称 wall-clock
...
CALL cpu_time(wl_wall_start)  ! 但实际是 CPU 时间
```

`cpu_time()` 报告的是进程的 CPU 时间，不包括其他进程的等待时间，在 MPI 程序中意义有限。

**建议**: 使用 `system_clock` 或 MPI 的墙钟时间接口：
```fortran
call system_clock(count=wall_start, count_rate=count_rate)
```

#### B8. `wanneff_JS.f90` 第 143 行 — Schur 补符号不考虑能量窗口

**位置**: `src/wanneff_JS.f90:143`

**问题**: Schur 补 downfolding 始终使用负号：

```fortran
call zgemm('N', 'N', n_cc, n_cc, n_ff, -cmplx_1, tmp, n_cc, &
            H_fc, n_ff, cmplx_1, H_cc, n_cc)
```

物理上：
- FF 态在 **费米能以下**（填充）→ 用 `-H_CF * H_FF^{-1} * H_FC`（当前实现）
- FF 态在 **费米能以上**（空）→ 应改用 `+H_CF * H_FF^{-1} * H_FC`

**后果**: 对于 f 电子在 `EF + Δ`（如 Kondo 系统）的情形，downfold 结果错误。

**建议**: 检测 H_FF 在 R=0 的 on-site 对角元，若为正（高于费米）则翻转符号。

#### B9. `wanneff_JS.f90` 第 1114-1125 行 — J_S_TENSOR 模式的 MC 温度缩放缺失

**位置**: `src/wanneff_JS.f90:1114-1125`

**问题**: `eff_mode=3`（J_S_TENSOR）时，MC 温度扫描循环中 `S_R_opt` 没有按 `mvec_vs_T` 缩放：

```fortran
do ii = 1, n_jrpt
    if (abs(jeff_R(ii)) < eps6) cycle
    do jj = 1, ham_out%nrpt
        if (all(abs(ham_out%rvec(:,jj) - rvec_J(:,ii)) < 0.5_dp)) then
            CALL add_js_coupling(ham_out, norb_bare, jeff_R(ii), S_R_opt(:,ii), jj)
            ! S_R_opt 恒定，不随温度变化！
```

而 `eff_mode=1` 和 `eff_mode=2` 正确使用了 `S_eff_vec = S_mag_opt * mvec_vs_T(:, iT)`。

**后果**: J_S_TENSOR 模式的温度依赖结果不准确。

#### B10. `wanneff_JS.f90` 第 1073 行 — Weiss mean-field 使用硬编码配位数 z=6

**位置**: `src/wanneff_JS.f90:1073`

**问题**:
```fortran
J_mc_used = J_opt * S_mag_opt / 6.0_dp  ! 硬编码 z=6
```

配位数 `z=6` 是 kagome 晶格的最近邻数，但代码没有根据实际晶格几何自动检测。

**后果**: 对于非 kagome 晶格，MC 温度扫描的 J_mc 不准确。

---

### 9.3 代码质量建议（Minor）

#### B11. `wanneff_JS.f90` 第 1062-1067 行 — `nsite_f` 和 `frac_pos_f` 计算的边界问题

**位置**: `src/wanneff_JS.f90:1062-1067`

**问题**: f-site 位置计算假设 seed 和 seedbare 共享相同的位点顺序，最后 `nsite_f` 个位点是 f-site：

```fortran
nsite_f = nsite_full - nsite_bare  ! 假设相同的站点顺序
if (nsite_f < 1) nsite_f = 1
do ii = 1, max(1, nsite_f)
    frac_pos_f(:, ii) = xat(:, min(nsite_bare + ii, nsite_full))
```

如果 seed 中额外的 f 轨道分散在不同原子位点（而非集中在一起），这个假设会导致 f-site 位置错误。

#### B12. `transp_calc.f90` `calc_g0` 外部依赖

**位置**: `modules/transp_calc.f90:328`

**问题**: `calc_sigma_xx` 中引用了外部过程：
```fortran
external :: calc_g0
```

但 `calc_g0` 定义在 `src/green.f90` 中（不在 `modules/` 下）。这使得 `transp_calc.f90` 依赖于特定的模块文件链接顺序。

**建议**: 将 `calc_g0` 移入 `modules/green.f90` 或在 `transp_calc.f90` 内部实现。

---

### 9.4 Bug 影响汇总

| ID | 严重性 | 文件 | 影响 |
|----|--------|------|------|
| B1 | 严重 | `cma_es.f90:134` | 运行时格式错误 |
| B2 | 严重 | `gp_bo.f90:393-398` | LML 计算错误 → BO 不收敛 |
| B3 | 严重 | `cma_es.f90:144` | 采样退化为均匀分布 |
| B4 | 警告 | `wanndata.f90:196` | R=0 缺失时越界访问 |
| B5 | 警告 | `linalgwrap.f90` | LAPACK 错误静默传播 |
| B6 | 警告 | `transp_calc.f90:319` | σ_xx 温度依赖缺失 |
| B7 | 警告 | `wannlog.f90:44` | CPU 时间≠wall-clock |
| B8 | 警告 | `wanneff_JS.f90:143` | Kondo 系统 downfold 错误 |
| B9 | 警告 | `wanneff_JS.f90:1114` | J_S_TENSOR 温度缩放缺失 |
| B10 | 警告 | `wanneff_JS.f90:1073` | 硬编码 z=6 |
| B11 | 建议 | `wanneff_JS.f90:1062` | f-site 位置计算假设 |
| B12 | 建议 | `transp_calc.f90:328` | 外部过程依赖 |

---

## 总结

本文档提供了 WannChi 项目中所有子程序的详细分析，包括：

1. **功能描述**: 每个子程序的具体作用
2. **接口**: 输入输出参数类型和含义
3. **实现方法**: 算法实现细节
4. **算法**: 计算流程和逻辑
5. **对应公式**: 物理和数学公式
6. **并行化状态**: MPI 使用情况（第8节）
7. **已知错误**: 系统性 Bug 文档（第9节，共12个问题）

所有子程序均按照模块和源文件进行组织，便于代码理解和维护。