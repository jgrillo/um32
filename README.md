# UM-32

[![Build Status](https://travis-ci.org/jgrillo/um32.svg?branch=master)](https://travis-ci.org/jgrillo/um32)
[![Crate](https://img.shields.io/crates/v/um32.svg)](https://crates.io/crates/um32)

This is a Rust implementation of the [UM-32 "Universal Machine"](https://esolangs.org/wiki/UM-32).
You can find the specification, various VM images, and more information about
the contest which originally spawned this horror at [boundvariable.org](http://boundvariable.org/).

## Architecture

The Universal Machine consists of 8 unsigned 32 bit registers, a dynamically
allocated heap containing arbitrarily sized arrays of unsigned 32 bit data,
and 14 opcodes. Universal Machine images are stored as a sequence of unsigned
32 bit words with big-endian byte order.

## Implementation

This is an extremely naive implementation written as an exercise to become
better acquainted with Rust. It's currently about 25% slower than Joe's
[C++ implementation](https://github.com/llllllllll/um-32). I'm not sure why.

### TODO

 - [ ] build a decent test suite
 - [x] fix performance issues

#### tests

There is good coverage of the instruction parsing code, and no coverage of
anything else.

#### performance

```
[jgrillo@localhost um32]$ sudo lshw -class cpu
  *-cpu
       description: CPU
       product: Intel(R) Core(TM) i7-6820HQ CPU @ 2.70GHz
       vendor: Intel Corp.
       physical id: 6
       bus info: cpu@0
       version: Intel(R) Core(TM) i7-6820HQ CPU @ 2.70GHz
       serial: None
       slot: U3E1
       size: 2855MHz
       capacity: 4005MHz
       width: 64 bits
       clock: 100MHz
       capabilities: lm fpu fpu_exception wp vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp x86-64 constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single pti ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp cpufreq
       configuration: cores=4 enabledcores=4 threads=8

[jgrillo@localhost um32]$ time ./target/release/um32 ~/src/boundvariable/umbin/midmark.um
read 120440 bytes from /home/jgrillo/src/boundvariable/umbin/midmark.um
 == UM beginning stress test / benchmark.. ==
4.   12345678.09abcdef
3.   6d58165c.2948d58d
2.   0f63b9ed.1d9c4076
1.   8dba0fc0.64af8685
0.   583e02ae.490775c0
Benchmark complete.

real	0m0.486s
user	0m0.473s
sys	0m0.005s

[jgrillo@localhost um32]$ time ./target/release/um32 ~/src/boundvariable/sandmark.umz
read 56364 bytes from /home/jgrillo/src/boundvariable/sandmark.umz
trying to Allocate array of size 0..
trying to Abandon size 0 allocation..
trying to Allocate size 11..
trying Array Index on allocated array..
trying Amendment of allocated array..
checking Amendment of allocated array..
trying Alloc(a,a) and amending it..
comparing multiple allocations..
pointer arithmetic..
check old allocation..
simple tests ok!
about to load program from some allocated array..
success.
verifying that the array and its copy are the same...
success.
testing aliasing..
success.
free after loadprog..
success.
loadprog ok.
 == SANDmark 19106 beginning stress test / benchmark.. ==
100. 12345678.09abcdef
99.  6d58165c.2948d58d
98.  0f63b9ed.1d9c4076

   ...

3.   7c7394b2.476c1ee5
2.   f3a52453.19cc755d
1.   2c80b43d.5646302f
0.   a8d1619e.5540e6cf
SANDmark complete.

real	0m28.798s
user	0m28.726s
sys	0m0.007s

```
According to [this source](https://github.com/rlew/um/tree/master/ums), the
`midmark.um` benchmark should run in "about one second".

These blog posts and articles provide a helpful introduction to profiling Rust programs:
 1. http://blog.adamperry.me/rust/2016/07/24/profiling-rust-perf-flamegraph/
 2. http://www.codeofview.com/fix-rs/2017/01/24/how-to-optimize-rust-programs-on-linux/
 3. https://gist.github.com/jFransham/369a86eff00e5f280ed25121454acec1
 4. https://llogiq.github.io/2017/06/01/perf-pitfalls.html

*NOTE*: Llogiq's post contains some very important advice on
configuring cargo to tell rustc to emit code optimized for a
particular CPU. The running times documented above were measured with
a binary compiled using the following settings in `~/.cargo/config`:

```toml
[target.'cfg(any(windows, unix))']
rustflags = ["-C target-cpu=native"]
```
