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
better acquainted with Rust. It's currently about an order of magnitude 
slower than a fast Universal Machine implementation. I'd like to correct this.

### TODO

 - [ ] build a decent test suite
 - [ ] fix performance issues

#### tests 

There is good coverage of the instruction parsing code, and no coverage of 
anything else. This will need to be fixed before trying to optimize.
 
#### performance 

This thing is about an order of magnitude slower than it should be:
```
[jgrillo@localhost um32]$ time ./target/release/um32 ~/src/boundvariable/umbin/midmark.um 
read 120440 bytes from /home/jgrillo/src/boundvariable/umbin/midmark.um
 == UM beginning stress test / benchmark.. ==
4.   12345678.09abcdef
3.   6d58165c.2948d58d
2.   0f63b9ed.1d9c4076
1.   8dba0fc0.64af8685
0.   583e02ae.490775c0
Benchmark complete.

real	0m11.254s
user	0m11.232s
sys	0m0.007s

```
According to [this source](https://github.com/rlew/um/tree/master/ums), the 
`midmark.um` benchmark should run in "about one second".

These blog posts provide a helpful introduction to profiling Rust programs:
 1. http://blog.adamperry.me/rust/2016/07/24/profiling-rust-perf-flamegraph/
 2. http://www.codeofview.com/fix-rs/2017/01/24/how-to-optimize-rust-programs-on-linux/
