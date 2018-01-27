extern crate byteorder;

use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::{stdout, stdin, Cursor};
use std::{process, mem};

use byteorder::{ReadBytesExt, BigEndian};

const PROGRAM_ADDRESS: usize = 0;

// extract a range of bits from a u32
fn extract_bits(value: u32, start: u32, n: u32) -> u32 {
    let mask: u32 = ((1 << n) - 1) << start;
    (value & mask) >> start
}

// the UM-32 has 14 opcodes.
#[derive(Debug, PartialEq)]
enum Opcode {
    Mov,
    ArrayGet,
    ArraySet,
    Add,
    Mul,
    Div,
    Nand,
    Halt,
    Allocate,
    Abandon,
    Output,
    Input,
    Load,
    Ortho,
    Err // this is used if we can't parse an opcode.
}

// functions for instruction parsing.

fn parse_op(instruction: u32) -> Opcode {
    let op = extract_bits(instruction, 28, 4);

    match op {
        0 => Opcode::Mov,
        1 => Opcode::ArrayGet,
        2 => Opcode::ArraySet,
        3 => Opcode::Add,
        4 => Opcode::Mul,
        5 => Opcode::Div,
        6 => Opcode::Nand,
        7 => Opcode::Halt,
        8 => Opcode::Allocate,
        9 => Opcode::Abandon,
        10 => Opcode::Output,
        11 => Opcode::Input,
        12 => Opcode::Load,
        13 => Opcode::Ortho,
        _ => Opcode::Err
    }
}

fn parse_a(instruction: u32, op: &Opcode) -> u32 {
    match *op {
        Opcode::Ortho => extract_bits(instruction, 25, 3),
        _ => extract_bits(instruction, 6, 3)
    }
}

fn parse_b(instruction: u32, op: &Opcode) -> Option<u32> {
    match *op {
        Opcode::Ortho => None,
        _ => Some(extract_bits(instruction, 3, 3))
    }
}

fn parse_c(instruction: u32, op: &Opcode) -> Option<u32> {
    match *op {
        Opcode::Ortho => None,
        _ => Some(extract_bits(instruction, 0, 3))
    }
}

fn parse_value(instruction: u32, op: &Opcode) -> Option<u32> {
    match *op {
        Opcode::Ortho => Some(extract_bits(instruction, 0, 25)),
        _ => None
    }
}

// each machine instruction is deserialized into a convenient
// representation.

#[derive(Debug)]
struct Instruction {
    op: Opcode,
    a: u32,
    b: Option<u32>,
    c: Option<u32>,
    value: Option<u32>
}

impl Instruction {
    fn new(instruction: u32) -> Instruction {
        let op = parse_op(instruction);
        let a = parse_a(instruction, &op);
        let b = parse_b(instruction, &op);
        let c = parse_c(instruction, &op);
        let value = parse_value(instruction, &op);

        Instruction {
            op,
            a,
            b,
            c,
            value
        }
    }
}

// the machine has 8 32bit registers.

#[derive(Debug)]
struct Registers {
    registers: [u32; 8]
}

impl Registers {
    fn new() -> Registers {
        Registers {
            registers: [0_u32; 8]
        }
    }

    // get the value at the given register.
    fn get(&self, register: usize) -> u32 {
        self.registers[register]
    }

    // set the value in the given register.
    fn set(&mut self, register: usize, value: u32) {
        self.registers[register] = value
    }
}

// the machine has a memory which can be dynamically allocated.

#[derive(Debug)]
struct Memory {
    free_list: Vec<usize>,
    heap: Vec<Vec<u32>>
}

impl Memory {
    fn new(instructions: Vec<u32>) -> Memory {
        Memory {
            free_list: Vec::new(),
            heap: vec![instructions]
        }
    }

    // allocate memory initialized with a zeros vec of the given size,
    // returns the address of the newly allocated memory.
    fn allocate(&mut self, size: usize) -> usize {
        let zeros = vec![0_u32; size];

        if self.free_list.len() == 0 {
            self.heap.push(zeros);

            self.heap.len() - 1
        } else {
            let address = self.free_list.pop()
                .expect("no free addresses available");

            mem::replace(
                self.heap.get_mut(address)
                    .expect("memory was not previously allocated"),
                zeros
            );

            address
        }
    }

    // deallocate the memory at the given address.
    fn abandon(&mut self, address: usize) {
        self.free_list.push(address);
        mem::replace(
            self.heap.get_mut(address)
                .expect("memory was not previously allocated"),
            Vec::new()
        );
    }

    // supply contents of the memory at the given address if
    // initialized, None otherwise.
    fn get(&self, address: usize) -> Option<&Vec<u32>> {
        self.heap.get(address)
    }

    // get the instruction corresponding to the given program counter
    fn get_instruction(&self, pc: usize) -> Instruction {
        match self.heap.get(PROGRAM_ADDRESS) {
            Some(program) => Instruction::new(program[pc]),
            None => panic!("program was unallocated")
        }
    }

    // write a value into the given array index at the given address.
    fn set(&mut self, address: usize, idx: usize, value: u32) {
        let memory = self.heap.get_mut(address)
            .expect("memory was unallocated");

        mem::replace(
            memory.get_mut(idx)
                .expect("no value present at given idx"),
            value
        );
    }

    // replace the program with the vector at the given address
    fn load(&mut self, address: usize) {
        let program = self.heap.get(address)
            .expect("found no program at the given address")
            .clone();

        mem::replace(
            self.heap.get_mut(PROGRAM_ADDRESS)
                .expect("found no existing program"),
            program
        );
    }
}

// the machine has 14 opcodes

struct Machine {
    memory: Memory,
    registers: Registers
}

impl Machine {
    fn new(instructions: Vec<u32>) -> Machine {
        Machine {
            memory: Memory::new(instructions),
            registers: Registers::new()
        }
    }

    fn get_instruction(&self, pc: usize) -> Instruction {
        self.memory.get_instruction(pc)
    }

    fn mov(&mut self, instruction: Instruction) {
        let a = instruction.a as usize;
        let b = instruction.b.unwrap() as usize;
        let c = instruction.c.unwrap() as usize;

        if self.registers.get(c) != 0 {
            let value = self.registers.get(b);
            self.registers.set(a, value);
        }
    }

    fn array_get(&mut self, instruction: Instruction) {
        let a = instruction.a as usize;
        let b = instruction.b.unwrap() as usize;
        let c = instruction.c.unwrap() as usize;

        let address = self.registers.get(b) as usize;

        let array = self.memory.get(address)
            .expect("found unallocated array at the given address");
        let idx = self.registers.get(c) as usize;

        let value = array[idx];

        self.registers.set(a, value);
    }

    fn array_set(&mut self, instruction: Instruction) {
        let a = instruction.a as usize;
        let b = instruction.b.unwrap() as usize;
        let c = instruction.c.unwrap() as usize;

        let address = self.registers.get(a) as usize;
        let idx = self.registers.get(b) as usize;
        let value = self.registers.get(c);

        self.memory.set(address, idx, value);
    }

    fn add(&mut self, instruction: Instruction) {
        let a = instruction.a as usize;
        let b = instruction.b.unwrap() as usize;
        let c = instruction.c.unwrap() as usize;

        let value = self.registers.get(b).wrapping_add(self.registers.get(c));

        self.registers.set(a, value);
    }

    fn mul(&mut self, instruction: Instruction) {
        let a = instruction.a as usize;
        let b = instruction.b.unwrap() as usize;
        let c = instruction.c.unwrap() as usize;

        let value = self.registers.get(b).wrapping_mul(self.registers.get(c));

        self.registers.set(a, value);
    }

    fn div(&mut self, instruction: Instruction) {
        let a = instruction.a as usize;
        let b = instruction.b.unwrap() as usize;
        let c = instruction.c.unwrap() as usize;

        let value = self.registers.get(b).wrapping_div(self.registers.get(c));
        self.registers.set(a, value);
    }

    fn nand(&mut self, instruction: Instruction) {
        let a = instruction.a as usize;
        let b = instruction.b.unwrap() as usize;
        let c = instruction.c.unwrap() as usize;

        let value = !(self.registers.get(b) & self.registers.get(c));

        self.registers.set(a, value);
    }

    fn allocate(&mut self, instruction: Instruction) {
        let b = instruction.b.unwrap() as usize;
        let c = instruction.c.unwrap() as usize;

        let size = self.registers.get(c) as usize;

        let address = self.memory.allocate(size);

        self.registers.set(b, address as u32);
    }

    fn abandon(&mut self, instruction: Instruction) {
        let c = instruction.c.unwrap() as usize;
        let address = self.registers.get(c) as usize;

        self.memory.abandon(address);
    }

    fn output(&self, instruction: Instruction) {
        let c = instruction.c.unwrap() as usize;

        let value = self.registers.get(c);

        let byte = value as u8;
        stdout().write(&[byte]).unwrap();
    }

    fn input(&mut self, instruction: Instruction) {
        let c = instruction.c.unwrap() as usize;

        match stdin().bytes().next().unwrap() { // EOF will be None
            Ok(value) => {
                if value as char == '\n' {
                    self.registers.set(c, std::u32::MAX);
                } else {
                    self.registers.set(c, value as u32);
                }
            },
            Err(e) => panic!("Encountered error while reading input: {}", e)
        }
    }

    fn load(&mut self, instruction: Instruction) -> usize {
        let b = instruction.b.unwrap() as usize;
        let c = instruction.c.unwrap() as usize;

        let address = self.registers.get(b) as usize;

        if address != PROGRAM_ADDRESS {
            self.memory.load(address);
        }

        self.registers.get(c) as usize
    }

    fn ortho(&mut self, instruction: Instruction) {
        let a = instruction.a as usize;
        let value = instruction.value.unwrap();

        self.registers.set(a, value);
    }
}

// read machine instructions from file

fn read_instructions(filename: &str) -> Vec<u32> {
    let mut f = File::open(filename).expect("file not found");
    let mut data = Vec::new();
    let mut instructions = Vec::new();

    match f.read_to_end(&mut data) {
        Ok(bytes) => {
            println!("read {} bytes from {}", bytes, filename);

            for i in 0..data.len() / 4 {
                let idx = i * 4;
                let buf = &data[idx..idx + 4];
                let mut rdr = Cursor::new(buf);
                instructions.push(rdr.read_u32::<BigEndian>().unwrap());
            }

            instructions
        },
        Err(e) => panic!(
            "Encountered error while reading from {}: {}", filename, e
        )
    }
}

// run the machine execution loop

fn main() {
    let args: Vec<String> = env::args().collect();
    let filename = &args[1];
    let instructions = read_instructions(filename);
    let mut machine = Machine::new(instructions);
    let mut pc: usize = 0;

    loop { // run the machine until it terminates
        let instruction = machine.get_instruction(pc);
        pc += 1;

        match instruction.op {
            Opcode::Mov => machine.mov(instruction),
            Opcode::ArrayGet => machine.array_get(instruction),
            Opcode::ArraySet => machine.array_set(instruction),
            Opcode::Add => machine.add(instruction),
            Opcode::Mul => machine.mul(instruction),
            Opcode::Div => machine.div(instruction),
            Opcode::Nand => machine.nand(instruction),
            Opcode::Halt => process::exit(0),
            Opcode::Allocate => machine.allocate(instruction),
            Opcode::Abandon => machine.abandon(instruction),
            Opcode::Output => machine.output(instruction),
            Opcode::Input => machine.input(instruction),
            Opcode::Load => pc = machine.load(instruction),
            Opcode::Ortho => machine.ortho(instruction),
            Opcode::Err => panic!(
                "Unknown opcode for instruction {:?}", instruction
            )
        }
    }
}

// tests

#[cfg(test)]
mod tests {
    use super::{extract_bits, Opcode, parse_op, parse_a, parse_b, parse_c, parse_value};

    // extract_bits

    #[test]
    fn test_extract_bits_0() {
        let word: u32 = 0;
        let bits = extract_bits(word, 0, 4);
        let bit_string = format!("{:032b}", bits);
        assert_eq!("00000000000000000000000000000000", bit_string);
    }

    #[test]
    fn test_extract_bits_1() {
        let word: u32 = 1;
        let bits = extract_bits(word, 0, 4);
        let bit_string = format!("{:032b}", bits);
        assert_eq!("00000000000000000000000000000001", bit_string);
    }

    #[test]
    fn test_extract_bits_5() {
        let word: u32 = 5;
        let bits = extract_bits(word, 0, 4);
        let bit_string = format!("{:032b}", bits);
        assert_eq!("00000000000000000000000000000101", bit_string);
    }

    #[test]
    fn test_extract_bits_15() {
        let word: u32 = 15;
        let bits = extract_bits(word, 0, 4);
        let bit_string = format!("{:032b}", bits);
        assert_eq!("00000000000000000000000000001111", bit_string);
    }

    #[test]
    fn test_extract_bits_256() {
        let word: u32 = 256;
        let bits = extract_bits(word, 8, 1);
        let bit_string = format!("{:032b}", bits);
        assert_eq!("00000000000000000000000000000001", bit_string);
    }

    // parse_op

    #[test]
    fn test_parse_op_mov() { // 0
        let instruction: u32 = 0b00000000_00000000_00000000_00000000_u32;
        let opcode = extract_bits(instruction, 28, 4);
        assert_eq!(0, opcode);
        assert_eq!(parse_op(instruction), Opcode::Mov);
    }

    #[test]
    fn test_parse_op_array_get() { // 1
        let instruction: u32 = 0b00010000_00000000_00000000_00000000_u32;
        let opcode = extract_bits(instruction, 28, 4);
        assert_eq!(1, opcode);
        assert_eq!(parse_op(instruction), Opcode::ArrayGet);
    }

    #[test]
    fn test_parse_op_array_set() { // 2
        let instruction: u32 = 0b00100000_00000000_00000000_00000000_u32;
        let opcode = extract_bits(instruction, 28, 4);
        assert_eq!(2, opcode);
        assert_eq!(parse_op(instruction), Opcode::ArraySet);
    }

    #[test]
    fn test_parse_op_add() { // 3
        let instruction: u32 = 0b00110000_00000000_00000000_00000000_u32;
        let opcode = extract_bits(instruction, 28, 4);
        assert_eq!(3, opcode);
        assert_eq!(parse_op(instruction), Opcode::Add);
    }

    #[test]
    fn test_parse_op_mul() { // 4
        let instruction: u32 = 0b01000000_00000000_00000000_00000000_u32;
        let opcode = extract_bits(instruction, 28, 4);
        assert_eq!(4, opcode);
        assert_eq!(parse_op(instruction), Opcode::Mul);
    }

    #[test]
    fn test_parse_op_div() { // 5
        let instruction: u32 = 0b01010000_00000000_00000000_00000000_u32;
        let opcode = extract_bits(instruction, 28, 4);
        assert_eq!(5, opcode);
        assert_eq!(parse_op(instruction), Opcode::Div);
    }

    #[test]
    fn test_parse_op_nand() { // 6
        let instruction: u32 = 0b01100000_00000000_00000000_00000000_u32;
        let opcode = extract_bits(instruction, 28, 4);
        assert_eq!(6, opcode);
        assert_eq!(parse_op(instruction), Opcode::Nand);
    }

    #[test]
    fn test_parse_op_halt() { // 7
        let instruction: u32 = 0b01110000_00000000_00000000_00000000_u32;
        let opcode = extract_bits(instruction, 28, 4);
        assert_eq!(7, opcode);
        assert_eq!(parse_op(instruction), Opcode::Halt);
    }

    #[test]
    fn test_parse_op_allocate() { // 8
        let instruction: u32 = 0b10000000_00000000_00000000_00000000_u32;
        let opcode = extract_bits(instruction, 28, 4);
        assert_eq!(8, opcode);
        assert_eq!(parse_op(instruction), Opcode::Allocate);
    }

    #[test]
    fn test_parse_op_abandon() { // 9
        let instruction: u32 = 0b10010000_00000000_00000000_00000000_u32;
        let opcode = extract_bits(instruction, 28, 4);
        assert_eq!(9, opcode);
        assert_eq!(parse_op(instruction), Opcode::Abandon);
    }

    #[test]
    fn test_parse_op_output() { // 10
        let instruction: u32 = 0b10100000_00000000_00000000_00000000_u32;
        let opcode = extract_bits(instruction, 28, 4);
        assert_eq!(10, opcode);
        assert_eq!(parse_op(instruction), Opcode::Output);
    }

    #[test]
    fn test_parse_op_input() { // 11
        let instruction: u32 = 0b10110000_00000000_00000000_00000000_u32;
        let opcode = extract_bits(instruction, 28, 4);
        assert_eq!(11, opcode);
        assert_eq!(parse_op(instruction), Opcode::Input);
    }

    #[test]
    fn test_parse_op_load() { // 12
        let instruction: u32 = 0b11000000_00000000_00000000_00000000_u32;
        let opcode = extract_bits(instruction, 28, 4);
        assert_eq!(12, opcode);
        assert_eq!(parse_op(instruction), Opcode::Load);
    }

    #[test]
    fn test_parse_op_ortho() { // 13
        let instruction: u32 = 0b11010000_00000000_00000000_00000000_u32;
        let opcode = extract_bits(instruction, 28, 4);
        assert_eq!(13, opcode);
        assert_eq!(parse_op(instruction), Opcode::Ortho);
    }

    #[test]
    fn test_parse_op_err() { // 15
        let instruction: u32 = 0b11110000_00000000_00000000_00000000_u32;
        let opcode = extract_bits(instruction, 28, 4);
        assert_eq!(15, opcode);
        assert_eq!(parse_op(instruction), Opcode::Err);
    }

    // parse_a

    #[test]
    fn test_parse_a() {
        let instruction: u32 = 0b00000000_00000000_00000001_11000000_u32;
        assert_eq!(7, parse_a(instruction, &Opcode::Err))
    }

    #[test]
    fn test_parse_a_ortho() {
        let instruction: u32 = 0b00001110_00000000_00000000_00000000_u32;
        assert_eq!(7, parse_a(instruction, &Opcode::Ortho))
    }

    // parse_b

    #[test]
    fn test_parse_b() {
        let instruction: u32 = 0b00000000_00000000_00000000_00111000_u32;
        assert_eq!(7, parse_b(instruction, &Opcode::Err).unwrap())
    }

    #[test]
    fn test_parse_b_ortho() {
        let instruction: u32 = 0b00000000_00000000_00000000_00000000_u32;
        assert_eq!(None, parse_b(instruction, &Opcode::Ortho))
    }

    // parse_c

    #[test]
    fn test_parse_c() {
        let instruction: u32 = 0b00000000_00000000_00000000_00000111_u32;
        assert_eq!(7, parse_c(instruction, &Opcode::Err).unwrap())
    }

    #[test]
    fn test_parse_c_ortho() {
        let instruction: u32 = 0b00000000_00000000_00000000_00000000_u32;
        assert_eq!(None, parse_c(instruction, &Opcode::Ortho))
    }

    // parse_value

    #[test]
    fn test_parse_value() {
        let instruction: u32 = 0b00000001_11111111_11111111_11111111_u32;
        assert_eq!(None, parse_value(instruction, &Opcode::Err))
    }

    #[test]
    fn test_parse_value_ortho() {
        let instruction: u32 = 0b00000001_11111111_11111111_11111111_u32;
        assert_eq!(
            0b1_11111111_11111111_11111111_u32,
            parse_value(instruction, &Opcode::Ortho).unwrap()
        )
    }
}
