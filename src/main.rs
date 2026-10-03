use std::io;

use miette::Result;

use crate::{
    codegen::{
        CodegenBackend, TargetAssembly, TargetInfo, TargetTriple, backends::x86::X86Codegen,
    },
    ir::parser::parse_module,
};

pub mod codegen;
pub mod ir;

fn main() -> Result<()> {
    let module = parse_module(
        "test.sob",
        r#"
fn fib(%0: i32) -> i32 {
entry:
    %1: i32 = const 0
    %2: i32 = icmp sle %0, %1
    br %2, iszero, notzero
iszero:
    ret %1
notzero:
    %3: i32 = const 1
    %4: i32 = icmp eq %0, %3
    br %4, isone, notone
isone:
    ret %3
notone:
    %5: i32 = sub %0, %3
    %6: i32 = call fib(%5)
    %7: i32 = sub %5, %3
    %8: i32 = call fib(%7)
    %9: i32 = add %6, %8
    ret %9
}

fn main() -> i32 {
entry:
    %0: i32 = const 6
    %1: i32 = call fib(%0)
    ret %1
}
    "#,
    )?;

    let x86 = X86Codegen {};
    let asm = x86.compile_module(
        module,
        TargetInfo {
            triple: TargetTriple {
                arch: "x86-64".to_string(),
                vendor: "unknown".to_string(),
                sys: "unknown".to_string(),
                abi: None,
            },
        },
    )?;

    asm.emit_asm(&mut io::stdout())?;
    Ok(())
}
