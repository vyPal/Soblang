use std::io;

use miette::Result;

use crate::{
    codegen::{
        CodegenBackend, TargetAssembly, TargetInfo, TargetTriple, backends::x86::X86Codegen,
    },
    diagnostics::{CompilerDiagnostic, DiagCtx, DiagnosticReport},
    ir::parser::parse_module,
};

pub mod codegen;
pub mod diagnostics;
pub mod ir;

fn main() -> Result<()> {
    let warnings = compile(
        "test.sob",
        r#"
extern fn printf(ptr, ...) -> i32
extern fn atoi(ptr) -> i32

global @fmt align 1 = "fib(%d) = %d\n\0"
global @usage align 1 = "usage: %s <n>\n\0"

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

fn main(%0: i32, %1: ptr) -> i32 {
entry:
    %2: i32 = const 2
    %3: i32 = icmp slt %0, %2
    br %3, usage, run
usage:
    %4: ptr = load %1, 0
    %5: ptr = globaladdr @usage
    %6: i32 = call printf(%5, %4)
    %7: i32 = const 1
    ret %7
run:
    %8: ptr = load %1, 8
    %9: i32 = call atoi(%8)
    %10: i32 = call fib(%9)
    %11: ptr = globaladdr @fmt
    %12: i32 = call printf(%11, %9, %10)
    %13: i32 = const 0
    ret %13
}
    "#,
    )?;

    for w in warnings {
        eprintln!("{:?}", miette::Report::new(w));
    }

    Ok(())
}

fn compile(filename: &str, source_code: &str) -> Result<Vec<CompilerDiagnostic>, DiagnosticReport> {
    let mut ctx = DiagCtx::new(filename, source_code);

    let module = parse_module(source_code, &mut ctx);

    if ctx.has_errors() {
        return ctx.finish();
    }

    if let Some(asm) = X86Codegen.compile_module(
        module,
        TargetInfo {
            triple: TargetTriple {
                arch: "x86-64".to_string(),
                vendor: "unknown".to_string(),
                sys: "unknown".to_string(),
                abi: None,
            },
        },
        &mut ctx,
    ) {
        let _ = asm.emit_asm(&mut io::stdout());
    }

    ctx.finish()
}
