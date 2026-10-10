use std::io;

use miette::Result;

use crate::{
    codegen::{
        CodegenBackend, TargetAssembly, TargetInfo, TargetTriple, backends::vut_inp::INPCodegen,
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
extern fn getchar() -> i8
extern fn putchar(i8)

; offsets:  0 prompt 1 | 16 prompt 2 | 33 result line (x@34 y@38 q@42 r@52) | 55 div-by-zero
global @text align 1 = "zadej cislo 1: \0\nzadej cislo 2: \0\nx / y = q, zbytek r\n\0\ndeleni nulou!\n\0"

fn main() {
entry:
    %0: i8 = const 0
    %1: i8 = const 1
    %2: i8 = const 2
    %3: i8 = const 4
    %4: i8 = const 10
    %5: i8 = const 16          ; prompt stride
    %6: i8 = const 33          ; result line
    %7: i8 = const 34          ; 'x' placeholder
    %8: i8 = const 38          ; 'y' placeholder
    %9: i8 = const 42          ; 'q' placeholder
    %10: i8 = const 52         ; 'r' placeholder
    %11: i8 = const 55         ; division by zero message
    %12: i8 = const 48         ; '0'
    %13: ptr = globaladdr @text
    %14: ptr = alloca 1        ; n: which number we're reading
    %15: ptr = alloca 1        ; print arg: index into @text
    %16: ptr = alloca 1        ; print return id: 0 -> read, 1 -> end
    store %14, 0, %0
    jmp ask

; --- ask for number n: print prompt n, then read ---
ask:
    %17: i8 = load %14, 0
    %18: i8 = mul %17, %5      ; prompt n starts at 16 * n
    store %15, 0, %18
    store %16, 0, %0           ; return to `read`
    jmp print

read:
    %19: i8 = call getchar()
    %20: i8 = sub %19, %12
    %21: i8 = icmp ult %20, %4 ; digit?
    br %21, got, read

got:
    %22: i8 = load %14, 0
    %23: i8 = mul %22, %3
    %24: i8 = add %23, %7      ; x/y placeholder at 34 + 4n
    %25: ptr = ptradd %13, %24
    store %25, 0, %19          ; write digit char into the template
    %26: i8 = add %22, %1
    store %14, 0, %26
    %27: i8 = icmp ult %26, %2
    br %27, ask, compute

; --- divide ---
compute:
    %28: ptr = ptradd %13, %7
    %29: i8 = load %28, 0
    %30: ptr = ptradd %13, %8
    %31: i8 = load %30, 0
    %32: i8 = sub %29, %12     ; a
    %33: i8 = sub %31, %12     ; b
    store %16, 0, %1           ; both paths print once more, then end
    %34: i8 = icmp eq %33, %0
    br %34, div_zero, divide

divide:
    %35: i8 = udiv %32, %33
    %36: i8 = urem %32, %33
    %37: i8 = add %35, %12
    %38: ptr = ptradd %13, %9
    store %38, 0, %37
    %39: i8 = add %36, %12
    %40: ptr = ptradd %13, %10
    store %40, 0, %39
    store %15, 0, %6
    jmp print

div_zero:
    store %15, 0, %11
    jmp print

; --- print(@text[idx..] up to \0), then return via ret id ---
print:
    %41: i8 = load %15, 0
    %42: ptr = ptradd %13, %41
    %43: i8 = load %42, 0
    %44: i8 = icmp eq %43, %0
    br %44, print_ret, print_put

print_put:
    call putchar(%43)
    %45: i8 = add %41, %1
    store %15, 0, %45
    jmp print

print_ret:
    %46: i8 = load %16, 0
    %47: i8 = icmp eq %46, %0
    br %47, read, end

end:
    ret
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

    if let Some(asm) = INPCodegen.compile_module(
        module,
        TargetInfo {
            triple: TargetTriple {
                arch: "INP2026".to_string(),
                vendor: "vyPal".to_string(),
                sys: "SobOS".to_string(),
                abi: None,
            },
        },
        &mut ctx,
    ) {
        let _ = asm.emit_asm(&mut io::stdout());
    }

    ctx.finish()
}
