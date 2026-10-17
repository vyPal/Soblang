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

fn main() {
entry:
    %0: i8 = const 0
    %1: i8 = const 1
    %2: i8 = const 2
    %3: i8 = const 3
    %4: i8 = const 10          ; base 10, also '\n'
    %5: i8 = const 100
    %6: i8 = const 32          ; ' '
    %7: i8 = const 33          ; '!'
    %8: i8 = const 44          ; ','
    %9: i8 = const 47          ; '/'
    %10: i8 = const 48         ; '0'
    %11: i8 = const 49         ; '1'
    %12: i8 = const 58         ; ':'
    %13: i8 = const 61         ; '='
    %14: i8 = const 97         ; 'a'
    %15: i8 = const 98         ; 'b'
    %16: i8 = const 99         ; 'c'
    %17: i8 = const 100        ; 'd'
    %18: i8 = const 101        ; 'e'
    %19: i8 = const 105        ; 'i'
    %20: i8 = const 106        ; 'j'
    %21: i8 = const 107        ; 'k'
    %22: i8 = const 108        ; 'l'
    %23: i8 = const 110        ; 'n'
    %24: i8 = const 111        ; 'o'
    %25: i8 = const 115        ; 's'
    %26: i8 = const 116        ; 't'
    %27: i8 = const 117        ; 'u'
    %28: i8 = const 121        ; 'y'
    %29: i8 = const 122        ; 'z'
    %30: ptr = alloca 1        ; n: which number we're reading
    %31: ptr = alloca 1        ; a (previous number)
    %32: ptr = alloca 1        ; b (current number)
    %33: ptr = alloca 1        ; acc: number being read
    %34: ptr = alloca 1        ; seen: got at least one digit
    %35: ptr = alloca 1        ; print_num argument
    %36: ptr = alloca 1        ; print_num return id
    %37: ptr = alloca 1        ; quotient
    %38: ptr = alloca 1        ; remainder
    store %30, 0, %0
    store %32, 0, %0
    jmp ask

; --- "zadej cislo <n+1>: ", then read a number ---
ask:
    call putchar(%29)          ; z
    call putchar(%14)          ; a
    call putchar(%17)          ; d
    call putchar(%18)          ; e
    call putchar(%20)          ; j
    call putchar(%6)
    call putchar(%16)          ; c
    call putchar(%19)          ; i
    call putchar(%25)          ; s
    call putchar(%22)          ; l
    call putchar(%24)          ; o
    call putchar(%6)
    %40: i8 = load %30, 0
    %41: i8 = add %40, %11     ; '1' + n
    call putchar(%41)
    call putchar(%12)          ; :
    call putchar(%6)
    store %33, 0, %0
    store %34, 0, %0
    jmp read

; digits accumulate into acc; the first non-digit after a digit ends the number,
; non-digits before it are skipped
read:
    %42: i8 = call getchar()
    %43: i8 = sub %42, %10
    %44: i8 = icmp ult %43, %4
    br %44, digit, non_digit

digit:
    %45: i8 = load %33, 0
    %46: i8 = mul %45, %4      ; acc = acc * 10 + d
    %47: i8 = add %46, %43
    store %33, 0, %47
    store %34, 0, %1
    jmp read

non_digit:
    %48: i8 = load %34, 0
    br %48, got, read

got:
    call putchar(%4)           ; '\n'
    %49: i8 = load %32, 0      ; a = b, b = acc
    store %31, 0, %49
    %50: i8 = load %33, 0
    store %32, 0, %50
    %51: i8 = load %30, 0
    %52: i8 = add %51, %1
    store %30, 0, %52
    %53: i8 = icmp ult %52, %2
    br %53, ask, compute

compute:
    %54: i8 = load %31, 0
    %55: i8 = load %32, 0
    %56: i8 = icmp eq %55, %0
    br %56, div_zero, divide

divide:
    %57: i8 = udiv %54, %55
    %58: i8 = urem %54, %55
    store %37, 0, %57
    store %38, 0, %58
    store %35, 0, %54          ; print a
    store %36, 0, %0
    jmp print_num

; --- "a / b = q, zbytek r\n", one print_num call per number ---
ret0:
    call putchar(%6)
    call putchar(%9)           ; /
    call putchar(%6)
    %59: i8 = load %32, 0      ; print b
    store %35, 0, %59
    store %36, 0, %1
    jmp print_num

ret1:
    call putchar(%6)
    call putchar(%13)          ; =
    call putchar(%6)
    %60: i8 = load %37, 0      ; print q
    store %35, 0, %60
    store %36, 0, %2
    jmp print_num

ret2:
    call putchar(%8)           ; ,
    call putchar(%6)
    call putchar(%29)          ; z
    call putchar(%15)          ; b
    call putchar(%28)          ; y
    call putchar(%26)          ; t
    call putchar(%18)          ; e
    call putchar(%21)          ; k
    call putchar(%6)
    %61: i8 = load %38, 0      ; print r
    store %35, 0, %61
    store %36, 0, %3
    jmp print_num

ret3:
    call putchar(%4)           ; '\n'
    ret

; --- print_num: decimal without leading zeros, then return by id ---
print_num:
    %62: i8 = load %35, 0
    %63: i8 = udiv %62, %5     ; hundreds
    %64: i8 = urem %62, %5
    %65: i8 = udiv %64, %4     ; tens
    %66: i8 = urem %64, %4     ; ones
    br %63, print_h, check_t

print_h:
    %67: i8 = add %63, %10
    call putchar(%67)
    jmp print_t                ; after a hundreds digit, tens always print

check_t:
    br %65, print_t, print_o

print_t:
    %68: i8 = add %65, %10
    call putchar(%68)
    jmp print_o

print_o:
    %69: i8 = add %66, %10
    call putchar(%69)
    %70: i8 = load %36, 0
    %71: i8 = icmp eq %70, %0
    br %71, ret0, ret_not0

ret_not0:
    %72: i8 = icmp eq %70, %1
    br %72, ret1, ret_not1

ret_not1:
    %73: i8 = icmp eq %70, %2
    br %73, ret2, ret3

div_zero:
    call putchar(%17)          ; d
    call putchar(%18)          ; e
    call putchar(%22)          ; l
    call putchar(%18)          ; e
    call putchar(%23)          ; n
    call putchar(%19)          ; i
    call putchar(%6)
    call putchar(%23)          ; n
    call putchar(%27)          ; u
    call putchar(%22)          ; l
    call putchar(%24)          ; o
    call putchar(%27)          ; u
    call putchar(%7)           ; !
    call putchar(%4)           ; '\n'
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
