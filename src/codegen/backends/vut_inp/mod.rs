use std::io::Write;

use miette::{IntoDiagnostic, Result, SourceSpan};

mod emit;
mod lower;

use crate::{
    codegen::{CodegenBackend, TargetAssembly, TargetInfo},
    diagnostics::{Diag, DiagCtx},
    ir::{IRModule, Width},
};

pub struct INPAssembly {
    pub program: String,
    pub data: Vec<u8>,
}

impl TargetAssembly for INPAssembly {
    fn emit_asm<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(self.program.as_bytes()).into_diagnostic()?;
        w.write_all(b"@").into_diagnostic()?;
        w.write_all(&self.data).into_diagnostic()?;
        Ok(())
    }
}

pub struct INPCodegen;

impl CodegenBackend for INPCodegen {
    type Assembly = INPAssembly;

    fn compile_module(
        &self,
        module: IRModule,
        _target: TargetInfo,
        ctx: &mut DiagCtx,
    ) -> Option<Self::Assembly> {
        validate_module(&module, ctx);

        if ctx.has_errors() {
            return None;
        }

        let Some(func) = module.functions.first() else {
            return Some(INPAssembly {
                program: String::new(),
                data: Vec::new(),
            });
        };
        let lir = lower::lower(func, ctx);
        if ctx.has_errors() {
            return None;
        }

        emit::emit(&lir, &module.globals, ctx)
    }
}

fn validate_module(module: &IRModule, ctx: &mut DiagCtx) {
    for ext in &module.externs {
        if ext.variadic {
            ctx.emit(Diag::error(
                "variadic functions not supported",
                ext.span.unwrap_or_else(|| (0, 0).into()),
                "variadic functions are not supported at this time",
            ));
        }
        if ext.name == "getchar" {
            if !ext.args.is_empty() {
                ctx.emit(Diag::error(
                    "incorrect argument count",
                    ext.span.unwrap_or_else(|| (0, 0).into()),
                    "`getchar` expects no arguments",
                ));
            }
            if !matches!(ext.ret, Some(Width::Width(8))) {
                ctx.emit(Diag::error(
                    "incorrect return type",
                    ext.span.unwrap_or_else(|| (0, 0).into()),
                    "`getchar` returns an 8-bit integer (`i8`)",
                ));
            }
        } else if ext.name == "putchar" {
            if ext.args.len() != 1 {
                ctx.emit(Diag::error(
                    "incorrect argument count",
                    ext.span.unwrap_or_else(|| (0, 0).into()),
                    "`putchar` expects exactly 1 argument",
                ));
            } else if ext.args[0] != Width::Width(8) {
                ctx.emit(Diag::error(
                    "incorrect argument type",
                    ext.span.unwrap_or_else(|| (0, 0).into()),
                    "`putchar` takes a single 8-bit integer (`i8`)",
                ));
            }
            if ext.ret.is_some() {
                ctx.emit(Diag::error(
                    "incorrect return type",
                    ext.span.unwrap_or_else(|| (0, 0).into()),
                    "`putchar` returns no value",
                ));
            }
        } else {
            ctx.emit(Diag::error(
                "unsupported external function",
                ext.span.unwrap_or_else(|| (0, 0).into()),
                "the INP ABI only defines `getchar` and `putchar`",
            ));
        }
    }

    for global in &module.globals {
        if global.export {
            ctx.emit(Diag::warning(
                "meaningless export",
                global.span.unwrap_or_else(|| (0, 0).into()),
                "there will be no other programs to export to",
            ));
        }
    }

    let validate_type = |w: Width, span: Option<SourceSpan>, ctx: &mut DiagCtx| {
        if !matches!(w, Width::Ptr | Width::Width(8)) {
            ctx.emit(Diag::error(
                "unsupported type",
                span.unwrap_or_else(|| (0, 0).into()),
                "only pointers and 8-bit integers (`i8`) are supported at this time",
            ));
        }
    };

    for (i, func) in module.functions.iter().enumerate() {
        if i > 0 {
            ctx.emit(Diag::error(
                "too many functions",
                func.span.unwrap_or_else(|| (0, 0).into()),
                "only defining a single function is supported at this time",
            ));
            break;
        }
        for arg in &func.params {
            validate_type(arg.width, arg.span, ctx);
        }
        if let Some(ret) = func.ret {
            validate_type(ret, func.span, ctx);
            ctx.emit(Diag::warning(
                "meaningless return",
                func.span.unwrap_or_else(|| (0, 0).into()),
                "returning a value does nothing",
            ));
        }
        for b in &func.blocks {
            for i in &b.instructions {
                if let Some(ret) = &i.meta.result {
                    validate_type(ret.width, ret.span, ctx);
                }
            }
        }
    }
}
