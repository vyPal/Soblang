use std::io::Write;

use miette::Result;

use crate::{diagnostics::DiagCtx, ir::IRModule};

pub mod backends;

pub struct TargetInfo {
    pub triple: TargetTriple,
}

pub struct TargetTriple {
    pub arch: String,
    pub vendor: String,
    pub sys: String,
    pub abi: Option<String>,
}

pub trait TargetAssembly {
    fn emit_asm<W: Write>(&self, writer: &mut W) -> Result<()>;
}

pub trait CodegenBackend {
    type Assembly: TargetAssembly;

    fn compile_module(
        &self,
        module: IRModule,
        target: TargetInfo,
        ctx: &mut DiagCtx,
    ) -> Option<Self::Assembly>;
}
