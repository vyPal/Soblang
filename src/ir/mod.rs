use std::str::FromStr;

use miette::SourceSpan;

pub mod parser;

// Value and address sizes, to allow for different sizes in the future
pub type Imm = i64;
pub type Value = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Width {
    Width(u16),
    Ptr,
}

pub struct IRModule {
    pub functions: Vec<IRFunction>,
    pub externs: Vec<IRExtern>,
    pub globals: Vec<IRGlobal>,
}

pub struct IRGlobal {
    pub name: String,
    pub init: GlobalInit,
    pub align: u32,
    pub export: bool,
    pub mutable: bool,
}

pub enum GlobalInit {
    Bytes(Vec<u8>),
    Zeroed(u32),
}

pub struct IRExtern {
    pub name: String,
    pub args: Vec<Width>,
    pub ret: Option<Width>,
    pub variadic: bool,
}

pub struct IRFunction {
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Option<Width>,
    pub blocks: Vec<IRBlock>,
}

pub struct Param {
    pub id: Value,
    pub width: Width,
    pub span: Option<SourceSpan>,
}
pub type IRResult = Param;

pub struct IRBlock {
    pub label: String,
    pub instructions: Vec<IRInst>,
}

pub struct IRInst {
    pub op: IROp,
    pub flags: IRInstFlags,
    pub meta: IRInstMeta,
}

// Flags related to specific instruction, could be used in future optimizatinos, maybe should be
// constant?
pub struct IRInstFlags {}

// Instruction metadata, can be used by compiler to pass info about origin of instruction for
// debugging and such
pub struct IRInstMeta {
    pub result: Option<IRResult>,
    pub span: Option<SourceSpan>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[rustfmt::skip]
pub enum ICmpKind {
    Eq, Ne,
    SLt, SLe, SGt, SGe,
    ULt, ULe, UGt, UGe,
}

#[rustfmt::skip]
impl FromStr for ICmpKind {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "eq" => Self::Eq, "ne" => Self::Ne,
            "slt" => Self::SLt, "sle" => Self::SLe, "sgt" => Self::SGt, "sge" => Self::SGe,
            "ult" => Self::ULt, "ule" => Self::ULe, "ugt" => Self::UGt, "uge" => Self::UGe,
            _ => return Err(())
        })
    }
}

pub enum IROp {
    // Constants
    Const(Imm),

    // Integer arithmetic
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    UDiv(Value, Value),
    SDiv(Value, Value),
    URem(Value, Value),
    SRem(Value, Value),

    // Bitwise
    And(Value, Value),
    Or(Value, Value),
    Xor(Value, Value),
    Shl(Value, Value),
    AShr(Value, Value),
    LShr(Value, Value),

    // Conversions
    ZExt(Value),
    SExt(Value),
    Trunc(Value),

    // Control flow
    Jmp(String),
    ICmp(ICmpKind, Value, Value),
    Br(Value, String, String),

    // Function calls
    Call(String, Vec<Value>),
    Ret(Option<Value>),

    PtrAdd(Value, Value),
    GlobalAddr(String),

    // Memory allocation
    Alloca(u32),
    Load(Value, u32),
    Store(Value, u32, Value),
}
