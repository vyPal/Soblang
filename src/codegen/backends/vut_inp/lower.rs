use std::collections::{HashMap, HashSet};

use miette::SourceSpan;

use crate::{
    diagnostics::{Diag, DiagCtx},
    ir::{ICmpKind, IRBlock, IRFunction, IROp, Value},
};

pub type BlockId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operand {
    V(Value),
    I(u8),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Base {
    Global(String),
    Stack(Value),
    Abs,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Addr {
    pub base: Base,
    pub off: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LOp {
    Add(Value, Operand),
    Sub(Operand, Value), // rewrite x-c as Add(x, -c)
    Mul(Value, Operand),
    DivMod {
        n: Operand,
        d: Operand,
        q: Option<Value>,
        r: Option<Value>,
    },
    Lt(Operand, Operand), // unsigned, 0/1
    IsZero(Value),        // 0/1
    And(Value, Operand),
    Or(Value, Operand),
    Xor(Value, Operand),
    Shl(Operand, Value), // non-constant
    LShr(Operand, Value),
    Load(Addr),
    Store(Addr, Operand),
    LoadIdx(Addr, Value),
    StoreIdx(Addr, Value, Operand),
    AddrOf(Addr),
    PtrAdd(Value, Operand),
    PtrAddImm(Value, i64),
    LoadDyn(Value, i64),
    StoreDyn(Value, i64, Operand),
    GetChar,
    PutChar(Operand),
}

impl LOp {
    pub fn uses(&self) -> impl Iterator<Item = Value> {
        use LOp::*;
        use Operand::V;
        let ops: [Option<Operand>; 2] = match self {
            Add(a, b) | Mul(a, b) | And(a, b) | Or(a, b) | Xor(a, b) | PtrAdd(a, b) => {
                [Some(V(*a)), Some(*b)]
            }
            Sub(a, b) | Shl(a, b) | LShr(a, b) => [Some(*a), Some(V(*b))],
            DivMod { n, d, .. } => [Some(*n), Some(*d)],
            Lt(a, b) => [Some(*a), Some(*b)],
            IsZero(a) | PtrAddImm(a, _) | LoadDyn(a, _) | LoadIdx(_, a) => [Some(V(*a)), None],
            StoreDyn(p, _, v) | StoreIdx(_, p, v) => [Some(V(*p)), Some(*v)],
            Store(_, v) | PutChar(v) => [Some(*v), None],
            Load(_) | AddrOf(_) | GetChar => [None, None],
        };
        ops.into_iter().flatten().filter_map(|o| match o {
            V(v) => Some(v),
            Operand::I(_) => None,
        })
    }

    fn removable(&self) -> bool {
        !matches!(
            self,
            LOp::Store(..) | LOp::StoreDyn(..) | LOp::StoreIdx(..) | LOp::GetChar | LOp::PutChar(_)
        )
    }

    fn cse_ok(&self) -> bool {
        self.removable()
            && !matches!(
                self,
                LOp::Load(_) | LOp::LoadDyn(..) | LOp::LoadIdx(..) | LOp::DivMod { .. }
            )
    }
}

#[derive(Debug)]
pub struct LInst {
    pub dst: Option<Value>,
    pub op: LOp,
    pub span: Option<SourceSpan>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Term {
    Jmp(BlockId),
    Br(Value, BlockId, BlockId), // non-zero = first
    Halt,
}

impl Term {
    pub fn succs(&self) -> Vec<BlockId> {
        match *self {
            Term::Jmp(t) => vec![t],
            Term::Br(_, t, e) => vec![t, e],
            Term::Halt => vec![],
        }
    }

    fn remap(self, m: &[BlockId]) -> Self {
        match self {
            Term::Jmp(t) => Term::Jmp(m[t]),
            Term::Br(c, t, e) => Term::Br(c, m[t], m[e]),
            Term::Halt => Term::Halt,
        }
    }
}

#[derive(Debug)]
pub struct LBlock {
    pub insts: Vec<LInst>,
    pub term: Term,
}

#[derive(Debug)]
pub struct LFunc {
    pub blocks: Vec<LBlock>,      // [0] is entry
    pub stack: Vec<(Value, u32)>, // live allocas, (id, size in cells)
}

pub fn lower(func: &IRFunction, ctx: &mut DiagCtx) -> LFunc {
    let labels: HashMap<&str, BlockId> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.label.as_str(), i))
        .collect();
    let next = func
        .params
        .iter()
        .map(|p| p.id)
        .chain(
            func.blocks
                .iter()
                .flat_map(|b| &b.instructions)
                .filter_map(|i| i.meta.result.as_ref().map(|r| r.id)),
        )
        .max()
        .map_or(0, |m| m + 1);

    let mut l = Lower {
        ctx,
        known: HashMap::new(),
        next,
        insts: Vec::new(),
        lvn: HashMap::new(),
        span: None,
        stack: Vec::new(),
    };

    let mut blocks: Vec<LBlock> = (0..func.blocks.len())
        .map(|_| LBlock {
            insts: Vec::new(),
            term: Term::Halt,
        })
        .collect();

    if !func.blocks.is_empty() {
        let succs: Vec<_> = func.blocks.iter().map(|b| ir_succs(b, &labels)).collect();
        for b in rpo(&succs) {
            blocks[b] = l.block(&func.blocks[b], &labels);
        }
    }

    let mut f = LFunc {
        blocks,
        stack: l.stack,
        //params: func.params.iter().map(|p| p.id).collect(),
    };
    while dce(&mut f) | simplify_cfg(&mut f) {}
    prune_stack(&mut f);
    f
}

#[derive(Debug, Clone)]
enum Known {
    Imm(i64),
    Val(Value),
    Addr(Addr),
    Dyn(Value, i64), // rt ptr + offset
    Not(Value),
    Truthy(Value),
    Idx(Addr, Value),
}

fn known(o: Operand) -> Known {
    match o {
        Operand::I(c) => Known::Imm(c as i64),
        Operand::V(v) => Known::Val(v),
    }
}

#[derive(Debug, Clone, Copy)]
enum Bit {
    And,
    Or,
    Xor,
}
#[derive(Debug, Clone, Copy)]
enum Sh {
    Shl,
    LShr,
    AShr,
}

enum Mem {
    Static(Addr),
    Idx(Addr, Value),
    Dyn(Value, i64),
}

struct Lower<'c> {
    ctx: &'c mut DiagCtx,
    known: HashMap<Value, Known>,
    next: Value,
    insts: Vec<LInst>,
    lvn: HashMap<LOp, Value>, // per-block cse
    span: Option<SourceSpan>,
    stack: Vec<(Value, u32)>,
}

impl Lower<'_> {
    fn fresh(&mut self) -> Value {
        self.next += 1;
        self.next - 1
    }

    fn emit(&mut self, op: LOp) -> Value {
        if let Some(&v) = self.lvn.get(&op) {
            return v;
        }
        let v = self.fresh();
        if op.cse_ok() {
            self.lvn.insert(op.clone(), v);
        }
        self.insts.push(LInst {
            dst: Some(v),
            op,
            span: self.span,
        });
        v
    }

    fn effect(&mut self, op: LOp) {
        self.insts.push(LInst {
            dst: None,
            op,
            span: self.span,
        });
    }

    fn unsupported(&mut self, msg: &'static str) {
        let span = self.span.unwrap_or_else(|| (0, 0).into());
        self.ctx
            .emit(Diag::error("unsupported operation", span, msg));
    }

    fn get(&self, v: Value) -> Known {
        self.known.get(&v).cloned().unwrap_or(Known::Val(v))
    }

    fn mat(&mut self, k: Known) -> Operand {
        use Operand::*;
        match k {
            Known::Imm(c) => I(c as u8),
            Known::Val(v) => V(v),
            Known::Not(x) => V(self.emit(LOp::IsZero(x))),
            Known::Truthy(x) => V(self.emit(LOp::Lt(I(0), V(x)))),
            Known::Addr(a) => V(self.emit(LOp::AddrOf(a))),
            Known::Dyn(p, 0) => V(p),
            Known::Dyn(p, o) => V(self.emit(LOp::PtrAddImm(p, o))),
            Known::Idx(p, i) => {
                let p = self.emit(LOp::AddrOf(p));
                V(self.emit(LOp::PtrAdd(p, V(i))))
            }
        }
    }

    fn op(&mut self, v: Value) -> Operand {
        let k = self.get(v);
        self.mat(k)
    }

    fn ops(&mut self, a: Value, b: Value) -> (Operand, Operand) {
        (self.op(a), self.op(b))
    }

    fn ptr_val(&mut self, k: Known) -> Value {
        match k {
            Known::Imm(c) => self.emit(LOp::AddrOf(Addr {
                base: Base::Abs,
                off: c,
            })),
            k => match self.mat(k) {
                Operand::V(v) => v,
                Operand::I(_) => unreachable!(),
            },
        }
    }

    fn add(&mut self, a: Operand, b: Operand) -> Known {
        use Operand::*;
        match (a, b) {
            (I(x), I(y)) => Known::Imm(x.wrapping_add(y) as i64),
            (V(v), I(0)) | (I(0), V(v)) => Known::Val(v),
            (V(v), I(c)) | (I(c), V(v)) => Known::Val(self.emit(LOp::Add(v, I(c)))),
            (V(x), V(y)) => Known::Val(self.emit(LOp::Add(x.min(y), V(x.max(y))))),
        }
    }

    fn sub(&mut self, a: Operand, b: Operand) -> Known {
        use Operand::*;
        match (a, b) {
            (I(x), I(y)) => Known::Imm(x.wrapping_sub(y) as i64),
            (_, I(c)) => self.add(a, I(c.wrapping_neg())),
            (V(x), V(y)) if x == y => Known::Imm(0),
            (a, V(y)) => Known::Val(self.emit(LOp::Sub(a, y))),
        }
    }

    // Only valid for 0 tests
    fn diff(&mut self, a: Operand, b: Operand) -> Known {
        match (a, b) {
            (Operand::I(_), Operand::V(_)) => self.sub(b, a),
            _ => self.sub(a, b),
        }
    }

    fn mul(&mut self, a: Operand, b: Operand) -> Known {
        use Operand::*;
        match (a, b) {
            (I(x), I(y)) => Known::Imm(x.wrapping_mul(y) as i64),
            (_, I(0)) | (I(0), _) => Known::Imm(0),
            (V(v), I(1)) | (I(1), V(v)) => Known::Val(v),
            (V(v), I(c)) | (I(c), V(v)) => Known::Val(self.emit(LOp::Mul(v, I(c)))),
            (V(x), V(y)) => Known::Val(self.emit(LOp::Mul(x.min(y), V(x.max(y))))),
        }
    }

    fn divmod(&mut self, n: Operand, d: Operand, rem: bool) -> Known {
        use Operand::*;
        match (n, d) {
            (_, I(0)) => {
                let span = self.span.unwrap_or_else(|| (0, 0).into());
                self.ctx.emit(Diag::error(
                    "division by zero",
                    span,
                    "the divisor is always 0",
                ));
                Known::Imm(0)
            }
            (I(x), I(y)) => Known::Imm((if rem { x % y } else { x / y }) as i64),
            (I(0), _) => Known::Imm(0),
            (V(v), I(1)) => {
                if rem {
                    Known::Imm(0)
                } else {
                    Known::Val(v)
                }
            }
            _ => {
                let v = self.fresh();
                for inst in &mut self.insts {
                    if let LOp::DivMod { n: n2, d: d2, q, r } = &mut inst.op
                        && (*n2, *d2) == (n, d)
                    {
                        let slot = if rem { r } else { q };
                        return Known::Val(*slot.get_or_insert(v));
                    }
                }
                let (q, r) = if rem {
                    (None, Some(v))
                } else {
                    (Some(v), None)
                };
                self.insts.push(LInst {
                    dst: None,
                    op: LOp::DivMod { n, d, q, r },
                    span: self.span,
                });
                Known::Val(v)
            }
        }
    }

    fn bit(&mut self, kind: Bit, a: Operand, b: Operand) -> Known {
        use Operand::*;
        let (a, b) = if let (I(_), V(_)) = (a, b) {
            (b, a)
        } else {
            (a, b)
        };
        match (kind, a, b) {
            (_, I(x), I(y)) => Known::Imm(
                (match kind {
                    Bit::And => x & y,
                    Bit::Or => x | y,
                    Bit::Xor => x ^ y,
                }) as i64,
            ),
            (Bit::And, _, I(0)) => Known::Imm(0),
            (Bit::Or, _, I(255)) => Known::Imm(255),
            (Bit::And, a, I(255)) | (Bit::Or | Bit::Xor, a, I(0)) => known(a),
            (Bit::And, a, I(m)) if (m + 1).is_power_of_two() => self.divmod(a, I(m + 1), true),
            (Bit::Xor, a, I(128)) => self.add(a, I(128)),
            (Bit::Xor, a, I(255)) => self.sub(I(255), a),
            (Bit::And | Bit::Or, V(x), V(y)) if x == y => Known::Val(x),
            (Bit::Xor, V(x), V(y)) if x == y => Known::Imm(0),
            (_, V(x), b) => {
                let (x, b) = match b {
                    V(y) if y < x => (y, V(x)),
                    _ => (x, b),
                };
                Known::Val(self.emit(match kind {
                    Bit::And => LOp::And(x, b),
                    Bit::Or => LOp::Or(x, b),
                    Bit::Xor => LOp::Xor(x, b),
                }))
            }
            (_, I(_), V(_)) => unreachable!(),
        }
    }

    fn shift(&mut self, kind: Sh, a: Operand, b: Operand) -> Known {
        use Operand::*;
        match (kind, a, b) {
            (_, a, I(0)) => known(a),
            (Sh::Shl | Sh::LShr, I(0), _) => Known::Imm(0),
            (Sh::Shl | Sh::LShr, _, I(k)) if k >= 8 => Known::Imm(0),
            (Sh::Shl, a, I(k)) => self.mul(a, I(1 << k)),
            (Sh::LShr, a, I(k)) => self.divmod(a, I(1 << k), false),
            (Sh::AShr, I(x), I(k)) => Known::Imm(((x as i8) >> k.min(7)) as u8 as i64),
            (Sh::AShr, V(v), I(k)) => {
                // floor(s / 2^k) == floor((s + 128) / 2^k) - 128 / 2^k
                let k = k.min(7);
                let biased = self.add(V(v), I(128));
                let biased = self.mat(biased);
                let q = self.divmod(biased, I(1 << k), false);
                let q = self.mat(q);
                self.sub(q, I(128 >> k))
            }
            (Sh::Shl, a, V(y)) => Known::Val(self.emit(LOp::Shl(a, y))),
            (Sh::LShr, a, V(y)) => Known::Val(self.emit(LOp::LShr(a, y))),
            (Sh::AShr, _, V(_)) => {
                self.unsupported("arithmetic shift by a non-constant value is not supported yet");
                Known::Imm(0)
            }
        }
    }

    fn lt(&mut self, a: Operand, b: Operand) -> Known {
        use Operand::*;
        match (a, b) {
            (I(x), I(y)) => Known::Imm((x < y) as i64),
            (_, I(0)) | (I(255), _) => Known::Imm(0),
            (V(x), V(y)) if x == y => Known::Imm(0),
            (I(0), V(y)) => Known::Truthy(y),
            (V(x), I(1)) => Known::Not(x),
            _ => Known::Val(self.emit(LOp::Lt(a, b))),
        }
    }

    fn not(&mut self, k: Known) -> Known {
        match k {
            Known::Imm(c) => Known::Imm((c as u8 == 0) as i64),
            Known::Val(x) | Known::Truthy(x) => Known::Not(x),
            Known::Not(x) => Known::Truthy(x),
            k => {
                let o = self.mat(k);
                self.not(known(o))
            }
        }
    }

    fn icmp(&mut self, kind: ICmpKind, a: Operand, b: Operand) -> Known {
        use ICmpKind as K;
        let (a, b) = if matches!(kind, K::SLt | K::SLe | K::SGt | K::SGe) {
            // a <s b <=> (a ^ 0x80) <u (b ^ 0x80), and ^0x80 == +128 on 8 bits
            let fa = self.add(a, Operand::I(128));
            let fb = self.add(b, Operand::I(128));
            (self.mat(fa), self.mat(fb))
        } else {
            (a, b)
        };
        match kind {
            K::Eq => {
                let d = self.diff(a, b);
                self.not(d)
            }
            K::Ne => {
                let d = self.diff(a, b);
                let z = self.not(d);
                self.not(z)
            }
            K::ULt | K::SLt => self.lt(a, b),
            K::UGt | K::SGt => self.lt(b, a),
            K::ULe | K::SLe => {
                let r = self.lt(b, a);
                self.not(r)
            }
            K::UGe | K::SGe => {
                let r = self.lt(a, b);
                self.not(r)
            }
        }
    }

    fn ptr_add(&mut self, p: Value, o: Value) -> Known {
        match (self.get(p), self.get(o)) {
            (k, Known::Imm(0)) => k,
            (Known::Imm(b), Known::Imm(c)) => Known::Addr(Addr {
                base: Base::Abs,
                off: b + c,
            }),
            (Known::Dyn(q, d), Known::Imm(c)) => Known::Dyn(q, d + c),
            (Known::Val(q), Known::Imm(c)) => Known::Dyn(q, c),
            (Known::Addr(a), ok) => match self.mat(ok) {
                Operand::V(i) => Known::Idx(a, i),
                Operand::I(c) => Known::Addr(Addr {
                    off: a.off + c as i64,
                    ..a
                }),
            },
            (pk, ok) => {
                let p = self.ptr_val(pk);
                let o = self.mat(ok);
                Known::Val(self.emit(LOp::PtrAdd(p, o)))
            }
        }
    }

    fn mem(&mut self, p: Value, off: u32) -> Mem {
        let off = off as i64;
        match self.get(p) {
            Known::Addr(a) => Mem::Static(Addr {
                off: a.off + off,
                ..a
            }),
            Known::Imm(c) => Mem::Static(Addr {
                base: Base::Abs,
                off: c + off,
            }),
            Known::Dyn(q, o) => Mem::Dyn(q, o + off),
            Known::Idx(a, i) => Mem::Idx(
                Addr {
                    off: a.off + off,
                    ..a
                },
                i,
            ),
            k => Mem::Dyn(self.ptr_val(k), off),
        }
    }

    fn branch(&mut self, c: Value, t: BlockId, e: BlockId) -> Term {
        match self.get(c) {
            Known::Imm(x) => Term::Jmp(if x as u8 != 0 { t } else { e }),
            Known::Not(x) => Term::Br(x, e, t),
            Known::Truthy(x) | Known::Val(x) => Term::Br(x, t, e),
            k => match self.mat(k) {
                Operand::V(x) => Term::Br(x, t, e),
                Operand::I(_) => unreachable!(),
            },
        }
    }

    fn finish(&mut self, term: Term) -> LBlock {
        LBlock {
            insts: std::mem::take(&mut self.insts),
            term,
        }
    }

    fn block(&mut self, blk: &IRBlock, labels: &HashMap<&str, BlockId>) -> LBlock {
        self.lvn.clear();
        for inst in &blk.instructions {
            self.span = inst.meta.span;
            let dst = inst.meta.result.as_ref().map(|r| r.id);
            let k = match &inst.op {
                IROp::Const(c) => Some(Known::Imm(*c)),
                IROp::Add(a, b) => {
                    let (a, b) = self.ops(*a, *b);
                    Some(self.add(a, b))
                }
                IROp::Sub(a, b) => {
                    let (a, b) = self.ops(*a, *b);
                    Some(self.sub(a, b))
                }
                IROp::Mul(a, b) => {
                    let (a, b) = self.ops(*a, *b);
                    Some(self.mul(a, b))
                }
                IROp::UDiv(a, b) | IROp::URem(a, b) => {
                    let rem = matches!(inst.op, IROp::URem(..));
                    let (a, b) = self.ops(*a, *b);
                    Some(self.divmod(a, b, rem))
                }
                IROp::SDiv(..) | IROp::SRem(..) => {
                    self.unsupported("signed division/remainder is not supported yet");
                    None
                }
                IROp::And(a, b) | IROp::Or(a, b) | IROp::Xor(a, b) => {
                    let kind = match inst.op {
                        IROp::And(..) => Bit::And,
                        IROp::Or(..) => Bit::Or,
                        _ => Bit::Xor,
                    };
                    let (a, b) = self.ops(*a, *b);
                    Some(self.bit(kind, a, b))
                }
                IROp::Shl(a, b) | IROp::LShr(a, b) | IROp::AShr(a, b) => {
                    let kind = match inst.op {
                        IROp::Shl(..) => Sh::Shl,
                        IROp::LShr(..) => Sh::LShr,
                        _ => Sh::AShr,
                    };
                    let (a, b) = self.ops(*a, *b);
                    Some(self.shift(kind, a, b))
                }
                IROp::ZExt(v) | IROp::SExt(v) | IROp::Trunc(v) => Some(self.get(*v)),
                IROp::ICmp(kind, a, b) => {
                    let (a, b) = self.ops(*a, *b);
                    Some(self.icmp(*kind, a, b))
                }
                IROp::Call(name, args) => match name.as_str() {
                    "getchar" => Some(Known::Val(self.emit(LOp::GetChar))),
                    "putchar" => {
                        let c = self.op(args[0]);
                        self.effect(LOp::PutChar(c));
                        None
                    }
                    _ => None,
                },
                IROp::PtrAdd(p, o) => Some(self.ptr_add(*p, *o)),
                IROp::GlobalAddr(name) => Some(Known::Addr(Addr {
                    base: Base::Global(name.clone()),
                    off: 0,
                })),
                IROp::Alloca(size) => dst.map(|d| {
                    self.stack.push((d, *size));
                    Known::Addr(Addr {
                        base: Base::Stack(d),
                        off: 0,
                    })
                }),
                IROp::Load(p, off) => Some(Known::Val(match self.mem(*p, *off) {
                    Mem::Static(a) => self.emit(LOp::Load(a)),
                    Mem::Idx(a, i) => self.emit(LOp::LoadIdx(a, i)),
                    Mem::Dyn(q, o) => self.emit(LOp::LoadDyn(q, o)),
                })),
                IROp::Store(p, off, v) => {
                    let v = self.op(*v);
                    let op = match self.mem(*p, *off) {
                        Mem::Static(a) => LOp::Store(a, v),
                        Mem::Idx(a, i) => LOp::StoreIdx(a, i, v),
                        Mem::Dyn(q, o) => LOp::StoreDyn(q, o, v),
                    };
                    self.effect(op);
                    None
                }
                IROp::Jmp(l) => return self.finish(Term::Jmp(labels[l.as_str()])),
                IROp::Br(c, t, e) => {
                    let term = self.branch(*c, labels[t.as_str()], labels[e.as_str()]);
                    return self.finish(term);
                }
                IROp::Ret(_) => return self.finish(Term::Halt),
            };
            if let (Some(d), Some(k)) = (dst, k) {
                self.known.insert(d, k);
            }
        }
        self.finish(Term::Halt)
    }
}

fn dce(f: &mut LFunc) -> bool {
    let mut changed = false;
    loop {
        let mut used = HashSet::new();
        let mut observed = HashSet::new();
        for b in &f.blocks {
            if let Term::Br(c, ..) = b.term {
                used.insert(c);
            }
            for i in &b.insts {
                used.extend(i.op.uses());
                if let LOp::Load(a) | LOp::AddrOf(a) | LOp::LoadIdx(a, _) = &i.op
                    && let Base::Stack(s) = a.base
                {
                    observed.insert(s);
                }
            }
        }

        let mut removed = false;
        for b in &mut f.blocks {
            b.insts.retain_mut(|i| {
                let keep = match &mut i.op {
                    LOp::DivMod { q, r, .. } => {
                        if q.take_if(|v| !used.contains(&*v)).is_some()
                            | r.take_if(|v| !used.contains(&*v)).is_some()
                        {
                            removed = true;
                        }
                        q.is_some() || r.is_some()
                    }
                    LOp::Store(
                        Addr {
                            base: Base::Stack(s),
                            ..
                        },
                        _,
                    )
                    | LOp::StoreIdx(
                        Addr {
                            base: Base::Stack(s),
                            ..
                        },
                        ..,
                    ) => observed.contains(&*s),
                    op if op.removable() => i.dst.is_some_and(|d| used.contains(&d)),
                    _ => true,
                };
                removed |= !keep;
                keep
            });
        }
        if !removed {
            return changed;
        }
        changed = true;
    }
}

fn simplify_cfg(f: &mut LFunc) -> bool {
    let n = f.blocks.len();
    if n == 0 {
        return false;
    }
    let mut changed = false;

    let fwd: Vec<BlockId> = (0..n)
        .map(|mut b| {
            for _ in 0..n {
                match &f.blocks[b] {
                    LBlock {
                        insts,
                        term: Term::Jmp(t),
                    } if insts.is_empty() => b = *t,
                    _ => break,
                }
            }
            b
        })
        .collect();
    let is_halt = |b: &LBlock| b.insts.is_empty() && b.term == Term::Halt;
    for i in 0..n {
        let new = match f.blocks[i].term {
            Term::Jmp(t) if is_halt(&f.blocks[fwd[t]]) => Term::Halt,
            Term::Jmp(t) => Term::Jmp(fwd[t]),
            Term::Br(_, t, e) if fwd[t] == fwd[e] => Term::Jmp(fwd[t]),
            Term::Br(c, t, e) => Term::Br(c, fwd[t], fwd[e]),
            Term::Halt => Term::Halt,
        };
        changed |= new != f.blocks[i].term;
        f.blocks[i].term = new;
    }

    let succs: Vec<_> = f.blocks.iter().map(|b| b.term.succs()).collect();
    let order = rpo(&succs);
    if !order.iter().copied().eq(0..n) {
        changed = true;
        let mut map = vec![usize::MAX; n];
        for (new, &old) in order.iter().enumerate() {
            map[old] = new;
        }
        let mut old: Vec<Option<LBlock>> = std::mem::take(&mut f.blocks)
            .into_iter()
            .map(Some)
            .collect();
        f.blocks = order
            .iter()
            .map(|&o| {
                let mut b = old[o].take().unwrap();
                b.term = b.term.remap(&map);
                b
            })
            .collect();
    }

    let mut preds = vec![0usize; f.blocks.len()];
    for b in &f.blocks {
        for s in b.term.succs() {
            preds[s] += 1;
        }
    }
    for a in 0..f.blocks.len() {
        while let Term::Jmp(b) = f.blocks[a].term {
            if b == a || b == 0 || preds[b] != 1 {
                break;
            }
            let merged = std::mem::replace(
                &mut f.blocks[b],
                LBlock {
                    insts: Vec::new(),
                    term: Term::Halt,
                },
            );
            f.blocks[a].insts.extend(merged.insts);
            f.blocks[a].term = merged.term;
            preds[b] = 0;
            changed = true;
        }
    }
    changed
}

fn prune_stack(f: &mut LFunc) {
    let used: HashSet<Value> = f
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|i| match &i.op {
            LOp::Load(a)
            | LOp::LoadIdx(a, _)
            | LOp::Store(a, _)
            | LOp::StoreIdx(a, ..)
            | LOp::AddrOf(a) => match a.base {
                Base::Stack(s) => Some(s),
                _ => None,
            },
            _ => None,
        })
        .collect();
    f.stack.retain(|(s, _)| used.contains(s));
}

fn ir_succs(b: &IRBlock, labels: &HashMap<&str, BlockId>) -> Vec<BlockId> {
    b.instructions
        .iter()
        .find_map(|i| match &i.op {
            IROp::Jmp(l) => Some(vec![labels[l.as_str()]]),
            IROp::Br(_, t, e) => Some(vec![labels[t.as_str()], labels[e.as_str()]]),
            IROp::Ret(_) => Some(vec![]),
            _ => None,
        })
        .unwrap_or_default()
}

fn rpo(succs: &[Vec<BlockId>]) -> Vec<BlockId> {
    let mut seen = vec![false; succs.len()];
    let mut post = Vec::new();
    let mut stack = vec![(0usize, 0usize)];
    seen[0] = true;
    while let Some(top) = stack.last_mut() {
        let (b, i) = *top;
        top.1 += 1;
        match succs[b].get(i) {
            Some(&s) => {
                if !seen[s] {
                    seen[s] = true;
                    stack.push((s, 0));
                }
            }
            None => {
                post.push(b);
                stack.pop();
            }
        }
    }
    post.reverse();
    post
}
