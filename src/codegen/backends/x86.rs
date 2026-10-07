use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    fmt::Display,
    io::Write,
    vec,
};

use miette::{IntoDiagnostic, Result, bail};

use crate::{
    codegen::{
        CodegenBackend, TargetAssembly, TargetInfo,
        backends::x86::Reg::{Physical, Virtual},
    },
    ir::{self, GlobalInit, ICmpKind, IRFunction, IRGlobal, IRInst, IRModule, IROp, Param, Value},
};

pub struct X86Assembly {
    pub insts: Vec<X86Instruction>,
    pub exports: Vec<String>,
    pub externs: Vec<String>,
    pub data: Vec<IRGlobal>,
    pub is_64: bool,
}

impl TargetAssembly for X86Assembly {
    fn emit_asm<W: Write>(&self, w: &mut W) -> Result<()> {
        if self.is_64 {
            writeln!(w, "default rel").into_diagnostic()?;
        }
        for e in &self.externs {
            writeln!(w, "extern {e}").into_diagnostic()?;
        }
        for g in &self.exports {
            writeln!(w, "global {g}").into_diagnostic()?;
        }
        writeln!(w, "section .text").into_diagnostic()?;
        for i in self.insts.iter() {
            writeln!(w, "{i}").into_diagnostic()?;
        }

        for (section, mutable, zeroed) in [
            (".rodata", false, None),
            (".data", true, Some(false)),
            (".bss", true, Some(true)),
        ] {
            let items: Vec<&IRGlobal> = self
                .data
                .iter()
                .filter(|g| g.mutable == mutable)
                .filter(|g| zeroed.is_none_or(|z| matches!(g.init, GlobalInit::Zeroed(_)) == z))
                .collect();
            if items.is_empty() {
                continue;
            }
            writeln!(w, "section {section}").into_diagnostic()?;
            for g in items {
                writeln!(w, "align {}", g.align).into_diagnostic()?;
                match &g.init {
                    GlobalInit::Zeroed(n) if section == ".bss" => {
                        writeln!(w, "{}: resb {n}", g.name)
                    }
                    GlobalInit::Zeroed(n) => writeln!(w, "{}: times {n} db 0", g.name),
                    GlobalInit::Bytes(b) if b.is_empty() => writeln!(w, "{}:", g.name),
                    GlobalInit::Bytes(b) => {
                        let list: Vec<String> = b.iter().map(u8::to_string).collect();
                        writeln!(w, "{}: db {}", g.name, list.join(", "))
                    }
                }
                .into_diagnostic()?;
            }
        }
        Ok(())
    }
}

pub struct X86Codegen;

impl CodegenBackend for X86Codegen {
    type Assembly = X86Assembly;

    fn compile_module(&self, module: IRModule, target: TargetInfo) -> Result<Self::Assembly> {
        let features = X86Features::from_target_info(target);
        let mut insts = Vec::new();
        let mut exports = Vec::new();

        let mut callees = HashMap::new();
        for f in &module.functions {
            callees.insert(
                f.name.clone(),
                CalleeInfo {
                    is_extern: false,
                    is_variadic: false,
                },
            );
        }
        for e in &module.externs {
            callees.insert(
                e.name.clone(),
                CalleeInfo {
                    is_extern: true,
                    is_variadic: e.variadic,
                },
            );
        }

        for f in module.functions {
            insts.extend(self.compile_function(&f, &features, callees.clone())?);
            exports.push(f.name);
        }

        Ok(X86Assembly {
            insts,
            exports,
            externs: module.externs.iter().map(|e| e.name.clone()).collect(),
            data: module.globals,
            is_64: features.is_64(),
        })
    }
}

impl X86Codegen {
    pub fn compile_function(
        &self,
        function: &IRFunction,
        feat: &X86Features,
        callees: HashMap<String, CalleeInfo>,
    ) -> Result<Vec<X86Instruction>> {
        let mut value_widths = HashMap::new();
        for param in &function.params {
            value_widths.insert(param.id, Width::from_bit_width(param.width, feat)?);
        }
        let mut consts = HashMap::new();
        for b in function.blocks.iter() {
            for inst in b.instructions.iter() {
                if let (IROp::Const(c), Some(res)) = (&inst.op, &inst.meta.result) {
                    consts.insert(res.id, *c);
                }
                if let Some(ref res) = inst.meta.result {
                    value_widths.insert(res.id, Width::from_bit_width(res.width, feat)?);
                }
            }
        }

        let alloca_layout = self.layout_allocas(function);

        let ctx = FunctionContext {
            func_name: function.name.clone(),
            epilogue_label: format!("{}.epilogue", function.name),
            value_widths: value_widths.clone(),
            alloca_layout,
            consts,
            callees,
            next_temp: Cell::new(value_widths.keys().max().copied().unwrap_or_default() + 1),
            out_bytes: Cell::new(0),
        };
        let uses = self.calc_uses(function);
        let mut asm = Vec::new();
        for b in function.blocks.iter() {
            asm.push(X86Instruction::Label(format!(
                "{}.{}",
                function.name, b.label
            )));
            let mut i = 0;
            while i < b.instructions.len() {
                if self.lower_inst(
                    &b.instructions[i],
                    b.instructions.get(i + 1),
                    &mut asm,
                    &uses,
                    feat,
                    &ctx,
                )? {
                    i += 2;
                } else {
                    i += 1;
                }
            }
        }
        let asm = self.remove_dead_consts(asm);
        let intervals = self.compute_intervals(&asm, &function.params)?;
        let (hints, copy_hints) = self.compute_hints(function, feat);
        let (allocations, spill_slots) =
            self.allocate_registers(intervals, hints, copy_hints, feat)?;

        let body = self.apply_allocations(asm, &allocations, feat);
        let body = self.expand_pseudo_insts(body, feat);
        let body = self.peephole(body);

        let mut body = [
            self.emit_param_moves(function, allocations, value_widths, feat),
            body,
        ]
        .concat();

        let callee_saved = self.callee_saved_used(&body, feat);
        let has_calls = body
            .iter()
            .any(|i| matches!(i, X86Instruction::Call { .. }));
        let frame = self.build_frame_info(
            callee_saved,
            ctx.alloca_layout.total_bytes,
            spill_slots,
            ctx.out_bytes.get(),
            has_calls,
            feat,
        );

        let ret_label = format!("{}.ret", function.name);
        let prologue = self.emit_prologue(&frame, feat);
        let mut insts = vec![X86Instruction::Label(ctx.func_name)];
        match self.shrink_wrap(&body, &frame.callee_saved) {
            Some(sw) => {
                let pw = feat.max_width();
                let w = feat.word_bytes();
                let slot = feat.cc().stack_arg_size() as i32;
                for &idx in &sw.frameless {
                    if let X86Instruction::Jmp(l) | X86Instruction::Jcc(_, l) = &mut body[idx]
                        && *l == ctx.epilogue_label
                    {
                        *l = ret_label.clone();
                    }
                    for op in body[idx].operands_mut() {
                        if let Operand::Frame(FrameRef::InArg(i), width) = *op {
                            *op = Operand::Mem {
                                base: Reg::Physical(PhysReg::Sp, pw),
                                offset: w + i as i32 * slot,
                                width,
                            };
                        }
                    }
                }
                body.splice(sw.insert_at..sw.insert_at, prologue);
            }
            None => insts.extend(prologue),
        }
        insts.append(&mut body);
        insts.push(X86Instruction::Label(ctx.epilogue_label));
        insts.extend(self.emit_epilogue(&frame, feat));
        insts.push(X86Instruction::Label(ret_label));
        insts.push(X86Instruction::Ret);

        let insts = self.resolve_frame(insts, &frame, feat);
        Ok(self.cleanup_control_flow(insts, &function.name))
    }

    pub fn lower_inst(
        &self,
        inst: &IRInst,
        next_inst: Option<&IRInst>,
        asm: &mut Vec<X86Instruction>,
        uses: &HashMap<Value, usize>,
        feat: &X86Features,
        ctx: &FunctionContext,
    ) -> Result<bool> {
        let (dst_value, width) = match &inst.meta.result {
            Some(res) => (res.id, Width::from_bit_width(res.width, feat)?),
            None => (0, Width::W0),
        };
        let dst = Operand::Reg(Reg::Virtual(dst_value, width));

        match inst.op {
            IROp::Const(imm) => asm.push(X86Instruction::Mov(dst, Operand::Imm(imm, width))),
            IROp::Add(src1, src2) => asm.push(X86Instruction::BinOp(
                BinKind::Add,
                dst,
                ctx.operand(src1, width),
                ctx.operand(src2, width),
            )),
            IROp::Sub(src1, src2) => asm.push(X86Instruction::BinOp(
                BinKind::Sub,
                dst,
                ctx.operand(src1, width),
                ctx.operand(src2, width),
            )),
            IROp::Mul(src1, src2) => {
                if width == Width::W8 {
                    let al = Operand::Reg(Physical(PhysReg::A, width));

                    asm.push(X86Instruction::Mov(al, Operand::Reg(Virtual(src1, width))));
                    asm.push(X86Instruction::IMul1(Operand::Reg(Virtual(src2, width))));
                    asm.push(X86Instruction::Mov(dst, al));
                } else {
                    asm.push(X86Instruction::BinOp(
                        BinKind::IMul,
                        dst,
                        ctx.operand(src1, width),
                        ctx.operand(src2, width),
                    ))
                }
            }
            IROp::UDiv(src1, src2) | IROp::URem(src1, src2) => {
                let ax = Operand::Reg(Physical(PhysReg::A, width));

                asm.push(X86Instruction::Mov(ax, Operand::Reg(Virtual(src1, width))));

                let reg = if width == Width::W8 {
                    Operand::Reg(Physical(PhysReg::Ah, Width::W8))
                } else {
                    Operand::Reg(Physical(PhysReg::D, Width::W32))
                };
                asm.push(X86Instruction::Xor(reg, reg));

                asm.push(X86Instruction::Div(Operand::Reg(Virtual(src2, width))));

                if matches!(inst.op, IROp::UDiv(_, _)) {
                    asm.push(X86Instruction::Mov(dst, ax));
                } else {
                    let dx = Operand::Reg(Reg::Physical(PhysReg::D, width));
                    asm.push(X86Instruction::Mov(dst, dx));
                }
            }
            IROp::SDiv(src1, src2) | IROp::SRem(src1, src2) => {
                let ax = Operand::Reg(Physical(PhysReg::A, width));

                asm.push(X86Instruction::Mov(ax, Operand::Reg(Virtual(src1, width))));

                match width {
                    Width::W8 => asm.push(X86Instruction::Cbw),
                    Width::W16 => asm.push(X86Instruction::Cwd),
                    Width::W32 => asm.push(X86Instruction::Cdq),
                    Width::W64 => asm.push(X86Instruction::Cqo),
                    _ => bail!("Unsupported divison width: {width:?}"),
                }

                asm.push(X86Instruction::IDiv(Operand::Reg(Virtual(src2, width))));

                if matches!(inst.op, IROp::SDiv(_, _)) {
                    asm.push(X86Instruction::Mov(dst, ax));
                } else {
                    let reg = if width == Width::W8 {
                        Operand::Reg(Physical(PhysReg::Ah, Width::W8))
                    } else {
                        Operand::Reg(Physical(PhysReg::D, width))
                    };
                    asm.push(X86Instruction::Mov(dst, reg));
                }
            }
            IROp::And(src1, src2) => asm.push(X86Instruction::BinOp(
                BinKind::And,
                dst,
                ctx.operand(src1, width),
                ctx.operand(src2, width),
            )),
            IROp::Or(src1, src2) => asm.push(X86Instruction::BinOp(
                BinKind::Or,
                dst,
                ctx.operand(src1, width),
                ctx.operand(src2, width),
            )),
            IROp::Xor(src1, src2) => asm.push(X86Instruction::BinOp(
                BinKind::Xor,
                dst,
                ctx.operand(src1, width),
                ctx.operand(src2, width),
            )),
            IROp::Shl(src, cnt) | IROp::AShr(src, cnt) | IROp::LShr(src, cnt) => {
                let kind = match inst.op {
                    IROp::Shl(..) => ShiftKind::Shl,
                    IROp::AShr(..) => ShiftKind::Sar,
                    _ => ShiftKind::Shr,
                };

                let cnt = match ctx.consts.get(&cnt) {
                    Some(&n) => Operand::Imm(n & 0x3f, Width::W8),
                    None => {
                        let cw = ctx.value_widths[&cnt];
                        asm.push(X86Instruction::Mov(
                            Operand::Reg(Reg::Physical(PhysReg::C, cw)),
                            Operand::Reg(Reg::Virtual(cnt, cw)),
                        ));
                        Operand::Reg(Reg::Physical(PhysReg::C, Width::W8))
                    }
                };

                asm.push(X86Instruction::Mov(dst, ctx.operand(src, width)));
                asm.push(X86Instruction::Shift(kind, dst, cnt));
            }
            IROp::ZExt(src) => {
                let sw = ctx.value_widths[&src];
                if sw == Width::W32 {
                    asm.push(X86Instruction::Mov(
                        Operand::Reg(Reg::Virtual(dst_value, sw)),
                        Operand::Reg(Reg::Virtual(src, sw)),
                    ));
                } else {
                    asm.push(X86Instruction::Movzx(
                        dst,
                        Operand::Reg(Reg::Virtual(src, sw)),
                    ));
                }
            }
            IROp::SExt(src) => {
                let sw = ctx.value_widths[&src];
                asm.push(X86Instruction::Movsx(
                    dst,
                    Operand::Reg(Reg::Virtual(src, sw)),
                ));
            }
            IROp::Trunc(src) => {
                let sw = ctx.value_widths[&src];

                asm.push(X86Instruction::Mov(
                    dst,
                    Operand::Reg(Reg::Virtual(src, sw)),
                ));
            }
            IROp::Jmp(ref label) => asm.push(X86Instruction::Jmp(format!(
                "{}.{}",
                ctx.func_name,
                label.clone()
            ))),
            IROp::Ret(src) => {
                if let Some(src) = src {
                    let w = ctx.value_widths[&src];
                    asm.push(X86Instruction::Mov(
                        Operand::Reg(Reg::Physical(PhysReg::A, w)),
                        Operand::Reg(Reg::Virtual(src, w)),
                    ));
                }
                asm.push(X86Instruction::Jmp(ctx.epilogue_label.clone()));
            }
            IROp::ICmp(ref kind, a, b) => {
                let ow = ctx.value_widths[&a];
                let (mut lhs, mut rhs, mut cc) =
                    (ctx.operand(a, ow), ctx.operand(b, ow), kind.to_cc());
                if matches!(lhs, Operand::Imm(..)) {
                    std::mem::swap(&mut lhs, &mut rhs);
                    cc = cc.swapped();
                }
                if let Operand::Imm(..) = lhs {
                    lhs = Operand::Reg(Reg::Virtual(a, ow)); // imm aren't supported on left
                }

                asm.push(X86Instruction::Cmp(lhs, rhs));

                if uses.get(&dst_value).is_some_and(|u| *u == 1)
                    && let Some(IRInst {
                        op: IROp::Br(_, t, f),
                        ..
                    }) = next_inst
                {
                    asm.push(X86Instruction::Jcc(cc, format!("{}.{t}", ctx.func_name)));
                    asm.push(X86Instruction::Jmp(format!("{}.{f}", ctx.func_name)));
                    return Ok(true);
                } else {
                    asm.push(X86Instruction::SetCC(
                        cc,
                        Operand::Reg(Reg::Virtual(dst_value, Width::W8)),
                    ));
                    if width != Width::W8 {
                        asm.push(X86Instruction::Movzx(
                            dst,
                            Operand::Reg(Reg::Virtual(dst_value, Width::W8)),
                        ));
                    }
                }
            }
            IROp::Br(val, ref t, ref f) => {
                let cw = ctx.value_widths[&val];
                asm.push(X86Instruction::Test(
                    Operand::Reg(Reg::Virtual(val, cw)),
                    Operand::Reg(Reg::Virtual(val, cw)),
                ));
                asm.push(X86Instruction::Jcc(
                    CondCode::Ne,
                    format!("{}.{t}", ctx.func_name),
                ));
                asm.push(X86Instruction::Jmp(format!("{}.{f}", ctx.func_name)));
            }
            IROp::Call(ref target, ref args) => {
                let cc = feat.cc();
                let int_regs = cc.int_arg_registers();
                let (reg_args, stack_args) = args.split_at(args.len().min(int_regs.len()));
                let pw = feat.max_width();
                let slot = cc.stack_arg_size();

                let reg_movs: Vec<(Operand, Operand)> = reg_args
                    .iter()
                    .zip(int_regs)
                    .map(|(&arg, &target_reg)| {
                        let w = ctx.value_widths[&arg];
                        (
                            Operand::Reg(Reg::Physical(target_reg, w)),
                            Operand::Reg(Reg::Virtual(arg, w)),
                        )
                    })
                    .collect();
                if !reg_movs.is_empty() {
                    asm.push(X86Instruction::ParallelMovs(reg_movs));
                }

                let raw_bytes = stack_args.len() as u32 * slot;
                ctx.out_bytes.set(ctx.out_bytes.get().max(raw_bytes));

                for (i, arg) in stack_args.iter().enumerate() {
                    let w = ctx.value_widths[arg];
                    asm.push(X86Instruction::Mov(
                        Operand::Mem {
                            base: Reg::Physical(PhysReg::Sp, pw),
                            offset: i as i32 * slot as i32,
                            width: w,
                        },
                        Operand::Reg(Reg::Virtual(*arg, w)),
                    ));
                }

                let info = &ctx.callees[target];
                let mut uses = int_regs[..reg_args.len()].to_vec();
                if info.is_variadic && feat.is_64() {
                    asm.push(X86Instruction::Mov(
                        Operand::Reg(Reg::Physical(PhysReg::A, Width::W32)),
                        Operand::Imm(0, Width::W32),
                    ));
                    uses.push(PhysReg::A);
                }

                asm.push(X86Instruction::Call {
                    target: target.clone(),
                    uses,
                    clobbers: cc.caller_saved().to_vec(),
                    plt: info.is_extern && feat.is_64(),
                });

                if width != Width::W0 {
                    asm.push(X86Instruction::Mov(
                        dst,
                        Operand::Reg(Reg::Physical(cc.return_register(), width)),
                    ));
                }
            }
            IROp::PtrAdd(p, off) => {
                let pw = feat.max_width();
                let ow = ctx.value_widths[&off];
                let off_op = match ctx.operand(off, pw) {
                    imm @ Operand::Imm(..) => imm,
                    _ if ow == pw => Operand::Reg(Reg::Virtual(off, pw)),
                    _ => {
                        let t = ctx.fresh();
                        asm.push(X86Instruction::Movsx(
                            Operand::Reg(Reg::Virtual(t, pw)),
                            Operand::Reg(Reg::Virtual(off, ow)),
                        ));
                        Operand::Reg(Reg::Virtual(t, pw))
                    }
                };
                asm.push(X86Instruction::BinOp(
                    BinKind::Add,
                    dst,
                    Operand::Reg(Reg::Virtual(p, pw)),
                    off_op,
                ));
            }
            IROp::GlobalAddr(ref name) => asm.push(X86Instruction::LeaGlobal {
                dst,
                name: name.clone(),
                rip_relative: feat.is_64(),
            }),
            IROp::Alloca(_) => {
                let local_offset = ctx.alloca_layout.local_offsets[&dst_value];
                asm.push(X86Instruction::Lea(
                    dst,
                    Operand::Frame(FrameRef::Alloca(local_offset), width),
                ));
            }
            IROp::Load(addr, offset) => {
                asm.push(X86Instruction::Mov(
                    dst,
                    Operand::Mem {
                        base: Reg::Virtual(addr, feat.max_width()),
                        offset: offset as i32,
                        width,
                    },
                ));
            }
            IROp::Store(addr, offset, val) => {
                asm.push(X86Instruction::Mov(
                    Operand::Mem {
                        base: Reg::Virtual(addr, feat.max_width()),
                        offset: offset as i32,
                        width: ctx.value_widths[&val],
                    },
                    Operand::Reg(Reg::Virtual(val, ctx.value_widths[&val])),
                ));
            }
        }

        Ok(false)
    }
}

pub trait CallingConvention {
    fn int_arg_registers(&self) -> &'static [PhysReg];
    fn return_register(&self) -> PhysReg;
    fn caller_saved(&self) -> &'static [PhysReg];
    fn callee_saved(&self) -> &'static [PhysReg];
    fn caller_cleans_stack(&self) -> bool;
    fn stack_arg_size(&self) -> u32;
}

pub struct SysV64;
impl CallingConvention for SysV64 {
    fn int_arg_registers(&self) -> &'static [PhysReg] {
        &[
            PhysReg::Di,
            PhysReg::Si,
            PhysReg::D,
            PhysReg::C,
            PhysReg::R8,
            PhysReg::R9,
        ]
    }
    fn return_register(&self) -> PhysReg {
        PhysReg::A
    }
    fn caller_saved(&self) -> &'static [PhysReg] {
        &[
            PhysReg::A,
            PhysReg::C,
            PhysReg::D,
            PhysReg::Si,
            PhysReg::Di,
            PhysReg::R8,
            PhysReg::R9,
            PhysReg::R10,
            PhysReg::R11,
        ]
    }
    fn callee_saved(&self) -> &'static [PhysReg] {
        &[
            PhysReg::B,
            PhysReg::R12,
            PhysReg::R13,
            PhysReg::R14,
            PhysReg::R15,
            PhysReg::Bp,
        ]
    }
    fn caller_cleans_stack(&self) -> bool {
        true
    }
    fn stack_arg_size(&self) -> u32 {
        8
    }
}

pub struct SysV32;
impl CallingConvention for SysV32 {
    fn int_arg_registers(&self) -> &'static [PhysReg] {
        &[]
    }
    fn return_register(&self) -> PhysReg {
        PhysReg::A
    }
    fn caller_saved(&self) -> &'static [PhysReg] {
        &[PhysReg::A, PhysReg::C, PhysReg::D]
    }
    fn callee_saved(&self) -> &'static [PhysReg] {
        &[PhysReg::B, PhysReg::Si, PhysReg::Di, PhysReg::Bp]
    }
    fn caller_cleans_stack(&self) -> bool {
        true
    }
    fn stack_arg_size(&self) -> u32 {
        4
    }
}

#[derive(Debug, Clone)]
pub struct CalleeInfo {
    pub is_extern: bool,
    pub is_variadic: bool,
}

pub struct FunctionContext {
    pub func_name: String,
    pub epilogue_label: String,
    pub value_widths: HashMap<Value, Width>,
    pub alloca_layout: AllocaLayout,
    pub consts: HashMap<Value, i64>,
    pub callees: HashMap<String, CalleeInfo>,
    pub next_temp: Cell<Value>,
    pub out_bytes: Cell<u32>,
}

impl FunctionContext {
    fn operand(&self, v: Value, w: Width) -> Operand {
        match self.consts.get(&v) {
            Some(c) => Operand::Imm(*c, w),
            None => Operand::Reg(Reg::Virtual(v, w)),
        }
    }

    fn fresh(&self) -> Value {
        let v = self.next_temp.get();
        self.next_temp.set(v + 1);
        v
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegId {
    Virt(Value),
    Phys(PhysReg),
}

impl Reg {
    pub fn id(&self) -> RegId {
        match self {
            Reg::Virtual(v, _) => RegId::Virt(*v),

            Reg::Physical(r, _) => RegId::Phys(match r {
                PhysReg::Ah => PhysReg::A,
                PhysReg::Bh => PhysReg::B,
                PhysReg::Ch => PhysReg::C,
                PhysReg::Dh => PhysReg::D,
                _ => *r,
            }),
        }
    }
}

struct LiveInterval {
    pub reg: RegId,
    pub start: usize,
    pub end: usize,
    pub needs_byte: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegAllocation {
    Reg(PhysReg),
    Spill(u32), // slot index
}

pub struct ActiveEntry {
    pub end: usize,
    pub reg: PhysReg,
    pub vreg: Value,
}

pub struct FrameInfo {
    pub callee_saved: Vec<PhysReg>,
    pub out_bytes: u32,
    pub alloca_base: u32,
    pub local_bytes: u32,
    pub frame_pointer: bool,
}

pub struct ShrinkWrap {
    pub insert_at: usize,
    pub frameless: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameRef {
    Alloca(u32),
    Spill(u32),
    InArg(u32),
}

pub struct AllocaLayout {
    pub local_offsets: HashMap<Value, u32>,
    pub total_bytes: u32,
}

impl X86Codegen {
    fn cleanup_control_flow(
        &self,
        mut insts: Vec<X86Instruction>,
        func: &str,
    ) -> Vec<X86Instruction> {
        use X86Instruction::*;
        let local = format!("{func}.");

        let forward: HashMap<String, String> = insts
            .windows(2)
            .filter_map(|w| match w {
                [Label(l), Jmp(t)] if l != t => Some((l.clone(), t.clone())),
                _ => None,
            })
            .collect();
        let resolve = |l: &str| {
            let mut cur = l;
            for _ in 0..=forward.len() {
                match forward.get(cur) {
                    Some(n) => cur = n,
                    None => break,
                }
            }
            cur.to_string()
        };
        for i in insts.iter_mut() {
            if let Jmp(l) | Jcc(_, l) = i {
                let t = resolve(l);
                *l = t;
            }
        }

        loop {
            let before = insts.len();

            let ret_labels: HashSet<String> = (0..insts.len())
                .filter_map(|i| match &insts[i] {
                    Label(l) => insts[i..]
                        .iter()
                        .find(|x| !matches!(x, Label(_)))
                        .filter(|x| matches!(x, Ret))
                        .map(|_| l.clone()),
                    _ => None,
                })
                .collect();
            for i in insts.iter_mut() {
                if matches!(i, Jmp(l) if ret_labels.contains(l)) {
                    *i = Ret;
                }
            }

            let referenced: HashSet<String> = insts
                .iter()
                .filter_map(|i| match i {
                    Jmp(l) | Jcc(_, l) => Some(l.clone()),
                    _ => None,
                })
                .collect();
            insts.retain(
                |i| !matches!(i, Label(l) if l.starts_with(&local) && !referenced.contains(l)),
            );

            let mut reachable = true;
            insts.retain(|i| {
                if matches!(i, Label(_)) {
                    reachable = true;
                }
                let keep = reachable;
                if matches!(i, Jmp(_) | Ret) {
                    reachable = false;
                }
                keep
            });

            let mut i = 0;
            while i + 2 < insts.len() {
                let inv = match (&insts[i], &insts[i + 1], &insts[i + 2]) {
                    (Jcc(cc, t), Jmp(f), Label(n)) if t == n => Some((cc.inverted(), f.clone())),
                    _ => None,
                };
                if let Some((cc, f)) = inv {
                    insts[i] = Jcc(cc, f);
                    insts.remove(i + 1);
                }
                i += 1;
            }

            let mut i = 0;
            while i + 1 < insts.len() {
                if matches!((&insts[i], &insts[i+1]), (Jmp(t), Label(n)) if t == n) {
                    insts.remove(i);
                } else {
                    i += 1;
                }
            }

            if insts.len() == before {
                break;
            }
        }
        insts
    }

    fn remove_dead_consts(&self, asm: Vec<X86Instruction>) -> Vec<X86Instruction> {
        let read: HashSet<Value> = asm
            .iter()
            .flat_map(|i| i.get_register_rw().0)
            .filter_map(|r| match r {
                Reg::Virtual(v, _) => Some(v),
                _ => None,
            })
            .collect();
        asm.into_iter().filter(|i| !matches!(i, X86Instruction::Mov(Operand::Reg(Reg::Virtual(v, _)), Operand::Imm(..)) if ! read.contains(v))).collect()
    }

    fn calc_uses(&self, func: &IRFunction) -> HashMap<Value, usize> {
        let mut uses = HashMap::new();
        for b in &func.blocks {
            for inst in &b.instructions {
                match &inst.op {
                    IROp::Const(_) | IROp::Jmp(_) | IROp::Alloca(_) | IROp::GlobalAddr(_) => {}
                    IROp::Br(a, _, _)
                    | IROp::Load(a, _)
                    | IROp::ZExt(a)
                    | IROp::SExt(a)
                    | IROp::Trunc(a) => *uses.entry(*a).or_default() += 1,
                    IROp::Ret(a) => {
                        if let Some(a) = a {
                            *uses.entry(*a).or_default() += 1;
                        }
                    }
                    IROp::Add(a, b)
                    | IROp::Sub(a, b)
                    | IROp::Mul(a, b)
                    | IROp::UDiv(a, b)
                    | IROp::SDiv(a, b)
                    | IROp::URem(a, b)
                    | IROp::SRem(a, b)
                    | IROp::ICmp(_, a, b)
                    | IROp::Store(a, _, b)
                    | IROp::And(a, b)
                    | IROp::Or(a, b)
                    | IROp::Xor(a, b)
                    | IROp::Shl(a, b)
                    | IROp::AShr(a, b)
                    | IROp::LShr(a, b)
                    | IROp::PtrAdd(a, b) => {
                        *uses.entry(*a).or_default() += 1;
                        *uses.entry(*b).or_default() += 1;
                    }
                    IROp::Call(_, args) => {
                        for a in args {
                            *uses.entry(*a).or_default() += 1;
                        }
                    }
                }
            }
        }
        uses
    }

    fn compute_hints(
        &self,
        func: &IRFunction,
        feat: &X86Features,
    ) -> (HashMap<Value, PhysReg>, HashMap<Value, Vec<Value>>) {
        let cc = feat.cc();
        let int_regs = cc.int_arg_registers();
        let mut hints = HashMap::new();
        let mut copy_hints = HashMap::new();

        for (p, &r) in func.params.iter().zip(int_regs) {
            hints.insert(p.id, r);
        }

        for b in &func.blocks {
            for inst in &b.instructions {
                match &inst.op {
                    IROp::Add(a, b) | IROp::Sub(a, b) | IROp::Mul(a, b) => {
                        if let Some(ref res) = inst.meta.result {
                            copy_hints.insert(res.id, vec![*a, *b]);
                        }
                    }
                    IROp::Shl(a, _) | IROp::AShr(a, _) | IROp::LShr(a, _) => {
                        if let Some(ref res) = inst.meta.result {
                            copy_hints.insert(res.id, vec![*a]);
                        }
                    }
                    IROp::Call(_, args) => {
                        for (a, &r) in args.iter().zip(int_regs) {
                            hints.entry(*a).or_insert(r);
                        }
                        if let Some(res) = &inst.meta.result {
                            hints.entry(res.id).or_insert(cc.return_register());
                        }
                    }
                    IROp::Ret(Some(v)) => {
                        hints.entry(*v).or_insert(cc.return_register());
                    }
                    _ => {}
                }
            }
        }

        (hints, copy_hints)
    }

    fn peephole(&self, insts: Vec<X86Instruction>) -> Vec<X86Instruction> {
        let mut out: Vec<X86Instruction> = Vec::new();
        for inst in insts.iter() {
            match inst {
                X86Instruction::Mov(Operand::Reg(a), Operand::Reg(b)) if a == b => continue,
                _ => out.push(inst.clone()),
            }
        }
        out
    }

    fn expand_pseudo_insts(
        &self,
        insts: Vec<X86Instruction>,
        feat: &X86Features,
    ) -> Vec<X86Instruction> {
        let pw = feat.max_width();
        let mut out = Vec::new();

        for inst in insts {
            match inst {
                X86Instruction::ParallelMovs(movs) => {
                    let (mem_src, reg_pairs): (Vec<_>, Vec<_>) = movs
                        .into_iter()
                        .partition(|(_, src)| !matches!(src, Operand::Reg(_)));

                    let phys_pairs: Vec<(PhysReg, PhysReg)> = reg_pairs
                        .iter()
                        .map(|(dst, src)| {
                            let d = match dst {
                                Operand::Reg(Reg::Physical(r, _)) => *r,
                                _ => unreachable!(),
                            };
                            let s = match src {
                                Operand::Reg(Reg::Physical(r, _)) => *r,
                                _ => unreachable!(),
                            };
                            (s, d)
                        })
                        .filter(|(s, d)| s != d)
                        .collect();

                    for (src, dst) in
                        self.sequentialize_moves(phys_pairs, feat.scratch_registers()[0])
                    {
                        out.push(X86Instruction::Mov(
                            Operand::Reg(Reg::Physical(dst, pw)),
                            Operand::Reg(Reg::Physical(src, pw)),
                        ));
                    }

                    for (dst, src) in mem_src {
                        out.push(X86Instruction::Mov(dst, src));
                    }
                }
                X86Instruction::BinOp(k, dst, a, b) => {
                    let same = |x: &Operand, y: &Operand| matches!((x, y), (Operand::Reg(p), Operand::Reg(q)) if p.id() == q.id());
                    let op = |d, s| match k {
                        BinKind::Add => X86Instruction::Add(d, s),
                        BinKind::Sub => X86Instruction::Sub(d, s),
                        BinKind::IMul => X86Instruction::IMul2(d, s),
                        BinKind::And => X86Instruction::And(d, s),
                        BinKind::Or => X86Instruction::Or(d, s),
                        BinKind::Xor => X86Instruction::Xor(d, s),
                    };

                    if same(&dst, &a) {
                        out.push(op(dst, b));
                    } else if same(&dst, &b) {
                        if k.commutative() {
                            out.push(op(dst, a));
                        } else {
                            out.push(X86Instruction::Neg(dst));
                            out.push(X86Instruction::Add(dst, a));
                        }
                    } else if let Some(lea) = self.try_lea(k, dst, a, b, feat) {
                        out.push(lea);
                    } else {
                        out.push(X86Instruction::Mov(dst, a));
                        out.push(op(dst, b));
                    }
                }
                _ => out.push(inst),
            }
        }

        out
    }

    fn try_lea(
        &self,
        k: BinKind,
        dst: Operand,
        a: Operand,
        b: Operand,
        feat: &X86Features,
    ) -> Option<X86Instruction> {
        let (a, b) = match (k, a, b) {
            (BinKind::Add, a @ Operand::Imm(..), b @ Operand::Reg(_)) => (b, a),
            (BinKind::Add | BinKind::Sub, a, b) => (a, b),
            _ => return None,
        };
        let (Operand::Reg(dreg), Operand::Reg(Reg::Physical(base, _)), Operand::Imm(imm, _)) =
            (dst, a, b)
        else {
            return None;
        };
        if !matches!(dreg.width(), Width::W32 | Width::W64) {
            return None;
        }
        let offset = i32::try_from(if k == BinKind::Sub {
            imm.checked_neg()?
        } else {
            imm
        })
        .ok()?;
        Some(X86Instruction::Lea(
            dst,
            Operand::Mem {
                base: Reg::Physical(base, feat.max_width()),
                offset,
                width: dreg.width(),
            },
        ))
    }

    fn spill_operand(&self, slot: u32, width: Width) -> Operand {
        Operand::Frame(FrameRef::Spill(slot), width)
    }

    fn frame_offset(&self, r: FrameRef, frame: &FrameInfo, feat: &X86Features) -> (PhysReg, i32) {
        let w = feat.word_bytes();
        let top = frame.local_bytes as i32 + frame.callee_saved.len() as i32 * w;
        let rsp_off = match r {
            FrameRef::Spill(s) => frame.out_bytes as i32 + s as i32 * w,
            FrameRef::Alloca(o) => (frame.alloca_base + o) as i32,
            FrameRef::InArg(i) => {
                let saved_bp = if frame.frame_pointer { w } else { 0 };
                top + saved_bp + w + i as i32 * feat.cc().stack_arg_size() as i32
            }
        };
        if frame.frame_pointer {
            (PhysReg::Bp, rsp_off - top)
        } else {
            (PhysReg::Sp, rsp_off)
        }
    }

    fn resolve_frame(
        &self,
        mut insts: Vec<X86Instruction>,
        frame: &FrameInfo,
        feat: &X86Features,
    ) -> Vec<X86Instruction> {
        for inst in insts.iter_mut() {
            for op in inst.operands_mut() {
                if let Operand::Frame(r, width) = *op {
                    let (base, offset) = self.frame_offset(r, frame, feat);
                    *op = Operand::Mem {
                        base: Reg::Physical(base, feat.max_width()),
                        offset,
                        width,
                    };
                }
            }
        }
        insts
    }

    fn shrink_wrap(&self, body: &[X86Instruction], callee_saved: &[PhysReg]) -> Option<ShrinkWrap> {
        use X86Instruction::*;

        let mut starts = vec![0];
        starts.extend(
            body.iter()
                .enumerate()
                .skip(1)
                .filter(|(_, i)| matches!(i, Label(_)))
                .map(|(i, _)| i),
        );
        let n = starts.len();
        let end = |b: usize| starts.get(b + 1).copied().unwrap_or(body.len());
        let label_block: HashMap<&str, usize> = starts
            .iter()
            .enumerate()
            .filter_map(|(b, &i)| match body.get(i) {
                Some(Label(l)) => Some((l.as_str(), b)),
                _ => None,
            })
            .collect();

        let mut succs = vec![Vec::new(); n];
        for (b, s) in succs.iter_mut().enumerate() {
            let insts = &body[starts[b]..end(b)];
            for i in insts {
                if let Jmp(l) | Jcc(_, l) = i
                    && let Some(&t) = label_block.get(l.as_str())
                {
                    s.push(t);
                }
            }
            if !matches!(insts.last(), Some(Jmp(_) | Ret)) && b + 1 < n {
                s.push(b + 1);
            }
        }

        let reach_from = |s: usize| {
            let mut seen = vec![false; n];
            let mut stack = vec![s];
            while let Some(b) = stack.pop() {
                if !seen[b] {
                    seen[b] = true;
                    stack.extend(&succs[b]);
                }
            }
            seen
        };
        let reachable = reach_from(0);

        let mut preds = vec![Vec::new(); n];
        for b in (0..n).filter(|&b| reachable[b]) {
            for &s in &succs[b] {
                preds[s].push(b);
            }
        }

        let mut dom = vec![vec![true; n]; n];
        dom[0] = (0..n).map(|i| i == 0).collect();
        let mut changed = true;
        while changed {
            changed = false;
            for b in (1..n).filter(|&b| reachable[b]) {
                let mut new = vec![true; n];
                for &p in &preds[b] {
                    for (x, d) in new.iter_mut().zip(&dom[p]) {
                        *x &= *d;
                    }
                }
                new[b] = true;
                if new != dom[b] {
                    dom[b] = new;
                    changed = true;
                }
            }
        }

        let needs_frame = |i: &X86Instruction| {
            if matches!(i, Call { .. }) {
                return true;
            }
            if i.operands().iter().any(|o| {
                matches!(
                    o,
                    Operand::Frame(FrameRef::Spill(_) | FrameRef::Alloca(_), _)
                )
            }) {
                return true;
            }
            let (r, w) = i.get_register_rw();
            r.iter().chain(&w).any(|r| {matches!(r.id(), RegId::Phys(p) if p ==PhysReg::Sp || callee_saved.contains(&p))})
        };
        let needy: Vec<usize> = (0..n)
            .filter(|&b| reachable[b] && body[starts[b]..end(b)].iter().any(needs_frame))
            .collect();
        if needy.is_empty() {
            return None;
        }

        let s = (0..n)
            .filter(|&c| needy.iter().all(|&b| dom[b][c]))
            .max_by_key(|&c| dom[c].iter().filter(|&&x| x).count())?;
        if s == 0 {
            return None;
        }

        let region = reach_from(s);
        let closed = (0..n).all(|b| !region[b] || dom[b][s]);
        let in_loop = (0..n).any(|b| region[b] && succs[b].contains(&s));
        if !closed || in_loop {
            return None;
        }

        Some(ShrinkWrap {
            insert_at: starts[s] + 1,
            frameless: (0..n)
                .filter(|&b| reachable[b] && !region[b])
                .flat_map(|b| starts[b]..end(b))
                .collect(),
        })
    }

    fn layout_allocas(&self, func: &IRFunction) -> AllocaLayout {
        let mut local_offsets = HashMap::new();
        let mut cursor = 0u32;

        for b in &func.blocks {
            for inst in &b.instructions {
                if let (IROp::Alloca(size), Some(res)) = (&inst.op, &inst.meta.result) {
                    let align = (*size).clamp(1, 16).next_power_of_two();
                    cursor = cursor.div_ceil(align) * align;
                    local_offsets.insert(res.id, cursor);
                    cursor += size;
                }
            }
        }

        AllocaLayout {
            local_offsets,
            total_bytes: cursor,
        }
    }

    fn compute_intervals(
        &self,
        asm: &[X86Instruction],
        params: &[Param],
    ) -> Result<Vec<LiveInterval>> {
        let mut virt: HashMap<Value, LiveInterval> = HashMap::new();
        let mut phys_open: HashMap<PhysReg, LiveInterval> = HashMap::new();
        let mut finished: Vec<LiveInterval> = Vec::new();

        for param in params {
            virt.insert(
                param.id,
                LiveInterval {
                    reg: RegId::Virt(param.id),
                    start: 0,
                    end: 0,
                    needs_byte: matches!(param.width, ir::Width::Width(8)),
                },
            );
        }

        for (idx, inst) in asm.iter().enumerate() {
            let (reads, writes) = inst.get_register_rw();
            let read_ids: Vec<RegId> = reads.iter().map(|r| r.id()).collect();

            for r in reads.iter().chain(writes.iter()) {
                if let Reg::Virtual(v, w) = *r {
                    let iv = virt.entry(v).or_insert(LiveInterval {
                        reg: RegId::Virt(v),
                        start: idx,
                        end: idx,
                        needs_byte: false,
                    });
                    iv.start = iv.start.min(idx);
                    iv.end = iv.end.max(idx);
                    iv.needs_byte |= w == Width::W8;
                }
            }

            let fresh = |p| LiveInterval {
                reg: RegId::Phys(p),
                start: idx,
                end: idx,
                needs_byte: false,
            };
            for r in &reads {
                if let RegId::Phys(p) = r.id() {
                    phys_open.entry(p).or_insert_with(|| fresh(p)).end = idx;
                }
            }

            for w in &writes {
                if let RegId::Phys(p) = w.id() {
                    if read_ids.contains(&w.id()) {
                        phys_open.entry(p).or_insert_with(|| fresh(p)).end = idx;
                    } else {
                        if let Some(old) = phys_open.remove(&p) {
                            finished.push(old);
                        }
                        phys_open.insert(p, fresh(p));
                    }
                }
            }
        }
        finished.extend(phys_open.into_values());

        self.extend_for_liveness(asm, &mut virt);

        finished.extend(virt.into_values());
        finished.sort_by_key(|i| i.start);
        Ok(finished)
    }

    fn extend_for_liveness(&self, asm: &[X86Instruction], virt: &mut HashMap<Value, LiveInterval>) {
        let starts: Vec<usize> = asm
            .iter()
            .enumerate()
            .filter(|(_, i)| matches!(i, X86Instruction::Label(_)))
            .map(|(i, _)| i)
            .collect();
        let n = starts.len();
        let end_of = |b: usize| starts.get(b + 1).copied().unwrap_or(asm.len()) - 1;
        let label_block: HashMap<&str, usize> = starts
            .iter()
            .enumerate()
            .map(|(b, &i)| match &asm[i] {
                X86Instruction::Label(l) => (l.as_str(), b),
                _ => unreachable!(),
            })
            .collect();

        let mut succs = vec![Vec::new(); n];
        let mut uses = vec![HashSet::new(); n];
        let mut defs = vec![HashSet::new(); n];

        for b in 0..n {
            for inst in asm.iter().take(end_of(b) + 1).skip(starts[b]) {
                if let X86Instruction::Jmp(l) | X86Instruction::Jcc(_, l) = inst
                    && let Some(&t) = label_block.get(l.as_str())
                {
                    succs[b].push(t);
                }
                let (reads, writes) = inst.get_register_rw();
                for r in reads {
                    if let Reg::Virtual(v, _) = r
                        && !defs[b].contains(&v)
                    {
                        uses[b].insert(v);
                    }
                }
                for w in writes {
                    if let Reg::Virtual(v, _) = w {
                        defs[b].insert(v);
                    }
                }
            }
            let falls_through =
                !matches!(asm[end_of(b)], X86Instruction::Jmp(_) | X86Instruction::Ret);
            if falls_through && b + 1 < n {
                succs[b].push(b + 1);
            }
        }

        let mut live_in: Vec<HashSet<Value>> = vec![HashSet::new(); n];
        let mut live_out: Vec<HashSet<Value>> = vec![HashSet::new(); n];
        loop {
            let mut changed = false;
            for b in (0..n).rev() {
                let out: HashSet<Value> = succs[b]
                    .iter()
                    .flat_map(|&s| live_in[s].iter().copied())
                    .collect();
                let inn: HashSet<Value> = uses[b]
                    .iter()
                    .copied()
                    .chain(out.difference(&defs[b]).copied())
                    .collect();

                if out != live_out[b] || inn != live_in[b] {
                    live_out[b] = out;
                    live_in[b] = inn;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        for b in 0..n {
            for v in &live_in[b] {
                if let Some(iv) = virt.get_mut(v) {
                    iv.start = iv.start.min(starts[b]);
                }
            }
            for v in &live_out[b] {
                if let Some(iv) = virt.get_mut(v) {
                    iv.end = iv.end.max(end_of(b));
                }
            }
        }
    }

    fn allocate_registers(
        &self,
        intervals: Vec<LiveInterval>,
        hints: HashMap<Value, PhysReg>,
        copy_hints: HashMap<Value, Vec<Value>>,
        feat: &X86Features,
    ) -> Result<(HashMap<Value, RegAllocation>, u32)> {
        let allocatable = feat.get_allocatable_registers();
        let (phys, virt): (Vec<_>, Vec<_>) = intervals
            .into_iter()
            .partition(|i| matches!(i.reg, RegId::Phys(_)));

        let overlaps = |a: &LiveInterval, b: &LiveInterval| a.start < b.end && b.start < a.end;
        let usable = |c: PhysReg, iv: &LiveInterval| {
            (!iv.needs_byte || feat.is_byte_capable(c))
                && !phys
                    .iter()
                    .any(|p| p.reg == RegId::Phys(c) && overlaps(p, iv))
        };

        let mut allocations = HashMap::new();
        let mut active: Vec<ActiveEntry> = Vec::new();
        let mut next_slot = 0u32;

        for iv in virt {
            let RegId::Virt(vreg) = iv.reg else {
                unreachable!()
            };
            active.retain(|e| e.end > iv.start);
            let taken: Vec<PhysReg> = active.iter().map(|e| e.reg).collect();

            let mut candidates: Vec<PhysReg> = hints.get(&vreg).copied().into_iter().collect();
            for src in copy_hints.get(&vreg).into_iter().flatten() {
                if let Some(RegAllocation::Reg(r)) = allocations.get(src) {
                    candidates.push(*r);
                }
            }
            let hinted = candidates
                .into_iter()
                .find(|&h| allocatable.contains(&h) && !taken.contains(&h) && usable(h, &iv));

            if let Some(r) = hinted.or_else(|| {
                allocatable
                    .iter()
                    .copied()
                    .find(|&c| !taken.contains(&c) && usable(c, &iv))
            }) {
                allocations.insert(vreg, RegAllocation::Reg(r));
                active.push(ActiveEntry {
                    end: iv.end,
                    reg: r,
                    vreg,
                });
                continue;
            }

            let victim = active
                .iter()
                .enumerate()
                .filter(|(_, e)| e.end >= iv.end && usable(e.reg, &iv))
                .max_by_key(|(_, e)| e.end)
                .map(|(i, _)| i);

            if let Some(i) = victim {
                let e = active.remove(i);
                allocations.insert(e.vreg, RegAllocation::Spill(next_slot));
                next_slot += 1;
                allocations.insert(vreg, RegAllocation::Reg(e.reg));
                active.push(ActiveEntry {
                    end: iv.end,
                    reg: e.reg,
                    vreg,
                });
            } else {
                allocations.insert(vreg, RegAllocation::Spill(next_slot));
                next_slot += 1;
            }
        }

        Ok((allocations, next_slot))
    }

    fn apply_allocations(
        &self,
        insts: Vec<X86Instruction>,
        allocations: &HashMap<Value, RegAllocation>,
        feat: &X86Features,
    ) -> Vec<X86Instruction> {
        let scratch = feat.scratch_registers();
        let mut out = Vec::with_capacity(insts.len());

        for mut inst in insts {
            if let X86Instruction::ParallelMovs(movs) = &mut inst {
                for (_, src) in movs.iter_mut() {
                    if let Operand::Reg(Reg::Virtual(v, w)) = *src {
                        *src = match allocations[&v] {
                            RegAllocation::Reg(r) => Operand::Reg(Reg::Physical(r, w)),
                            RegAllocation::Spill(s) => self.spill_operand(s, w),
                        }
                    }
                }
                out.push(inst);
                continue;
            }

            if let X86Instruction::BinOp(_, _, a, b) = &mut inst {
                for op in [a, b] {
                    if let Operand::Reg(Reg::Virtual(v, w)) = *op
                        && let Some(RegAllocation::Spill(s)) = allocations.get(&v)
                    {
                        *op = self.spill_operand(*s, w);
                    }
                }
            }

            let (reads, writes) = inst.get_register_rw();
            let mut pre = Vec::new();
            let mut post = Vec::new();
            let mut assigned: Vec<(Value, PhysReg)> = Vec::new();

            inst.rewrite_registers_with(|reg| {
                let Reg::Virtual(v, width) = *reg else {
                    return;
                };
                match allocations.get(&v) {
                    Some(RegAllocation::Reg(r)) => *reg = Reg::Physical(*r, width),
                    Some(RegAllocation::Spill(slot)) => {
                        let s = match assigned.iter().find(|(x, _)| *x == v) {
                            Some(&(_, s)) => s,
                            None => {
                                let s = scratch
                                    .iter()
                                    .copied()
                                    .find(|c| {
                                        !assigned.iter().any(|(_, t)| t == c)
                                            && (width != Width::W8 || feat.is_byte_capable(*c))
                                    })
                                    .expect("no usable scratch register for spilled operand");
                                assigned.push((v, s));
                                let mem = self.spill_operand(*slot, width);
                                let sreg = Operand::Reg(Reg::Physical(s, width));
                                if reads.iter().any(|r| r.id() == RegId::Virt(v)) {
                                    pre.push(X86Instruction::Mov(sreg, mem));
                                }
                                if writes.iter().any(|w| w.id() == RegId::Virt(v)) {
                                    post.push(X86Instruction::Mov(mem, sreg));
                                }
                                s
                            }
                        };
                        *reg = Reg::Physical(s, width);
                    }
                    None => {}
                }
            });

            out.extend(pre);
            out.push(inst);
            out.extend(post);
        }

        out
    }

    pub fn callee_saved_used(&self, body: &[X86Instruction], feat: &X86Features) -> Vec<PhysReg> {
        let mut written = HashSet::new();
        for inst in body {
            for r in inst.get_register_rw().1 {
                if let RegId::Phys(p) = r.id() {
                    written.insert(p);
                }
            }
        }
        feat.cc()
            .callee_saved()
            .iter()
            .copied()
            .filter(|r| written.contains(r) && !(feat.frame_pointer && *r == PhysReg::Bp))
            .collect()
    }

    fn build_frame_info(
        &self,
        callee_saved: Vec<PhysReg>,
        alloca_total: u32,
        spill_slots: u32,
        out_bytes: u32,
        has_calls: bool,
        feat: &X86Features,
    ) -> FrameInfo {
        let w = feat.word_bytes() as u32;
        let below_allocas = out_bytes + spill_slots * w;
        let alloca_base = if alloca_total > 0 {
            below_allocas.next_multiple_of(16)
        } else {
            below_allocas
        };
        let mut local_bytes = (alloca_base + alloca_total).next_multiple_of(w);

        let pushed = (1 + feat.frame_pointer as u32 + callee_saved.len() as u32) * w;
        if has_calls || alloca_total > 0 {
            while !(pushed + local_bytes).is_multiple_of(16) {
                local_bytes += w;
            }
        }

        FrameInfo {
            callee_saved,
            out_bytes,
            alloca_base,
            local_bytes,
            frame_pointer: feat.frame_pointer,
        }
    }

    fn emit_prologue(&self, frame: &FrameInfo, feat: &X86Features) -> Vec<X86Instruction> {
        let bp = Operand::Reg(Reg::Physical(PhysReg::Bp, feat.max_width()));
        let sp = Operand::Reg(Reg::Physical(PhysReg::Sp, feat.max_width()));

        let mut out = Vec::new();
        if frame.frame_pointer {
            out.push(X86Instruction::Push(bp));
            out.push(X86Instruction::Mov(bp, sp));
        }
        for r in frame.callee_saved.iter().rev() {
            out.push(X86Instruction::Push(Operand::Reg(Reg::Physical(
                *r,
                feat.max_width(),
            ))));
        }
        if frame.local_bytes > 0 {
            out.push(X86Instruction::Sub(
                sp,
                Operand::Imm(frame.local_bytes as i64, feat.max_width()),
            ));
        }

        out
    }

    fn emit_param_moves(
        &self,
        func: &IRFunction,
        allocations: HashMap<Value, RegAllocation>,
        value_widths: HashMap<Value, Width>,
        feat: &X86Features,
    ) -> Vec<X86Instruction> {
        let int_regs = feat.cc().int_arg_registers();
        let scratch = feat.scratch_registers()[0];
        let mut spills = Vec::new();
        let mut reg_moves = Vec::new();
        let mut stack_lods = Vec::new();

        for (i, param) in func.params.iter().enumerate() {
            let w = value_widths[&param.id];
            let Some(&alloc) = allocations.get(&param.id) else {
                continue;
            };

            if let Some(&arriving) = int_regs.get(i) {
                match alloc {
                    RegAllocation::Spill(s) => spills.push(X86Instruction::Mov(
                        self.spill_operand(s, w),
                        Operand::Reg(Reg::Physical(arriving, w)),
                    )),
                    RegAllocation::Reg(r) if r != arriving => reg_moves.push((arriving, r)),
                    _ => {}
                }
            } else {
                let src = Operand::Frame(FrameRef::InArg((i - int_regs.len()) as u32), w);
                match alloc {
                    RegAllocation::Reg(r) => {
                        stack_lods.push(X86Instruction::Mov(Operand::Reg(Reg::Physical(r, w)), src))
                    }
                    RegAllocation::Spill(s) => {
                        let t = Operand::Reg(Reg::Physical(scratch, w));
                        stack_lods.push(X86Instruction::Mov(t, src));
                        stack_lods.push(X86Instruction::Mov(self.spill_operand(s, w), t));
                    }
                }
            }
        }

        let mut out = spills;
        for (src, dst) in self.sequentialize_moves(reg_moves, scratch) {
            out.push(X86Instruction::Mov(
                Operand::Reg(Reg::Physical(dst, feat.max_width())),
                Operand::Reg(Reg::Physical(src, feat.max_width())),
            ));
        }
        out.extend(stack_lods);

        out
    }

    fn sequentialize_moves(
        &self,
        moves: Vec<(PhysReg, PhysReg)>,
        scratch: PhysReg,
    ) -> Vec<(PhysReg, PhysReg)> {
        let mut pending = moves;
        let mut result = Vec::new();

        while !pending.is_empty() {
            let safe_idx = pending
                .iter()
                .position(|&(_, dst)| !pending.iter().any(|&(src, _)| src == dst));

            if let Some(i) = safe_idx {
                result.push(pending.remove(i));
            } else {
                let (src, _) = pending[0];
                result.push((src, scratch));
                for pair in pending.iter_mut() {
                    if pair.0 == src {
                        pair.0 = scratch;
                    }
                }
            }
        }

        result
    }

    fn emit_epilogue(&self, frame: &FrameInfo, feat: &X86Features) -> Vec<X86Instruction> {
        let bp = Operand::Reg(Reg::Physical(PhysReg::Bp, feat.max_width()));
        let sp = Operand::Reg(Reg::Physical(PhysReg::Sp, feat.max_width()));

        let mut out = Vec::new();
        if frame.local_bytes > 0 {
            out.push(X86Instruction::Add(
                sp,
                Operand::Imm(frame.local_bytes as i64, feat.max_width()),
            ));
        }
        for r in frame.callee_saved.iter() {
            out.push(X86Instruction::Pop(Operand::Reg(Reg::Physical(
                *r,
                feat.max_width(),
            ))));
        }
        out.push(X86Instruction::Ret);
        if frame.frame_pointer {
            out.push(X86Instruction::Pop(bp));
        }
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86Mode {
    Bit32,
    Bit64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct X86Features {
    pub mode: X86Mode,
    pub arch: String,
    pub avx1: bool,
    pub avx2: bool,
    pub avx512: bool,
    pub frame_pointer: bool, // keep rbp as frame pointer (if false it becomes gp)
}

impl X86Features {
    pub fn word_bytes(&self) -> i32 {
        if self.is_64() { 8 } else { 4 }
    }

    pub fn from_target_info(info: TargetInfo) -> Self {
        X86Features {
            mode: if matches!(info.triple.arch.as_str(), "x86_64" | "x86-64" | "amd64") {
                X86Mode::Bit64
            } else {
                X86Mode::Bit32
            },
            arch: info.triple.arch,
            // TODO: Parse
            avx1: false,
            avx2: false,
            avx512: false,
            frame_pointer: false,
        }
    }

    pub fn is_64(&self) -> bool {
        matches!(self.mode, X86Mode::Bit64)
    }

    pub fn cc(&self) -> Box<dyn CallingConvention> {
        // BUG: Add other (non-linux) conventions
        match self.is_64() {
            true => Box::new(SysV64),
            false => Box::new(SysV32),
        }
    }

    pub fn max_width(&self) -> Width {
        if self.is_64() { Width::W64 } else { Width::W32 }
    }

    pub fn scratch_registers(&self) -> [PhysReg; 2] {
        if self.is_64() {
            [PhysReg::R11, PhysReg::R10]
        } else {
            [PhysReg::B, PhysReg::Di]
        }
    }

    pub fn get_allocatable_registers(&self) -> Vec<PhysReg> {
        let mut regs = self.base_allocatable_registers();
        if !self.frame_pointer {
            regs.push(PhysReg::Bp);
        }
        regs
    }
    fn base_allocatable_registers(&self) -> Vec<PhysReg> {
        if self.is_64() {
            vec![
                PhysReg::C,
                PhysReg::Si,
                PhysReg::Di,
                PhysReg::R8,
                PhysReg::R9,
                PhysReg::R12,
                PhysReg::R13,
                PhysReg::R14,
                PhysReg::R15,
                PhysReg::D,
                PhysReg::A,
                PhysReg::B,
            ]
        } else {
            vec![PhysReg::C, PhysReg::Si, PhysReg::D, PhysReg::A]
        }
    }

    pub fn is_byte_capable(&self, r: PhysReg) -> bool {
        self.is_64() || matches!(r, PhysReg::A | PhysReg::B | PhysReg::C | PhysReg::D)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Width {
    W0,
    W8,
    W16,
    W32,
    W64,  // x86_64
    W128, // AVX1
    W256, // AVX2
    W512, // AVX512
}

impl Width {
    pub fn from_bit_width(width: ir::Width, feat: &X86Features) -> Result<Self> {
        Ok(match width {
            ir::Width::Width(w) => match w {
                0 => Self::W0,
                8 => Self::W8,
                16 => Self::W16,
                32 => Self::W32,
                64 if feat.is_64() => Self::W64,
                128 if feat.avx1 => Self::W128,
                256 if feat.avx1 => Self::W256,
                512 if feat.avx512 => Self::W512,
                _ => bail!("{} does not support a bit width of {w}", feat.arch),
            },
            ir::Width::Ptr => feat.max_width(),
        })
    }
}

#[rustfmt::skip]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysReg {
    A, B, C, D, Si, Di, Sp, Bp,
    R8, R9, R10, R11, R12, R13, R14, R15, // x86_64
    Xmm(u8), // AVX
    Ah, Bh, Ch, Dh, // Special cases
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reg {
    Virtual(Value, Width),
    Physical(PhysReg, Width),
}

impl Reg {
    pub fn width(&self) -> Width {
        match self {
            Reg::Physical(_, w) | Reg::Virtual(_, w) => *w,
        }
    }
}

impl Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reg::Virtual(id, w) => write!(f, "%v{id}.{w:?}"),
            Reg::Physical(reg, w) => {
                let prefix = |w: &Width| match w {
                    Width::W32 => "e",
                    Width::W64 => "r",
                    Width::W128 => "x",
                    Width::W256 => "y",
                    Width::W512 => "z",
                    _ => "",
                };
                let suffix = |w: &Width| match w {
                    Width::W8 => "b",
                    Width::W16 => "w",
                    Width::W32 => "d",
                    _ => "",
                };
                let reg_code = |reg: &PhysReg| match reg {
                    PhysReg::A => "a",
                    PhysReg::B => "b",
                    PhysReg::C => "c",
                    PhysReg::D => "d",
                    PhysReg::Si => "si",
                    PhysReg::Di => "di",
                    PhysReg::Sp => "sp",
                    PhysReg::Bp => "bp",
                    PhysReg::R8 => "r8",
                    PhysReg::R9 => "r9",
                    PhysReg::R10 => "r10",
                    PhysReg::R11 => "r11",
                    PhysReg::R12 => "r12",
                    PhysReg::R13 => "r13",
                    PhysReg::R14 => "r14",
                    PhysReg::R15 => "r15",
                    _ => "",
                };
                let name = match reg {
                    PhysReg::Xmm(n) => format!("{}mm{n}", prefix(w)),
                    PhysReg::Ah => "ah".to_string(),
                    PhysReg::Bh => "bh".to_string(),
                    PhysReg::Ch => "ch".to_string(),
                    PhysReg::Dh => "dh".to_string(),
                    PhysReg::A | PhysReg::B | PhysReg::C | PhysReg::D => match w {
                        Width::W8 => format!("{}l", reg_code(reg)),
                        w => format!("{}{}x", prefix(w), reg_code(reg)),
                    },
                    PhysReg::Si | PhysReg::Di | PhysReg::Sp | PhysReg::Bp => match w {
                        Width::W8 => format!("{}l", reg_code(reg)),
                        w => format!("{}{}", prefix(w), reg_code(reg)),
                    },
                    reg => format!("{}{}", reg_code(reg), suffix(w)),
                };
                write!(f, "{name}")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operand {
    Reg(Reg),
    Imm(i64, Width),
    Mem {
        base: Reg,
        offset: i32,
        width: Width,
    },
    Frame(FrameRef, Width),
}

impl Operand {
    pub fn width(&self) -> Width {
        match self {
            Self::Reg(reg) => reg.width(),
            Self::Imm(_, width)
            | Self::Frame(_, width)
            | Self::Mem {
                base: _,
                offset: _,
                width,
            } => *width,
        }
    }

    pub fn as_reg(&self) -> Option<&Reg> {
        match self {
            Self::Reg(reg) => Some(reg),
            _ => None,
        }
    }

    pub fn reg_reads(&self) -> Option<Reg> {
        match self {
            Self::Reg(reg) => Some(*reg),
            Self::Mem { base, .. } => Some(*base),
            _ => None,
        }
    }

    pub fn replace_vregs_with(&mut self, mut f: impl FnMut(&mut Reg)) {
        match self {
            Self::Reg(r) => {
                f(r);
            }
            Self::Mem {
                base,
                offset: _,
                width: _,
            } => {
                f(base);
            }
            _ => {}
        };
    }
}

impl Display for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reg(reg) => write!(f, "{reg}"),
            Self::Imm(val, _) => write!(f, "{val}"),
            Self::Mem {
                base,
                offset,
                width,
            } => {
                let word = match width {
                    Width::W8 => "BYTE ",
                    Width::W16 => "WORD ",
                    Width::W32 => "DWORD ",
                    Width::W64 => "QWORD ",
                    _ => "",
                };
                match offset {
                    0 => write!(f, "{word}[{base}]"),
                    o if *o < 0 => write!(f, "{word}[{base} - {}]", -(*o as i64)),
                    o => write!(f, "{word}[{base} + {o}]"),
                }
            }
            Self::Frame(r, w) => write!(f, "<{r:?}.{w:?}>"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[rustfmt::skip]
pub enum CondCode { E, Ne, L, Le, G, Ge, B, Be, A, Ae }

impl CondCode {
    fn swapped(self) -> Self {
        use CondCode::*;
        match self {
            E => E,
            Ne => Ne,
            L => G,
            G => L,
            Le => Ge,
            Ge => Le,
            B => A,
            A => B,
            Be => Ae,
            Ae => Be,
        }
    }

    fn inverted(self) -> Self {
        use CondCode::*;
        match self {
            E => Ne,
            Ne => E,
            L => Ge,
            Ge => L,
            G => Le,
            Le => G,
            B => Ae,
            Ae => B,
            A => Be,
            Be => A,
        }
    }
}

impl ICmpKind {
    fn to_cc(self) -> CondCode {
        match self {
            Self::Eq => CondCode::E,
            Self::Ne => CondCode::Ne,
            Self::SLt => CondCode::L,
            Self::SLe => CondCode::Le,
            Self::SGt => CondCode::G,
            Self::SGe => CondCode::Ge,
            Self::ULt => CondCode::B,
            Self::ULe => CondCode::Be,
            Self::UGt => CondCode::A,
            Self::UGe => CondCode::Ae,
        }
    }
}

impl Display for CondCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::E => "e",
                Self::Ne => "ne",
                Self::L => "l",
                Self::Le => "le",
                Self::G => "g",
                Self::Ge => "ge",
                Self::B => "b",
                Self::Be => "be",
                Self::A => "a",
                Self::Ae => "ae",
            },
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinKind {
    Add,
    Sub,
    IMul,
    And,
    Or,
    Xor,
}

impl BinKind {
    fn commutative(self) -> bool {
        !matches!(self, BinKind::Sub)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftKind {
    Shl,
    Shr,
    Sar,
}

#[derive(Debug, Clone)]
pub enum X86Instruction {
    Mov(Operand, Operand),
    Movzx(Operand, Operand),
    Movsx(Operand, Operand),
    Add(Operand, Operand),
    Sub(Operand, Operand),
    IMul1(Operand),
    IMul2(Operand, Operand),
    Div(Operand),
    IDiv(Operand),
    Cbw,
    Cwd,
    Cdq,
    Cqo,
    And(Operand, Operand),
    Or(Operand, Operand),
    Xor(Operand, Operand),
    Shift(ShiftKind, Operand, Operand),
    Jmp(String),
    Ret,
    Label(String),
    Push(Operand),
    Pop(Operand),
    Cmp(Operand, Operand),
    SetCC(CondCode, Operand),
    Test(Operand, Operand),
    Jcc(CondCode, String),
    Call {
        target: String,
        uses: Vec<PhysReg>,
        clobbers: Vec<PhysReg>,
        plt: bool,
    },
    Lea(Operand, Operand),
    LeaGlobal {
        dst: Operand,
        name: String,
        rip_relative: bool,
    },

    // Pseudo instructions
    ParallelMovs(Vec<(Operand, Operand)>),
    BinOp(BinKind, Operand, Operand, Operand),
    Neg(Operand),
}

impl Display for X86Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mov(dst, src) => write!(f, "\tmov {dst}, {src}"),
            Self::Movzx(dst, src) => write!(f, "\tmovzx {dst}, {src}"),
            Self::Movsx(dst, src) if src.width() == Width::W32 => {
                write!(f, "\tmovsxd {dst}, {src}")
            }
            Self::Movsx(dst, src) => write!(f, "\tmovsx {dst}, {src}"),
            Self::Add(dst, src) => write!(f, "\tadd {dst}, {src}"),
            Self::Sub(dst, src) => write!(f, "\tsub {dst}, {src}"),
            Self::IMul1(src) => write!(f, "\timul {src}"),
            Self::IMul2(dst, src) => write!(f, "\timul {dst}, {src}"),
            Self::Div(div) => write!(f, "\tdiv {div}"),
            Self::IDiv(div) => write!(f, "\tidiv {div}"),
            Self::Cbw => write!(f, "\tcbw"),
            Self::Cwd => write!(f, "\tcwd"),
            Self::Cdq => write!(f, "\tcdq"),
            Self::Cqo => write!(f, "\tcqo"),
            Self::And(dst, src) => write!(f, "\tand {dst}, {src}"),
            Self::Or(dst, src) => write!(f, "\tor {dst}, {src}"),
            Self::Xor(dst, src) => write!(f, "\txor {dst}, {src}"),
            Self::Shift(kind, dst, src) => write!(
                f,
                "\t{} {dst}, {src}",
                match kind {
                    ShiftKind::Shl => "shl",
                    ShiftKind::Shr => "shr",
                    ShiftKind::Sar => "sar",
                }
            ),
            Self::Jmp(jmp) => write!(f, "\tjmp {jmp}"),
            Self::Ret => write!(f, "\tret"),
            Self::Label(lbl) => write!(f, "{lbl}:"),
            Self::Push(reg) => write!(f, "\tpush {reg}"),
            Self::Pop(reg) => write!(f, "\tpop {reg}"),
            Self::Cmp(val1, val2) => write!(f, "\tcmp {val1}, {val2}"),
            Self::SetCC(cc, dst) => write!(f, "\tset{cc} {dst}"),
            Self::Test(val1, val2) => write!(f, "\ttest {val1}, {val2}"),
            Self::Jcc(cc, jmp) => write!(f, "\tj{cc} {jmp}"),
            Self::Call {
                target, plt: true, ..
            } => write!(f, "\tcall {target} wrt ..plt"),
            Self::Call { target, .. } => write!(f, "\tcall {target}"),
            Self::Lea(dst, src) => write!(f, "\tlea {dst}, {src}"),
            Self::LeaGlobal {
                dst,
                name,
                rip_relative: true,
            } => write!(f, "\tlea {dst}, [rel {name}]"),
            Self::LeaGlobal {
                dst,
                name,
                rip_relative: false,
            } => write!(f, "\tmov {dst}, {name}"),
            Self::ParallelMovs(_) | Self::BinOp(..) | Self::Neg(_) => {
                unreachable!()
            }
        }
    }
}

impl X86Instruction {
    fn get_register_rw(&self) -> (Vec<Reg>, Vec<Reg>) {
        match self {
            // Special case when used for clearing
            Self::Xor(Operand::Reg(a), Operand::Reg(b)) if a.id() == b.id() => (vec![], vec![*a]),
            Self::Mov(dst, src)
            | Self::Movzx(dst, src)
            | Self::Movsx(dst, src)
            | Self::Lea(dst, src) => {
                let mut reads: Vec<Reg> = src.reg_reads().into_iter().collect();
                if let Operand::Mem { base, .. } = dst {
                    reads.push(*base);
                }
                (reads, dst.as_reg().into_iter().copied().collect())
            }
            Self::Add(dst, src)
            | Self::Sub(dst, src)
            | Self::IMul2(dst, src)
            | Self::And(dst, src)
            | Self::Or(dst, src)
            | Self::Xor(dst, src) => (
                src.reg_reads().into_iter().chain(dst.reg_reads()).collect(),
                dst.as_reg().into_iter().copied().collect(),
            ),
            Self::Shift(_, dst, cnt) => (
                dst.reg_reads().into_iter().chain(cnt.reg_reads()).collect(),
                dst.as_reg().into_iter().copied().collect(),
            ),
            Self::IMul1(src) => (
                src.as_reg().map_or_else(
                    || vec![Reg::Physical(PhysReg::A, src.width())],
                    |r| vec![*r, Reg::Physical(PhysReg::A, src.width())],
                ),
                vec![Reg::Physical(PhysReg::A, src.width())],
            ),
            Self::Div(div) | Self::IDiv(div) => match div.width() {
                // TODO: The widths here are hardcoded and wrong in most cases, this shouldn't be an
                // issue though (I hope)
                Width::W8 => (
                    div.as_reg().map_or_else(
                        || vec![Reg::Physical(PhysReg::A, Width::W16)],
                        |r| vec![*r, Reg::Physical(PhysReg::A, Width::W16)],
                    ),
                    vec![Reg::Physical(PhysReg::A, Width::W16)],
                ),
                _ => (
                    div.as_reg().map_or_else(
                        || {
                            vec![
                                Reg::Physical(PhysReg::A, Width::W16),
                                Reg::Physical(PhysReg::D, Width::W16),
                            ]
                        },
                        |r| {
                            vec![
                                *r,
                                Reg::Physical(PhysReg::A, Width::W16),
                                Reg::Physical(PhysReg::D, Width::W16),
                            ]
                        },
                    ),
                    vec![
                        Reg::Physical(PhysReg::A, Width::W16),
                        Reg::Physical(PhysReg::D, Width::W16),
                    ],
                ),
            },
            Self::Cbw | Self::Cwd | Self::Cdq | Self::Cqo => (
                vec![Reg::Physical(PhysReg::A, Width::W16)],
                vec![Reg::Physical(PhysReg::D, Width::W16)],
            ),
            Self::Jmp(_) | Self::Label(_) | Self::Ret | Self::Jcc(_, _) => (vec![], vec![]),
            Self::Push(reg) => (reg.as_reg().map_or_default(|r| vec![*r]), vec![]),
            Self::Pop(reg) => (vec![], reg.as_reg().map_or_default(|r| vec![*r])),
            Self::Cmp(a, b) | Self::Test(a, b) => (
                a.reg_reads().into_iter().chain(b.reg_reads()).collect(),
                vec![],
            ),
            Self::SetCC(_, dst) => (vec![], dst.as_reg().map_or_default(|r| vec![*r])),
            Self::Call { uses, clobbers, .. } => (
                uses.iter().map(|&r| Reg::Physical(r, Width::W64)).collect(),
                clobbers
                    .iter()
                    .map(|&r| Reg::Physical(r, Width::W64))
                    .collect(),
            ),
            Self::LeaGlobal { dst, .. } => (vec![], dst.as_reg().into_iter().copied().collect()),
            Self::ParallelMovs(movs) => (
                movs.iter().flat_map(|(_, src)| src.reg_reads()).collect(),
                movs.iter()
                    .filter_map(|(dst, _)| dst.as_reg().copied())
                    .collect(),
            ),
            Self::BinOp(_, dst, a, b) => (
                a.reg_reads().into_iter().chain(b.reg_reads()).collect(),
                dst.as_reg().into_iter().copied().collect(),
            ),
            Self::Neg(a) => (
                a.reg_reads().into_iter().collect(),
                a.as_reg().into_iter().copied().collect(),
            ),
        }
    }

    pub fn rewrite_registers_with(&mut self, mut f: impl FnMut(&mut Reg)) {
        match self {
            Self::Mov(a, b)
            | Self::Movzx(a, b)
            | Self::Movsx(a, b)
            | Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::IMul2(a, b)
            | Self::And(a, b)
            | Self::Or(a, b)
            | Self::Xor(a, b)
            | Self::Shift(_, a, b)
            | Self::Cmp(a, b)
            | Self::Test(a, b)
            | Self::Lea(a, b) => {
                a.replace_vregs_with(&mut f);
                b.replace_vregs_with(f);
            }
            Self::IMul1(a)
            | Self::Div(a)
            | Self::IDiv(a)
            | Self::Push(a)
            | Self::Pop(a)
            | Self::SetCC(_, a)
            | Self::LeaGlobal { dst: a, .. } => a.replace_vregs_with(f),
            Self::Cbw
            | Self::Cwd
            | Self::Cdq
            | Self::Cqo
            | Self::Jmp(_)
            | Self::Label(_)
            | Self::Ret
            | Self::Jcc(_, _)
            | Self::Call { .. } => {}
            Self::ParallelMovs(movs) => {
                for mov in movs {
                    mov.0.replace_vregs_with(&mut f);
                    mov.1.replace_vregs_with(&mut f);
                }
            }
            Self::BinOp(_, d, a, b) => {
                d.replace_vregs_with(&mut f);
                a.replace_vregs_with(&mut f);
                b.replace_vregs_with(f);
            }
            Self::Neg(a) => a.replace_vregs_with(f),
        }
    }

    pub fn operands(&self) -> Vec<&Operand> {
        match self {
            Self::Mov(a, b)
            | Self::Movzx(a, b)
            | Self::Movsx(a, b)
            | Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::IMul2(a, b)
            | Self::And(a, b)
            | Self::Or(a, b)
            | Self::Xor(a, b)
            | Self::Shift(_, a, b)
            | Self::Cmp(a, b)
            | Self::Test(a, b)
            | Self::Lea(a, b) => vec![a, b],
            Self::IMul1(a)
            | Self::Div(a)
            | Self::IDiv(a)
            | Self::Push(a)
            | Self::Pop(a)
            | Self::SetCC(_, a)
            | Self::Neg(a)
            | Self::LeaGlobal { dst: a, .. } => vec![a],
            Self::BinOp(_, d, a, b) => vec![d, a, b],
            Self::ParallelMovs(m) => m.iter().flat_map(|(d, s)| [d, s]).collect(),
            _ => vec![],
        }
    }

    pub fn operands_mut(&mut self) -> Vec<&mut Operand> {
        match self {
            Self::Mov(a, b)
            | Self::Movzx(a, b)
            | Self::Movsx(a, b)
            | Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::IMul2(a, b)
            | Self::And(a, b)
            | Self::Or(a, b)
            | Self::Xor(a, b)
            | Self::Shift(_, a, b)
            | Self::Cmp(a, b)
            | Self::Test(a, b)
            | Self::Lea(a, b) => vec![a, b],
            Self::IMul1(a)
            | Self::Div(a)
            | Self::IDiv(a)
            | Self::Push(a)
            | Self::Pop(a)
            | Self::SetCC(_, a)
            | Self::Neg(a)
            | Self::LeaGlobal { dst: a, .. } => vec![a],
            Self::BinOp(_, d, a, b) => vec![d, a, b],
            Self::ParallelMovs(m) => m.iter_mut().flat_map(|(d, s)| [d, s]).collect(),
            _ => vec![],
        }
    }
}
