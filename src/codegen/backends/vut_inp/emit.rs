use std::collections::{HashMap, HashSet};

use miette::SourceSpan;

use crate::{
    codegen::backends::vut_inp::{
        INPAssembly,
        lower::{Addr, Base, BlockId, LFunc, LInst, LOp, Operand, Term},
    },
    diagnostics::{Diag, DiagCtx},
    ir::{GlobalInit, IRGlobal, Value},
};

const MEM_SIZE: usize = 8192;
const NT: usize = 8; // T0..T5 divmod, T6 shift counter, T7 copy temp

// `>n d 1 0 0 0` -> `>0 d-r r+1 q 0 0` (for d = 1)
const DIVMOD: &str = "[->-[>+>>]>[[-<+>]+>+>>]<<<<<]";
const WALK_LOAD: &str = "[-[->>+<<]+>>]>[-<+>>+<]>[-<+>]<<<<[->>[-<<+>>]<<<<]";
const WALK_STORE: &str = "[>>[->>+<<]<<-[->>+<<]+>>]>[-]>[-<+>]<<<<[-<<]";

struct Em {
    out: String,
    pos: usize,
    k: usize,
}

impl Em {
    fn raw(&mut self, s: &str) {
        self.out.push_str(s);
    }

    fn goto(&mut self, c: usize) {
        let ch = if c > self.pos { '>' } else { '<' };
        self.out
            .extend(std::iter::repeat_n(ch, c.abs_diff(self.pos)));
        self.pos = c;
    }

    fn bump(&mut self, c: usize, d: i32) {
        if d == 0 {
            return;
        }
        self.goto(c);
        let ch = if d > 0 { '+' } else { '-' };
        self.out
            .extend(std::iter::repeat_n(ch, d.unsigned_abs() as usize));
    }

    fn clear(&mut self, c: usize) {
        self.goto(c);
        self.raw("[-]");
    }

    fn lp(&mut self, c: usize, body: impl FnOnce(&mut Self)) {
        self.goto(c);
        self.raw("[");
        body(self);
        self.goto(c);
        self.raw("]");
    }

    fn add(&mut self, c: usize, k: u8) {
        let (m, s) = if k <= 128 {
            (k as i32, 1)
        } else {
            (256 - k as i32, -1)
        };
        let d = c.abs_diff(self.k) as i32;
        let best = (2..=16)
            .filter_map(|a: i32| {
                let b = (m + a / 2) / a;
                let r = m - a * b;
                let cost = a + b + r.abs() + 3 + 3 * d;
                (b > 1 && cost < m).then_some((cost, a, b, r))
            })
            .min();
        match best {
            Some((_, a, b, r)) => {
                let kc = self.k;
                self.bump(kc, a);
                self.lp(kc, |e| {
                    e.bump(kc, -1);
                    e.bump(c, s * b);
                });
                self.bump(c, s * r);
            }
            None => self.bump(c, s * m),
        }
    }

    fn mv(&mut self, src: usize, dsts: &[(usize, u8)]) {
        self.lp(src, |e| {
            e.bump(src, -1);
            for &(d, m) in dsts {
                e.add(d, m);
            }
        });
    }

    fn if_else(&mut self, x: usize, then: impl FnOnce(&mut Self), els: impl FnOnce(&mut Self)) {
        self.bump(x + 1, 1);
        self.goto(x);
        self.raw("[");
        then(self);
        self.goto(x);
        self.raw(">-]>[<");
        self.pos = x;
        els(self);
        self.goto(x);
        self.raw(">->]<<");
        self.pos = x;
    }
}

struct Alloc {
    color: HashMap<Value, usize>,
    n: usize,
    dies: Vec<Vec<HashSet<Value>>>,
    br_dies: Vec<bool>,
}

fn defs(i: &LInst) -> Vec<Value> {
    let mut d: Vec<Value> = i.dst.into_iter().collect();
    if let LOp::DivMod { q, r, .. } = &i.op {
        d.extend(q.iter().chain(r).copied());
    }
    d
}

fn edge(adj: &mut HashMap<Value, HashSet<Value>>, a: Value, b: Value) {
    if a != b {
        adj.entry(a).or_default().insert(b);
        adj.entry(b).or_default().insert(a);
    }
}

fn alloc(f: &LFunc) -> Alloc {
    let nb = f.blocks.len();
    let mut ue = vec![HashSet::new(); nb];
    let mut kill = vec![HashSet::new(); nb];
    for (b, blk) in f.blocks.iter().enumerate() {
        for i in &blk.insts {
            for v in i.op.uses() {
                if !kill[b].contains(&v) {
                    ue[b].insert(v);
                }
            }
            kill[b].extend(defs(i));
        }
        if let Term::Br(c, ..) = blk.term
            && !kill[b].contains(&c)
        {
            ue[b].insert(c);
        }
    }

    let mut live_in: Vec<HashSet<Value>> = vec![HashSet::new(); nb];
    let mut live_out = live_in.clone();
    loop {
        let mut changed = false;
        for b in (0..nb).rev() {
            let out: HashSet<Value> = f.blocks[b]
                .term
                .succs()
                .into_iter()
                .flat_map(|s| live_in[s].iter().copied())
                .collect();
            let inn: HashSet<Value> = ue[b]
                .iter()
                .copied()
                .chain(out.difference(&kill[b]).copied())
                .collect();
            changed |= inn != live_in[b];
            live_in[b] = inn;
            live_out[b] = out;
        }
        if !changed {
            break;
        }
    }

    let mut adj: HashMap<Value, HashSet<Value>> = HashMap::new();
    let mut hints: HashMap<Value, Value> = HashMap::new();
    let mut dies = Vec::with_capacity(nb);
    let mut br_dies = vec![false; nb];
    for (b, blk) in f.blocks.iter().enumerate() {
        let mut live = live_out[b].clone();
        if let Term::Br(c, ..) = blk.term {
            br_dies[b] = !live.contains(&c);
            live.insert(c);
        }
        let mut db = vec![HashSet::new(); blk.insts.len()];
        for (idx, i) in blk.insts.iter().enumerate().rev() {
            let ds = defs(i);
            for &d in &ds {
                adj.entry(d).or_default();
                for &l in &live {
                    edge(&mut adj, d, l);
                }
                for &d2 in &ds {
                    edge(&mut adj, d, d2);
                }
            }
            for d in &ds {
                live.remove(d);
            }
            let uses: Vec<Value> = i.op.uses().collect();
            db[idx] = uses.iter().copied().filter(|u| !live.contains(u)).collect();
            if let (Some(&d), Some(&h)) = (ds.first(), uses.iter().find(|u| db[idx].contains(*u))) {
                hints.insert(d, h);
            }
            live.extend(uses);
        }
        dies.push(db);
    }
    for &a in &live_in[0] {
        adj.entry(a).or_default();
        for &b in &live_in[0] {
            edge(&mut adj, a, b);
        }
    }

    let mut order: Vec<Value> = live_in[0].iter().copied().collect();
    order.sort_unstable();
    for blk in &f.blocks {
        for i in &blk.insts {
            order.extend(defs(i));
        }
    }
    let mut color = HashMap::new();
    let mut n = 0;
    for v in order {
        let taken: HashSet<usize> = adj
            .get(&v)
            .into_iter()
            .flatten()
            .filter_map(|x| color.get(x).copied())
            .collect();
        let c = hints
            .get(&v)
            .and_then(|h| color.get(h).copied())
            .filter(|c| !taken.contains(c))
            .unwrap_or_else(|| (0..).find(|c| !taken.contains(c)).unwrap());
        color.insert(v, c);
        n = n.max(c + 1);
    }

    Alloc {
        color,
        n,
        dies,
        br_dies,
    }
}

#[derive(Debug, Clone, Copy)]
struct Obj {
    cell: usize,
    size: usize,
    strided: bool,
}

impl Obj {
    fn elem(&self, j: usize) -> usize {
        if self.strided {
            self.cell + 3 + 2 * j
        } else {
            self.cell + j
        }
    }

    fn trail(&self, j: usize) -> usize {
        self.cell + 2 + 2 * j
    }

    fn cells(&self) -> usize {
        if self.strided {
            2 * self.size + 3
        } else {
            self.size
        }
    }
}

struct Layout {
    run: usize,
    flag: HashMap<BlockId, usize>,
    p: usize,
    t: usize,
    vals: usize,
    objs: HashMap<Base, Obj>,
}

struct Uses(HashMap<Value, usize>);

impl Uses {
    fn one(&mut self, v: Value) -> bool {
        match self.0.get_mut(&v) {
            Some(n) => {
                *n = n.saturating_sub(1);
                *n == 0
            }
            None => false,
        }
    }

    fn all(&mut self, v: Value) -> bool {
        self.0.remove(&v).is_some()
    }
}

struct Gen<'a> {
    e: Em,
    l: &'a Layout,
    alloc: &'a Alloc,
    ctx: &'a mut DiagCtx,
    p_val: Option<u8>,
    dispatch: bool,
}

impl Gen<'_> {
    fn t(&self, i: usize) -> usize {
        self.l.t + i
    }

    fn aux(&self) -> usize {
        self.t(7)
    }

    fn cell(&self, v: Value) -> usize {
        self.l.vals + self.alloc.color[&v]
    }

    fn err(&mut self, span: Option<SourceSpan>, title: &'static str, msg: &'static str) {
        self.ctx.emit(Diag::error(
            title,
            span.unwrap_or_else(|| (0, 0).into()),
            msg,
        ));
    }

    fn copy(&mut self, src: usize, dst: usize) {
        let a = self.aux();
        self.e.mv(src, &[(dst, 1), (a, 1)]);
        self.e.mv(a, &[(src, 1)]);
    }

    // dst += o
    fn load(&mut self, o: Operand, dst: usize, u: &mut Uses) {
        match o {
            Operand::I(c) => self.e.add(dst, c),
            Operand::V(v) => {
                let cv = self.cell(v);
                if u.one(v) {
                    self.e.mv(cv, &[(dst, 1)])
                } else {
                    self.copy(cv, dst)
                }
            }
        }
    }

    fn scratch(&mut self, o: Operand, t: usize, u: &mut Uses) -> usize {
        match o {
            Operand::V(v) => {
                let cv = self.cell(v);
                if u.one(v) {
                    return cv;
                }
                self.copy(cv, t);
            }
            Operand::I(c) => self.e.add(t, c),
        }
        t
    }

    fn store_result(&mut self, dst: Value, src: usize) {
        let dc = self.cell(dst);
        self.e.clear(dc);
        self.e.mv(src, &[(dc, 1)]);
    }

    fn addr(&mut self, a: &Addr, span: Option<SourceSpan>) -> Option<usize> {
        let l = self.l;
        match l.objs.get(&a.base) {
            Some(o) if (0..o.size as i64).contains(&a.off) => Some(o.elem(a.off as usize)),
            Some(_) => {
                self.err(
                    span,
                    "out of bounds access",
                    "this address is outside of the object",
                );
                None
            }
            None => {
                self.err(
                    span,
                    "unsupported address",
                    "only globals and stack slots can be addressed",
                );
                None
            }
        }
    }

    fn walk_start(&mut self, a: &Addr, span: Option<SourceSpan>) -> Option<usize> {
        self.addr(a, span)?;
        Some(self.l.objs[&a.base].trail(a.off as usize))
    }

    fn load_idx(&mut self, dst: Value, a: &Addr, i: Value, span: Option<SourceSpan>, u: &mut Uses) {
        let Some(t) = self.walk_start(a, span) else {
            return;
        };
        self.load(Operand::V(i), t, u);
        self.e.goto(t);
        self.e.raw(WALK_LOAD);
        self.e.pos = t - 2;
        self.store_result(dst, t);
    }

    fn store_idx(
        &mut self,
        a: &Addr,
        i: Value,
        v: Operand,
        span: Option<SourceSpan>,
        u: &mut Uses,
    ) {
        let Some(t) = self.walk_start(a, span) else {
            return;
        };
        self.load(Operand::V(i), t, u);
        self.load(v, t + 2, u);
        self.e.goto(t);
        self.e.raw(WALK_STORE);
        self.e.pos = t - 2;
    }

    // dst = sum(o*mult)
    fn lin(&mut self, dst: Value, terms: &[(Operand, u8)], u: &mut Uses) {
        let dc = self.cell(dst);
        let mut k = 0u8;
        let mut vs: Vec<(Value, u8)> = Vec::new();
        for &(o, m) in terms {
            match o {
                Operand::I(c) => k = k.wrapping_add(c.wrapping_mul(m)),
                Operand::V(v) => match vs.iter_mut().find(|e| e.0 == v) {
                    Some(e) => e.1 = e.1.wrapping_add(m),
                    None => vs.push((v, m)),
                },
            }
        }
        let mut srcs = Vec::new();
        match vs.iter().position(|&(v, _)| self.cell(v) == dc) {
            Some(i) => {
                let (v, m) = vs.remove(i);
                u.all(v);
                match m {
                    1 => {}
                    0 => self.e.clear(dc),
                    _ => {
                        let t0 = self.t(0);
                        self.e.mv(dc, &[(t0, 1)]);
                        srcs.push((t0, m, true));
                    }
                }
            }
            None => self.e.clear(dc),
        }
        for (v, m) in vs {
            let destroy = u.all(v);
            if m != 0 {
                srcs.push((self.cell(v), m, destroy));
            }
        }
        for (c, m, destroy) in srcs {
            if destroy {
                self.e.mv(c, &[(dc, m)]);
            } else {
                let a = self.aux();
                self.e.mv(c, &[(dc, m), (a, 1)]);
                self.e.mv(a, &[(c, 1)]);
            }
        }
        self.e.add(dc, k);
    }

    fn mul(&mut self, dst: Value, x: Value, y: Value, u: &mut Uses) {
        let (t0, t1, t2) = (self.t(0), self.t(1), self.t(2));
        self.load(Operand::V(x), t0, u);
        self.load(Operand::V(y), t1, u);
        let dc = self.cell(dst);
        self.e.clear(dc);
        self.e.lp(t0, |e| {
            e.bump(t0, -1);
            e.mv(t1, &[(dc, 1), (t2, 1)]);
            e.mv(t2, &[(t1, 1)]);
        });
        self.e.clear(t1);
    }

    fn divmod(&mut self, n: Operand, d: Operand, q: Option<Value>, r: Option<Value>, u: &mut Uses) {
        let t: [usize; 4] = std::array::from_fn(|i| self.t(i));
        self.load(n, t[0], u);
        self.load(d, t[1], u);
        self.e.bump(t[2], 1);
        self.e.goto(t[0]);
        self.e.raw(DIVMOD);
        self.e.clear(t[1]);
        match r {
            Some(r) => {
                self.e.bump(t[2], -1);
                self.store_result(r, t[2]);
            }
            None => self.e.clear(t[2]),
        }
        match q {
            Some(q) => self.store_result(q, t[3]),
            None => self.e.clear(t[3]),
        }
    }

    fn lt(&mut self, dst: Value, a: Operand, b: Operand, u: &mut Uses) {
        let (t0, t3, t4) = (self.t(0), self.t(3), self.t(4));
        self.load(a, t0, u);
        let bc = self.scratch(b, t3, u);
        // while b { if a {a--; b--} else {r=1; b=0} }
        self.e.lp(bc, |e| {
            e.if_else(
                t0,
                |e| {
                    e.bump(t0, -1);
                    e.bump(bc, -1);
                },
                |e| {
                    e.bump(t4, 1);
                    e.clear(bc);
                },
            );
        });
        self.e.clear(t0);
        self.store_result(dst, t4);
    }

    fn is_zero(&mut self, dst: Value, v: Value, u: &mut Uses) {
        let t1 = self.t(1);
        let x = self.scratch(Operand::V(v), self.t(0), u);
        self.e.bump(t1, 1);
        self.e.lp(x, |e| {
            e.clear(x);
            e.bump(t1, -1);
        });
        self.store_result(dst, t1);
    }

    fn shift(&mut self, dst: Value, a: Operand, y: Value, right: bool, u: &mut Uses) {
        let t: [usize; 7] = std::array::from_fn(|i| self.t(i));
        self.load(a, t[0], u);
        let cnt = self.scratch(Operand::V(y), t[6], u);
        self.e.lp(cnt, |e| {
            e.bump(cnt, -1);
            if right {
                e.bump(t[1], 2);
                e.bump(t[2], 1);
                e.goto(t[0]);
                e.raw(DIVMOD);
                e.clear(t[1]);
                e.clear(t[2]);
                e.mv(t[3], &[(t[0], 1)]);
            } else {
                e.mv(t[0], &[(t[1], 2)]);
                e.mv(t[1], &[(t[0], 1)]);
            }
        });
        self.store_result(dst, t[0]);
    }

    fn put_const(&mut self, c: u8) {
        let p = self.l.p;
        match self.p_val {
            Some(old) => self.e.add(p, c.wrapping_sub(old)),
            None => {
                self.e.clear(p);
                self.e.add(p, c);
            }
        }
        self.p_val = Some(c);
        self.e.goto(p);
        self.e.raw(".");
    }

    fn inst(&mut self, i: &LInst, u: &mut Uses) {
        use Operand::*;
        match &i.op {
            LOp::Add(x, b) => self.lin(i.dst.unwrap(), &[(V(*x), 1), (*b, 1)], u),
            LOp::Sub(a, y) => self.lin(i.dst.unwrap(), &[(*a, 1), (V(*y), 255)], u),
            LOp::Mul(x, I(c)) => self.lin(i.dst.unwrap(), &[(V(*x), *c)], u),
            LOp::Mul(x, V(y)) => self.mul(i.dst.unwrap(), *x, *y, u),
            LOp::DivMod { n, d, q, r } => self.divmod(*n, *d, *q, *r, u),
            LOp::Lt(a, b) => self.lt(i.dst.unwrap(), *a, *b, u),
            LOp::IsZero(v) => self.is_zero(i.dst.unwrap(), *v, u),
            LOp::Shl(a, y) => self.shift(i.dst.unwrap(), *a, *y, false, u),
            LOp::LShr(a, y) => self.shift(i.dst.unwrap(), *a, *y, true, u),
            LOp::Load(a) => {
                if let Some(ca) = self.addr(a, i.span) {
                    let dc = self.cell(i.dst.unwrap());
                    self.e.clear(dc);
                    self.copy(ca, dc);
                }
            }
            LOp::LoadIdx(a, idx) => self.load_idx(i.dst.unwrap(), a, *idx, i.span, u),
            LOp::Store(a, o) => {
                if let Some(ca) = self.addr(a, i.span) {
                    self.e.clear(ca);
                    self.load(*o, ca, u);
                }
            }
            LOp::StoreIdx(a, idx, v) => self.store_idx(a, *idx, *v, i.span, u),
            LOp::GetChar=> {
                let dc = self.cell(i.dst.unwrap());
                self.e.goto(dc);
                self.e.raw(",");
            }
            LOp::PutChar(V(v)) => {
                let cv = self.cell(*v);
                self.e.goto(cv);
                self.e.raw(".");
            }
            LOp::PutChar(I(c)) => self.put_const(*c),
            LOp::And(..) | LOp::Or(..) | LOp::Xor(..) => self.err(i.span, "unsupported operation", "bitwise and/or/xor of runtime values is not yet supported"),
            LOp::AddrOf(_) | LOp::PtrAdd(..) | LOp::PtrAddImm(..) | LOp::LoadDyn(..) | LOp::StoreDyn(..) => self.err(i.span, "unsupported operation", "pointers must be a global/stack slot plus constants and at most one runtime i8 index"),
        }
    }

    fn term(&mut self, t: Term, c_dies: bool) {
        match t {
            Term::Jmp(b) => {
                let fl = self.l.flag[&b];
                self.e.bump(fl, 1);
            }
            Term::Br(c, tb, fb) => {
                let (ft, ff) = (self.l.flag[&tb], self.l.flag[&fb]);
                let mut u = Uses(if c_dies {
                    HashMap::from([(c, 1)])
                } else {
                    HashMap::new()
                });
                let x = self.scratch(Operand::V(c), self.t(0), &mut u);
                // ff = 1; if x { ff = 0; ft = 1 }
                self.e.bump(ff, 1);
                self.e.lp(x, |e| {
                    e.clear(x);
                    e.bump(ff, -1);
                    e.bump(ft, 1);
                });
            }
            Term::Halt => {
                if self.dispatch {
                    self.e.bump(self.l.run, -1);
                }
            }
        }
    }

    fn block(&mut self, f: &LFunc, b: BlockId, p_val: Option<u8>) {
        self.p_val = p_val;
        let alloc = self.alloc;
        let blk = &f.blocks[b];
        for (idx, i) in blk.insts.iter().enumerate() {
            let dies = &alloc.dies[b][idx];
            let mut left = HashMap::new();
            for v in i.op.uses() {
                if dies.contains(&v) {
                    *left.entry(v).or_insert(0) += 1;
                }
            }
            self.inst(i, &mut Uses(left));
        }
        self.term(blk.term, alloc.br_dies[b]);
    }
}

pub fn emit(f: &LFunc, globals: &[IRGlobal], ctx: &mut DiagCtx) -> Option<INPAssembly> {
    if f.blocks.is_empty() {
        return Some(INPAssembly {
            program: String::new(),
            data: Vec::new(),
        });
    }
    let alloc = alloc(f);

    let entry_outside = !f.blocks.iter().any(|b| b.term.succs().contains(&0));
    let loop_blocks: Vec<BlockId> = (0..f.blocks.len())
        .filter(|&b| b != 0 || !entry_outside)
        .collect();
    let dispatch = !loop_blocks.is_empty();

    let mut used: HashSet<&Base> = HashSet::new();
    let mut strided: HashSet<&Base> = HashSet::new();
    for i in f.blocks.iter().flat_map(|b| &b.insts) {
        match &i.op {
            LOp::Load(a) | LOp::Store(a, _) | LOp::AddrOf(a) => {
                used.insert(&a.base);
            }
            LOp::LoadIdx(a, _) | LOp::StoreIdx(a, ..) => {
                used.insert(&a.base);
                strided.insert(&a.base);
            }
            _ => {}
        }
    }

    let mut next = 0;
    let mut take = |n: usize| {
        next += n;
        next - n
    };
    let run = take(1);
    let flag: HashMap<BlockId, usize> = loop_blocks.iter().map(|&b| (b, take(1))).collect();
    let p = take(1);
    let k = take(1);
    let t = take(NT);
    let vals = take(alloc.n);
    let mut objects: Vec<(Base, usize)> = Vec::new();
    for g in globals {
        let base = Base::Global(g.name.clone());
        if used.contains(&base) {
            let size = match &g.init {
                GlobalInit::Bytes(b) => b.len(),
                GlobalInit::Zeroed(n) => *n as usize,
            };
            objects.push((base, size));
        }
    }
    for &(id, size) in &f.stack {
        objects.push((Base::Stack(id), size as usize));
    }
    objects.sort_by_key(|(base, size)| (strided.contains(base), *size));
    let mut objs = HashMap::new();
    for (base, size) in objects {
        let mut o = Obj {
            cell: 0,
            size,
            strided: strided.contains(&base),
        };
        o.cell = take(o.cells());
        objs.insert(base, o);
    }
    let total = next;
    let l = Layout {
        run,
        flag,
        p,
        t,
        vals,
        objs,
    };

    let mut g = Gen {
        e: Em {
            out: String::new(),
            pos: 0,
            k,
        },
        l: &l,
        alloc: &alloc,
        ctx: &mut *ctx,
        p_val: None,
        dispatch,
    };
    if entry_outside {
        g.block(f, 0, Some(0));
    }
    if dispatch {
        g.e.goto(l.run);
        g.e.raw("[");
        for &b in &loop_blocks {
            let fl = l.flag[&b];
            g.e.goto(fl);
            g.e.raw("[");
            g.e.bump(fl, -1);
            g.block(f, b, None);
            g.e.goto(fl);
            g.e.raw("]");
        }
        g.e.goto(l.run);
        g.e.raw("]");
    }
    let program = g.e.out;

    let mut data = vec![0u8; total];
    if dispatch {
        data[l.run] = 1;
        if !entry_outside {
            data[l.flag[&0]] = 1;
        }
    }
    for gl in globals {
        if let (GlobalInit::Bytes(b), Some(o)) =
            (&gl.init, l.objs.get(&Base::Global(gl.name.clone())))
        {
            for (j, &byte) in b.iter().enumerate() {
                data[o.elem(j)] = byte;
            }
        }
    }

    if program.len() + 1 + data.len() > MEM_SIZE {
        ctx.emit(Diag::warning(
            "program too large",
            (0, 0).into(),
            "code + `@` + data exceeded 8 KiB of memory",
        ));
    }
    if ctx.has_errors() {
        return None;
    }
    Some(INPAssembly { program, data })
}
