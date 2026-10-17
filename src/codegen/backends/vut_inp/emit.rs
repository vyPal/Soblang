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
const NP: usize = 10;
const LAYOUT_ROUNDS: usize = 6;

// `>n d 1 0 0 0` -> `>0 d-r r+1 q 0 0` (for d = 1)
const DIVMOD: &str = "[->-[>+>>]>[[-<+>]+>+>>]<<<<<]";
const WALK_LOAD: &str = "[-[->>+<<]+>>]>[-<+>>+<]>[-<+>]<<<<[->>[-<<+>>]<<<<]";
const WALK_STORE: &str = "[>>[->>+<<]<<-[->>+<<]+>>]>[-]>[-<+>]<<<<[-<<]";

#[derive(Debug, Clone)]
enum Tok {
    Go(usize),
    Txt(&'static str),
    At(usize),
}

struct Em<'p> {
    toks: Vec<Tok>,
    pos: usize,
    k: usize,
    phys: &'p [usize],
    zero: HashSet<usize>,
    touched: HashSet<usize>,
}

impl Em<'_> {
    fn dist(&self, a: usize, b: usize) -> usize {
        self.phys[a].abs_diff(self.phys[b])
    }

    fn txt(&mut self, s: &'static str) {
        self.toks.push(Tok::Txt(s));
    }

    fn mark(&mut self, c: usize) {
        self.touched.insert(c);
        self.zero.remove(&c);
    }

    fn raw(&mut self, s: &'static str, end: usize, cells: impl IntoIterator<Item = usize>) {
        self.txt(s);
        self.pos = end;
        self.toks.push(Tok::At(end));
        for c in cells {
            self.mark(c);
        }
    }

    fn goto(&mut self, c: usize) {
        if c != self.pos {
            self.toks.push(Tok::Go(c));
            self.pos = c;
        }
    }

    fn bump(&mut self, c: usize, d: i32) {
        if d == 0 {
            return;
        }
        self.goto(c);
        for _ in 0..d.unsigned_abs() {
            self.txt(if d > 0 { "+" } else { "-" });
        }
        self.mark(c);
    }

    fn clear(&mut self, c: usize) {
        if self.zero.contains(&c) {
            return;
        }
        self.goto(c);
        self.txt("[-]");
        self.touched.insert(c);
        self.zero.insert(c);
    }

    fn lp(&mut self, c: usize, body: impl Fn(&mut Self)) {
        self.goto(c);
        self.txt("[");
        let (len, pos) = (self.toks.len(), self.pos);
        let pre = std::mem::take(&mut self.zero);
        let outer = std::mem::take(&mut self.touched);

        body(self);
        let dry = std::mem::take(&mut self.touched);
        self.toks.truncate(len);
        self.pos = pos;

        self.zero = pre
            .iter()
            .copied()
            .filter(|x| !dry.contains(x) && *x != c)
            .collect();
        body(self);
        self.goto(c);
        self.txt("]");

        let wrote = std::mem::replace(&mut self.touched, outer);
        self.zero.retain(|x| pre.contains(x));
        self.zero.insert(c);
        self.touched.extend(wrote);
        self.touched.insert(c);
    }

    fn add(&mut self, c: usize, k: u8) {
        let (m, s) = if k <= 128 {
            (k as i32, 1)
        } else {
            (256 - k as i32, -1)
        };
        if m < 10 {
            return self.bump(c, s * m); // a loop can't win
        }
        let kc = self
            .zero
            .iter()
            .copied()
            .filter(|&z| z != c)
            .min_by_key(|&z| (self.dist(c, z), z))
            .unwrap_or(self.k);
        let d = self.dist(c, kc) as i32;
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

    fn if_else(&mut self, x: usize, then: impl Fn(&mut Self), els: impl Fn(&mut Self)) {
        self.bump(x + 1, 1);
        let pre = self.zero.clone();
        let outer = std::mem::take(&mut self.touched);
        self.goto(x);
        self.txt("[");
        self.zero.remove(&x);
        then(self);
        self.goto(x);
        self.raw(">-]>[<", x, [x + 1, x]);
        let then_end = std::mem::replace(&mut self.zero, pre);
        self.zero.insert(x);
        els(self);
        self.goto(x);
        self.raw(">->]<<", x, [x + 1]);
        let wrote = std::mem::replace(&mut self.touched, outer);
        self.zero.retain(|c| then_end.contains(c));
        self.zero.insert(x + 1);
        self.touched.extend(wrote);
    }
}

struct Alloc {
    color: HashMap<Value, usize>,
    alias: HashMap<Value, Addr>,
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

fn aliases(f: &LFunc, live_out: &[HashSet<Value>]) -> HashMap<Value, Addr> {
    let mut cand: HashMap<Value, Addr> = HashMap::new();
    for i in f.blocks.iter().flat_map(|b| &b.insts) {
        if let (LOp::Load(a), Some(d)) = (&i.op, i.dst)
            && a.base != Base::Abs
        {
            cand.insert(d, a.clone());
        }
    }
    let mut bad = HashSet::new();
    for (b, blk) in f.blocks.iter().enumerate() {
        let mut live = live_out[b].clone();
        if let Term::Br(c, ..) = blk.term {
            live.insert(c);
        }
        for i in blk.insts.iter().rev() {
            let clobbers = |a: &Addr| match &i.op {
                LOp::Store(w, _) => w == a,
                LOp::StoreIdx(w, ..) => w.base == a.base,
                LOp::StoreDyn(..) => true,
                _ => false,
            };
            for (v, a) in &cand {
                if clobbers(a) && (live.contains(v) || i.op.uses().any(|u| u == *v)) {
                    bad.insert(*v);
                }
            }
            for d in defs(i) {
                live.remove(&d);
            }
            live.extend(i.op.uses());
        }
    }
    let mut uses: HashMap<Value, usize> = HashMap::new();
    let mut walked = HashSet::new();
    for blk in &f.blocks {
        for i in &blk.insts {
            for u in i.op.uses() {
                *uses.entry(u).or_default() += 1;
            }
            if let LOp::LoadIdx(a, _) | LOp::StoreIdx(a, ..) = &i.op {
                walked.insert(a.base.clone());
            }
        }
        if let Term::Br(c, ..) = blk.term {
            *uses.entry(c).or_default() += 1;
        }
    }
    cand.retain(|v, a| {
        !bad.contains(v) && (uses.get(v).copied().unwrap_or(0) <= 1 || !walked.contains(&a.base))
    });
    cand
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

    let alias = aliases(f, &live_out);

    let mut adj: HashMap<Value, HashSet<Value>> = HashMap::new();
    let mut hints: HashMap<Value, Value> = HashMap::new();
    let mut dies = Vec::with_capacity(nb);
    let mut br_dies = vec![false; nb];
    for (b, blk) in f.blocks.iter().enumerate() {
        let mut live = live_out[b].clone();
        if let Term::Br(c, ..) = blk.term {
            br_dies[b] = !live.contains(&c) && !alias.contains_key(&c);
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
            db[idx] = uses
                .iter()
                .copied()
                .filter(|u| !live.contains(u) && !alias.contains_key(u))
                .collect();
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
        if alias.contains_key(&v) {
            continue;
        }
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
        alias,
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

struct Shape {
    rpo: Vec<usize>,
    merge: Vec<bool>,
    header: Vec<bool>,
    kids: Vec<Vec<BlockId>>,
    dominated: Vec<Vec<BlockId>>,
}

impl Shape {
    fn back(&self, from: BlockId, to: BlockId) -> bool {
        self.rpo[to] <= self.rpo[from]
    }
}

fn shape(f: &LFunc) -> Option<Shape> {
    let n = f.blocks.len();
    let succs: Vec<Vec<BlockId>> = f.blocks.iter().map(|b| b.term.succs()).collect();
    let order = rpo(&succs);
    let mut rpo_n = vec![usize::MAX; n];
    for (i, &b) in order.iter().enumerate() {
        rpo_n[b] = i;
    }
    let mut preds = vec![Vec::new(); n];
    for (b, ss) in succs.iter().enumerate() {
        for &s in ss {
            preds[s].push(b);
        }
    }

    let mut idom = vec![usize::MAX; n];
    idom[order[0]] = order[0];
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &order[1..] {
            let mut new = usize::MAX;
            for &p in &preds[b] {
                if idom[p] == usize::MAX {
                    continue;
                }
                new = if new == usize::MAX {
                    p
                } else {
                    let (mut a, mut c) = (p, new);
                    while a != c {
                        while rpo_n[a] > rpo_n[c] {
                            a = idom[a];
                        }
                        while rpo_n[c] > rpo_n[a] {
                            c = idom[c];
                        }
                    }
                    a
                };
            }
            if idom[b] != new {
                idom[b] = new;
                changed = true;
            }
        }
    }
    let dominates = |a: BlockId, mut b: BlockId| loop {
        if a == b {
            return true;
        }
        if b == order[0] {
            return false;
        }
        b = idom[b];
    };

    let mut fwd = vec![0; n];
    let mut header = vec![false; n];
    for (b, ss) in succs.iter().enumerate() {
        for &s in ss {
            if rpo_n[s] <= rpo_n[b] {
                if !dominates(s, b) {
                    return None;
                }
                header[s] = true;
            } else {
                fwd[s] += 1;
            }
        }
    }
    let merge: Vec<bool> = fwd.iter().map(|&k| k >= 2).collect();
    let mut kids = vec![Vec::new(); n];
    for &b in &order[1..] {
        if merge[b] {
            kids[idom[b]].push(b);
        }
    }
    let dominated = (0..n)
        .map(|a| order.iter().copied().filter(|&b| dominates(a, b)).collect())
        .collect();
    Some(Shape {
        rpo: rpo_n,
        merge,
        header,
        kids,
        dominated,
    })
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

struct Layout {
    run: usize,
    flag: HashMap<BlockId, usize>,
    p: Vec<usize>,
    pc: Vec<u8>,
    k: usize,
    t: usize,
    vals: usize,
    objs: HashMap<Base, Obj>,
    total: usize,
    units: Vec<(usize, usize)>,
    lcell: HashMap<BlockId, usize>,
    xcell: HashMap<BlockId, (usize, usize)>,
    inv_zero: HashSet<usize>,
    rest_zero: HashSet<usize>,
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

type Err = (Option<SourceSpan>, &'static str, &'static str);
type PState = Vec<Option<u8>>;
type PIn = HashMap<usize, PState>;
type PStats = (
    HashMap<usize, Vec<PState>>,
    HashMap<(usize, usize), Vec<u8>>,
);

struct Gen<'a> {
    e: Em<'a>,
    l: &'a Layout,
    alloc: &'a Alloc,
    shape: Option<&'a Shape>,
    errs: Vec<Err>,
    p_val: Vec<Option<u8>>,
    p_in: &'a PIn,
    p_seen: HashMap<usize, Vec<PState>>,
    p_entry: Vec<Option<usize>>,
    p_need: HashMap<(usize, usize), Vec<u8>>,
    dispatch: bool,
}

impl Gen<'_> {
    fn t(&self, i: usize) -> usize {
        self.l.t + i
    }

    fn aux(&self, src: usize, dsts: &[usize]) -> usize {
        let e = &self.e;
        let cost = |a: usize| {
            let mut c = e.dist(src, dsts[0]) + e.dist(dsts[dsts.len() - 1], a) + 4 * e.dist(a, src);
            for w in dsts.windows(2) {
                c += e.dist(w[0], w[1]);
            }
            (c, a)
        };
        e.zero
            .iter()
            .copied()
            .filter(|&a| a != src && a != self.l.k && !dsts.contains(&a))
            .map(cost)
            .min()
            .map(|(_, a)| a)
            .unwrap_or(self.t(7))
    }

    fn cell(&self, v: Value) -> usize {
        match self.alloc.alias.get(&v) {
            Some(a) => self
                .l
                .objs
                .get(&a.base)
                .map_or(0, |o| o.elem((a.off.max(0) as usize).min(o.size - 1))),
            None => self.l.vals + self.alloc.color[&v],
        }
    }

    fn err(&mut self, span: Option<SourceSpan>, title: &'static str, msg: &'static str) {
        self.errs.push((span, title, msg));
    }

    fn copy(&mut self, src: usize, dst: usize) {
        let a = self.aux(src, &[dst]);
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

    fn walk_start(&mut self, a: &Addr, span: Option<SourceSpan>) -> Option<(usize, Obj)> {
        self.addr(a, span)?;
        let o = self.l.objs[&a.base];
        Some((o.trail(a.off as usize), o))
    }

    fn load_idx(&mut self, dst: Value, a: &Addr, i: Value, span: Option<SourceSpan>, u: &mut Uses) {
        let Some((t, o)) = self.walk_start(a, span) else {
            return;
        };
        self.load(Operand::V(i), t, u);
        self.e.goto(t);
        self.e.raw(WALK_LOAD, t - 2, o.cell..o.cell + o.cells());
        self.trails_zero(o, Some(t));
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
        let Some((t, o)) = self.walk_start(a, span) else {
            return;
        };
        self.load(Operand::V(i), t, u);
        self.load(v, t + 2, u);
        self.e.goto(t);
        self.e.raw(WALK_STORE, t - 2, o.cell..o.cell + o.cells());
        self.trails_zero(o, None);
    }

    fn trails_zero(&mut self, o: Obj, except: Option<usize>) {
        let cells = (0..=o.size).map(|j| o.trail(j)).chain([o.cell]);
        self.e.zero.extend(cells.filter(|&c| Some(c) != except));
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
                let a = self.aux(c, &[dc]);
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

    fn divmod_raw(&mut self) {
        let t0 = self.t(0);
        self.e.goto(t0);
        self.e.raw(DIVMOD, t0, t0..t0 + 6);
        self.e.zero.extend([t0, t0 + 4, t0 + 5]);
    }

    fn divmod(&mut self, n: Operand, d: Operand, q: Option<Value>, r: Option<Value>, u: &mut Uses) {
        let t: [usize; 4] = std::array::from_fn(|i| self.t(i));
        self.load(n, t[0], u);
        self.load(d, t[1], u);
        self.e.bump(t[2], 1);
        self.divmod_raw();
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
                e.raw(DIVMOD, t[0], t[0]..t[0] + 6);
                e.zero.extend([t[0], t[4], t[5]]);
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

    /// Brings print cell `i` to `c`.
    fn set_p(&mut self, i: usize, c: u8) {
        if let Some(fl) = self.p_entry[i].take() {
            self.p_need.entry((fl, i)).or_default().push(c);
        }
        let p = self.l.p[i];
        match self.p_val[i] {
            Some(old) => self.e.add(p, c.wrapping_sub(old)),
            None => {
                self.e.clear(p);
                self.e.add(p, c);
            }
        }
        self.p_val[i] = Some(c);
    }

    fn put_const(&mut self, c: u8) {
        let cost = |g: &Self, i: usize| {
            let adj = match g.p_val[i] {
                Some(old) => {
                    let d = c.wrapping_sub(old);
                    d.min(d.wrapping_neg()) as usize
                }
                None => 3 + c.min(c.wrapping_neg()) as usize,
            };
            adj + g.e.dist(g.e.pos, g.l.p[i])
        };
        let i = (0..self.l.p.len()).min_by_key(|&i| cost(self, i)).unwrap();
        self.set_p(i, c);
        self.e.goto(self.l.p[i]);
        self.e.txt(".");
    }

    fn restore_p(&mut self) {
        for i in 0..self.l.p.len() {
            self.set_p(i, self.l.pc[i]);
        }
    }

    fn canonical_p(&self) -> Vec<Option<u8>> {
        self.l.pc.iter().map(|&c| Some(c)).collect()
    }

    fn p_want(&self, c: usize) -> PState {
        self.p_in
            .get(&c)
            .cloned()
            .unwrap_or_else(|| vec![None; self.l.p.len()])
    }

    fn p_handoff(&mut self, c: usize) {
        self.p_seen.entry(c).or_default().push(self.p_val.clone());
        for (i, w) in self.p_want(c).into_iter().enumerate() {
            if let Some(v) = w {
                self.set_p(i, v);
            }
        }
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
            LOp::Load(a) if self.alloc.alias.contains_key(&i.dst.unwrap()) => {
                self.addr(a, i.span);
            }
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
                self.e.txt(",");
                self.e.mark(dc);
            }
            LOp::PutChar(V(v)) => {
                let cv = self.cell(*v);
                self.e.goto(cv);
                self.e.txt(".");
            }
            LOp::PutChar(I(c)) => self.put_const(*c),
            LOp::And(..) | LOp::Or(..) | LOp::Xor(..) => self.err(i.span, "unsupported operation", "bitwise and/or/xor of runtime values is not yet supported"),
            LOp::AddrOf(_) | LOp::PtrAdd(..) | LOp::PtrAddImm(..) | LOp::LoadDyn(..) | LOp::StoreDyn(..) => self.err(i.span, "unsupported operation", "pointers must be a global/stack slot plus constants and at most one runtime i8 index"),
        }
    }

    fn term(&mut self, t: Term, c_dies: bool) {
        if self.dispatch && t != Term::Halt {
            self.restore_p();
        }
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

    fn insts(&mut self, f: &LFunc, b: BlockId) {
        let alloc = self.alloc;
        for (idx, i) in f.blocks[b].insts.iter().enumerate() {
            let dies = &alloc.dies[b][idx];
            let mut left = HashMap::new();
            for v in i.op.uses() {
                if dies.contains(&v) {
                    *left.entry(v).or_insert(0) += 1;
                }
            }
            self.inst(i, &mut Uses(left));
        }
    }

    fn block(&mut self, f: &LFunc, b: BlockId, zero: HashSet<usize>) {
        self.p_val = self.canonical_p();
        self.e.zero = zero;
        self.insts(f, b);
        self.term(f.blocks[b].term, self.alloc.br_dies[b]);
    }

    fn enter_ctl(&mut self, c: usize) {
        self.e.zero = self.l.inv_zero.clone();
        self.p_val = self.p_want(c);
        self.p_entry = vec![Some(c); self.l.p.len()];
    }

    fn leave_ctl(&mut self, c: usize) {
        self.e.zero = self.l.rest_zero.clone();
        self.e.zero.insert(c);
        self.p_val = self.canonical_p();
        self.p_entry = vec![None; self.l.p.len()];
    }

    fn cond(&mut self, x: usize, dec: bool, body: impl FnOnce(&mut Self)) {
        let pre = self.e.zero.clone();
        let p_pre = self.p_val.clone();
        let outer = std::mem::take(&mut self.e.touched);
        self.e.goto(x);
        self.e.txt("[");
        self.e.zero.remove(&x);
        if dec {
            self.e.bump(x, -1);
            self.e.zero.insert(x);
        } else {
            self.e.clear(x);
        }
        body(self);
        self.e.goto(x);
        self.e.txt("]");
        let wrote = std::mem::replace(&mut self.e.touched, outer);
        self.e.zero.retain(|c| pre.contains(c));
        self.e.zero.insert(x);
        self.e.touched.extend(wrote);
        for (v, pre) in self.p_val.iter_mut().zip(p_pre) {
            if *v != pre {
                *v = None;
            }
        }
    }

    fn tree(&mut self, f: &LFunc, x: BlockId) {
        if !self.shape.unwrap().header[x] {
            return self.within(f, x);
        }
        let lc = self.l.lcell[&x];
        self.p_handoff(lc);
        self.e.bump(lc, 1);
        self.e.goto(lc);
        self.e.txt("[");
        self.e.bump(lc, -1);
        self.enter_ctl(lc);
        self.within(f, x);
        self.e.goto(lc);
        self.e.txt("]");
        self.leave_ctl(lc);
    }

    fn within(&mut self, f: &LFunc, x: BlockId) {
        self.insts(f, x);
        match f.blocks[x].term {
            Term::Jmp(t) => self.edge(f, x, t),
            Term::Br(c, t, e) => self.br(f, x, c, t, e),
            Term::Halt => {}
        }
        for &y in &self.shape.unwrap().kids[x] {
            let fl = self.l.flag[&y];
            self.e.goto(fl);
            self.e.txt("[");
            self.e.bump(fl, -1);
            self.enter_ctl(fl);
            self.tree(f, y);
            self.e.goto(fl);
            self.e.txt("]");
            self.leave_ctl(fl);
        }
    }

    fn edge_flag(&self, from: BlockId, to: BlockId) -> Option<usize> {
        let sh = self.shape.unwrap();
        if sh.back(from, to) {
            Some(self.l.lcell[&to])
        } else if sh.merge[to] {
            Some(self.l.flag[&to])
        } else {
            None
        }
    }

    fn edge(&mut self, f: &LFunc, from: BlockId, to: BlockId) {
        match self.edge_flag(from, to) {
            Some(fl) => {
                self.p_handoff(fl);
                self.e.bump(fl, 1);
            }
            None => self.tree(f, to),
        }
    }

    fn br(&mut self, f: &LFunc, b: BlockId, c: Value, t: BlockId, e: BlockId) {
        let (xc, ec) = self.l.xcell[&b];
        let cc = self.cell(c);
        let then_reuses = self.edge_flag(b, t).is_none()
            && self.shape.unwrap().dominated[t].iter().any(|&d| {
                f.blocks[d]
                    .insts
                    .iter()
                    .flat_map(defs)
                    .any(|v| self.cell(v) == cc)
            });
        let x = if self.alloc.br_dies[b] && !then_reuses {
            cc
        } else {
            if self.alloc.br_dies[b] {
                self.e.mv(cc, &[(xc, 1)]);
            } else {
                self.copy(cc, xc);
            }
            xc
        };
        match self.edge_flag(b, e) {
            Some(fe) => {
                self.p_handoff(fe);
                self.e.bump(fe, 1);
                self.cond(x, false, |g| {
                    g.e.bump(fe, -1);
                    g.edge(f, b, t);
                });
            }
            None => {
                self.e.bump(ec, 1);
                let (zero, p_val, p_entry) = (
                    self.e.zero.clone(),
                    self.p_val.clone(),
                    self.p_entry.clone(),
                );
                self.cond(x, false, |g| {
                    g.e.bump(ec, -1);
                    g.edge(f, b, t);
                });
                self.cond(ec, true, |g| {
                    g.e.zero.extend(zero.into_iter().filter(|&c| c != ec));
                    g.e.zero.insert(x);
                    g.p_val = p_val;
                    g.p_entry = p_entry;
                    g.edge(f, b, e)
                });
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn generate(
    f: &LFunc,
    l: &Layout,
    alloc: &Alloc,
    shape: Option<&Shape>,
    p_in: &PIn,
    phys: &[usize],
    init: &[u8],
    entry_outside: bool,
    loop_blocks: &[BlockId],
) -> (Vec<Tok>, Vec<Err>, PStats) {
    let mut g = Gen {
        e: Em {
            toks: Vec::new(),
            pos: 0,
            k: l.k,
            phys,
            zero: HashSet::new(),
            touched: HashSet::new(),
        },
        l,
        alloc,
        shape,
        errs: Vec::new(),
        p_val: l.pc.iter().map(|&c| Some(c)).collect(),
        p_in,
        p_seen: HashMap::new(),
        p_entry: vec![None; l.p.len()],
        p_need: HashMap::new(),
        dispatch: !loop_blocks.is_empty(),
    };
    g.e.pos = (0..l.total).find(|&c| phys[c] == 0).unwrap_or(0);
    g.e.toks.push(Tok::At(g.e.pos));

    if shape.is_some() {
        g.e.zero = (0..l.total).filter(|&c| init[c] == 0).collect();
        g.tree(f, 0);
        return (g.e.toks, g.errs, (g.p_seen, g.p_need));
    }

    if entry_outside {
        let zero = (0..l.total).filter(|&c| init[c] == 0).collect();
        g.block(f, 0, zero);
    }
    if g.dispatch {
        g.e.goto(l.run);
        g.e.txt("[");
        for &b in loop_blocks {
            let fl = l.flag[&b];
            g.e.goto(fl);
            g.e.txt("[");
            g.e.bump(fl, -1);
            g.block(f, b, l.inv_zero.clone());
            g.e.goto(fl);
            g.e.txt("]");
        }
        g.e.goto(l.run);
        g.e.txt("]");
    }
    (g.e.toks, g.errs, (g.p_seen, g.p_need))
}

fn render(toks: &[Tok], phys: &[usize]) -> String {
    let mut out: Vec<u8> = Vec::new();
    let mut pos = 0;
    let put = |c: u8, out: &mut Vec<u8>| {
        let inv = match c {
            b'<' => b'>',
            b'>' => b'<',
            b'+' => b'-',
            b'-' => b'+',
            _ => 0,
        };
        if inv != 0 && out.last() == Some(&inv) {
            out.pop();
        } else {
            out.push(c);
        }
    };
    for t in toks {
        match *t {
            Tok::Go(c) => {
                let p = phys[c];
                let ch = if p > pos { b'>' } else { b'<' };
                for _ in 0..p.abs_diff(pos) {
                    put(ch, &mut out);
                }
                pos = p;
            }
            Tok::Txt(s) => s.bytes().for_each(|b| put(b, &mut out)),
            Tok::At(c) => pos = phys[c],
        }
    }
    String::from_utf8(out).unwrap()
}

struct Travel {
    unit_of: Vec<usize>,
    off_of: Vec<usize>,
    lens: Vec<usize>,
    adj: Vec<Vec<(usize, i64, i64, u64)>>,
}

impl Travel {
    fn new(toks: &[Tok], l: &Layout) -> Self {
        let mut unit_of = vec![0; l.total];
        let mut off_of = vec![0; l.total];
        for (u, &(s, n)) in l.units.iter().enumerate() {
            for i in 0..n {
                unit_of[s + i] = u;
                off_of[s + i] = i;
            }
        }
        let mut w: HashMap<(usize, usize), u64> = HashMap::new();
        let mut cur = usize::MAX;
        for t in toks {
            match *t {
                Tok::Go(c) => {
                    *w.entry((cur, c)).or_default() += 1;
                    cur = c;
                }
                Tok::At(c) if cur != usize::MAX || !w.is_empty() => cur = c,
                _ => {}
            }
        }
        let mut adj = vec![Vec::new(); l.units.len()];
        for ((a, b), n) in w {
            let (ub, ob) = (unit_of[b], off_of[b] as i64);
            if a == usize::MAX {
                adj[ub].push((usize::MAX, ob, 0, n));
                continue;
            }
            let (ua, oa) = (unit_of[a], off_of[a] as i64);
            if ua != ub {
                adj[ua].push((ub, oa, ob, n));
                adj[ub].push((ua, ob, oa, n));
            }
        }
        Travel {
            unit_of,
            off_of,
            lens: l.units.iter().map(|u| u.1).collect(),
            adj,
        }
    }

    fn starts(&self, order: &[usize]) -> Vec<i64> {
        let mut s = vec![0; order.len()];
        let mut at = 0;
        for &u in order {
            s[u] = at;
            at += self.lens[u] as i64;
        }
        s
    }

    fn cost(&self, order: &[usize]) -> u64 {
        let s = self.starts(order);
        let mut c = 0;
        for (u, adj) in self.adj.iter().enumerate() {
            for &(v, ou, ov, n) in adj {
                let other = if v == usize::MAX { 0 } else { s[v] + ov };
                if v == usize::MAX || u < v {
                    c += n * (s[u] + ou).abs_diff(other);
                }
            }
        }
        c
    }

    fn local(&self, s: &[i64], a: usize, b: usize) -> i64 {
        let mut c = 0;
        for &x in &[a, b] {
            for &(v, ou, ov, n) in &self.adj[x] {
                if x == b && v == a {
                    continue;
                }
                let other = if v == usize::MAX { 0 } else { s[v] + ov };
                c += (n * (s[x] + ou).abs_diff(other)) as i64;
            }
        }
        c
    }

    fn swap(&self, order: &mut [usize], s: &mut [i64], j: usize) -> i64 {
        let (a, b) = (order[j], order[j + 1]);
        let before = self.local(s, a, b);
        s[b] = s[a];
        s[a] = s[b] + self.lens[b] as i64;
        order.swap(j, j + 1);
        self.local(s, a, b) - before
    }

    fn local_search(&self, order: &mut [usize]) -> u64 {
        let n = order.len();
        let mut s = self.starts(order);
        loop {
            let mut improved = false;
            for u in 0..n {
                let i = order.iter().position(|&x| x == u).unwrap();
                let (mut best, mut at) = (0i64, i);
                let mut d = 0;
                for j in (0..i).rev() {
                    d += self.swap(order, &mut s, j);
                    if d < best {
                        (best, at) = (d, j);
                    }
                }
                for j in 0..i {
                    d += self.swap(order, &mut s, j);
                }
                for j in i..n - 1 {
                    d += self.swap(order, &mut s, j);
                    if d < best {
                        (best, at) = (d, j + 1);
                    }
                }
                for j in (i..n - 1).rev() {
                    self.swap(order, &mut s, j);
                }
                if at < i {
                    for j in (at..i).rev() {
                        self.swap(order, &mut s, j);
                    }
                } else {
                    for j in i..at {
                        self.swap(order, &mut s, j);
                    }
                }
                improved |= best < 0;
            }
            if !improved {
                return self.cost(order);
            }
        }
    }

    fn optimize(&self, phys: &[usize]) -> Vec<usize> {
        let n = self.lens.len();
        let mut first = vec![usize::MAX; n];
        for (c, &u) in self.unit_of.iter().enumerate() {
            first[u] = first[u].min(phys[c]);
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&u| first[u]);

        let mut best = self.local_search(&mut order);
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        let mut next = |m: usize| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % m as u64) as usize
        };
        for _ in 0..40 {
            if n < 3 {
                break;
            }
            let mut cand = order.clone();
            for _ in 0..3 {
                let u = cand.remove(next(n));
                cand.insert(next(n), u);
            }
            let c = self.local_search(&mut cand);
            if c < best {
                best = c;
                order = cand;
            }
        }

        let s = self.starts(&order);
        (0..phys.len())
            .map(|c| s[self.unit_of[c]] as usize + self.off_of[c])
            .collect()
    }
}

fn learn_p((seen, need): &PStats, n: usize) -> PIn {
    let adj = |a: Option<u8>, b: u8| -> usize {
        let d = match a {
            Some(a) => b.wrapping_sub(a),
            None => b,
        };
        let d = d.min(d.wrapping_neg()) as usize;
        let d = d.min(8 + 2 * d.isqrt());
        if a.is_none() { d + 3 } else { d }
    };
    seen.iter()
        .map(|(&c, states)| {
            let want = (0..n)
                .map(|i| {
                    let needs = need.get(&(c, i)).map_or(&[][..], |v| &v[..]);
                    let mut cands: Vec<Option<u8>> = vec![None];
                    cands.extend(
                        states
                            .iter()
                            .map(|s| s[i])
                            .chain(needs.iter().map(|&v| Some(v))),
                    );
                    cands
                        .into_iter()
                        .map(|cand| {
                            let into: usize = match cand {
                                None => 0,
                                Some(v) => states.iter().map(|s| adj(s[i], v)).sum(),
                            };
                            let out: usize = needs.iter().map(|&v| adj(cand, v)).sum();
                            (into + out, cand)
                        })
                        .min()
                        .unwrap()
                        .1
                })
                .collect();
            (c, want)
        })
        .collect()
}

fn print_cells(f: &LFunc, k: usize) -> Vec<u8> {
    let mut count: HashMap<u8, usize> = HashMap::new();
    for i in f.blocks.iter().flat_map(|b| &b.insts) {
        if let LOp::PutChar(Operand::I(c)) = i.op {
            *count.entry(c).or_default() += 1;
        }
    }
    let mut vals: Vec<u8> = count.keys().copied().collect();
    vals.sort_unstable();
    let k = k.min(vals.len());
    let dist = |a: u8, b: u8| a.wrapping_sub(b).min(b.wrapping_sub(a)) as usize;
    let cost = |set: &[u8]| -> usize {
        count
            .iter()
            .map(|(&c, &n)| n * set.iter().map(|&s| dist(c, s)).min().unwrap())
            .sum()
    };
    let mut best: Option<(usize, Vec<u8>)> = None;
    let mut pick = vec![0usize; k];
    fn rec(
        i: usize,
        from: usize,
        pick: &mut Vec<usize>,
        vals: &[u8],
        cost: &dyn Fn(&[u8]) -> usize,
        best: &mut Option<(usize, Vec<u8>)>,
    ) {
        if i == pick.len() {
            let set: Vec<u8> = pick.iter().map(|&j| vals[j]).collect();
            let c = cost(&set);
            if best.as_ref().is_none_or(|(b, _)| c < *b) {
                *best = Some((c, set));
            }
            return;
        }
        for j in from..vals.len() {
            pick[i] = j;
            rec(i + 1, j + 1, pick, vals, cost, best);
        }
    }
    rec(0, 0, &mut pick, &vals, &cost, &mut best);
    best.map(|(_, s)| s).unwrap_or_default()
}

pub fn emit(f: &LFunc, globals: &[IRGlobal], ctx: &mut DiagCtx) -> Option<INPAssembly> {
    if f.blocks.is_empty() {
        return Some(INPAssembly {
            program: String::new(),
            data: Vec::new(),
        });
    }
    let alloc = alloc(f);

    let shape = shape(f);
    let entry_outside = !f.blocks.iter().any(|b| b.term.succs().contains(&0));
    let loop_blocks: Vec<BlockId> = match shape {
        Some(_) => Vec::new(),
        None => (0..f.blocks.len())
            .filter(|&b| b != 0 || !entry_outside)
            .collect(),
    };
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
    let mut units = Vec::new();
    let mut take = |n: usize, together: bool| {
        if together {
            units.push((next, n));
        } else {
            units.extend((next..next + n).map(|c| (c, 1)));
        }
        next += n;
        next - n
    };
    let run = if dispatch { take(1, false) } else { usize::MAX };
    let (flag, lcell, xcell): (HashMap<BlockId, usize>, HashMap<_, _>, HashMap<_, _>) = match &shape
    {
        None => (
            loop_blocks.iter().map(|&b| (b, take(1, false))).collect(),
            HashMap::new(),
            HashMap::new(),
        ),
        Some(sh) => {
            let n = f.blocks.len();
            let flag = (0..n)
                .filter(|&b| sh.merge[b])
                .map(|b| (b, take(1, false)))
                .collect();
            let lcell = (0..n)
                .filter(|&b| sh.header[b])
                .map(|b| (b, take(1, false)))
                .collect();
            let xcell = (0..n)
                .filter(|&b| matches!(f.blocks[b].term, Term::Br(..)))
                .map(|b| (b, (take(1, false), take(1, false))))
                .collect();
            (flag, lcell, xcell)
        }
    };
    let pc = print_cells(f, NP);
    let p: Vec<usize> = pc.iter().map(|_| take(1, false)).collect();
    let k = take(1, false);
    let t = take(NT, true);
    let vals = take(alloc.n, false);
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
        o.cell = take(o.cells(), o.strided);
        objs.insert(base, o);
    }
    let total = next;

    let mut init = vec![0u8; total];
    if dispatch {
        init[run] = 1;
        if !entry_outside {
            init[flag[&0]] = 1;
        }
    }
    for gl in globals {
        if let (GlobalInit::Bytes(b), Some(o)) =
            (&gl.init, objs.get(&Base::Global(gl.name.clone())))
        {
            for (j, &byte) in b.iter().enumerate() {
                init[o.elem(j)] = byte;
            }
        }
    }

    // T, K, walk trails and branch cells are zero between operations
    let mut rest_zero: HashSet<usize> = (t..t + NT).collect();
    rest_zero.insert(k);
    rest_zero.extend(xcell.values().flat_map(|&(x, e)| [x, e]));
    for o in objs.values().filter(|o| o.strided) {
        rest_zero.extend((0..=o.size).map(|j| o.trail(j)));
        rest_zero.insert(o.cell);
    }
    // only one flag is ever set, and the block it guards clears it on entry
    let mut inv_zero = rest_zero.clone();
    inv_zero.extend(flag.values().chain(lcell.values()));

    for (&c, &v) in p.iter().zip(&pc) {
        init[c] = v;
    }
    let l = Layout {
        run,
        flag,
        p,
        pc,
        k,
        t,
        vals,
        objs,
        total,
        units,
        lcell,
        xcell,
        inv_zero,
        rest_zero,
    };

    let n = l.units.len();
    let mut starts = vec![(0..n).collect::<Vec<_>>(), (0..n).rev().collect()];
    let mut big_first: Vec<usize> = (0..n).collect();
    big_first.sort_by_key(|&u| std::cmp::Reverse(l.units[u].1));
    starts.push(big_first);

    let mut best: Option<(String, Vec<usize>)> = None;
    let ctl = || l.flag.values().chain(l.lcell.values());
    let canonical: PIn = ctl()
        .map(|&c| (c, l.pc.iter().map(|&v| Some(v)).collect()))
        .collect();
    let none: PIn = ctl().map(|&c| (c, vec![None; l.p.len()])).collect();
    for (order, p_start) in starts
        .iter()
        .flat_map(|o| [(o.clone(), &canonical), (o.clone(), &none)])
    {
        let mut phys = vec![0; total];
        let mut at = 0;
        for u in order {
            let (s, len) = l.units[u];
            for i in 0..len {
                phys[s + i] = at + i;
            }
            at += len;
        }
        let mut p_in = p_start.clone();
        for _ in 0..LAYOUT_ROUNDS {
            let (toks, errs, p_seen) = generate(
                f,
                &l,
                &alloc,
                shape.as_ref(),
                &p_in,
                &phys,
                &init,
                entry_outside,
                &loop_blocks,
            );
            if !errs.is_empty() {
                for (span, title, msg) in errs {
                    ctx.emit(Diag::error(
                        title,
                        span.unwrap_or_else(|| (0, 0).into()),
                        msg,
                    ));
                }
                return None;
            }
            let prog = render(&toks, &phys);
            if best.as_ref().is_none_or(|(b, _)| prog.len() < b.len()) {
                best = Some((prog, phys.clone()));
            }
            phys = Travel::new(&toks, &l).optimize(&phys);
            p_in = learn_p(&p_seen, l.p.len());
        }
    }
    let (program, phys) = best.unwrap();

    let mut data = vec![0u8; total];
    for c in 0..total {
        data[phys[c]] = init[c];
    }

    if program.len() + 1 + total > MEM_SIZE {
        ctx.emit(Diag::warning(
            "program too large",
            (0, 0).into(),
            "code + `@` + data exceed the 8 KiB memory",
        ));
        return None;
    }
    Some(INPAssembly { program, data })
}
