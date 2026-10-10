use miette::{Result, SourceSpan};

use crate::{
    diagnostics::{Diag, DiagCtx},
    ir::{
        GlobalInit, ICmpKind, IRBlock, IRExtern, IRFunction, IRGlobal, IRInst, IRInstFlags,
        IRInstMeta, IRModule, IROp, IRResult, Param, Value, Width,
    },
};

#[derive(Debug, Clone, PartialEq)]
#[rustfmt::skip]
enum Tok {
    Ident(String),
    Val(Value),
    Int(i64),
    Global(String),
    Str(Vec<u8>),
    LParen, RParen, LBrace, RBrace,
    LBracket, RBracket,
    Comma, Colon, Eq, Arrow,
    Newline, Eof,
}

impl Tok {
    fn describe(&self) -> String {
        match self {
            Tok::Ident(s) => format!("`{s}`"),
            Tok::Val(v) => format!("`%{v}`"),
            Tok::Int(i) => format!("`{i}`"),
            Tok::Global(g) => format!("`@{g}`"),
            Tok::Str(_) => "string literal".into(),
            Tok::LParen => "`(`".into(),
            Tok::RParen => "`)`".into(),
            Tok::LBrace => "`{`".into(),
            Tok::RBrace => "`}`".into(),
            Tok::LBracket => "`[`".into(),
            Tok::RBracket => "`]`".into(),
            Tok::Comma => "`,`".into(),
            Tok::Colon => "`:`".into(),
            Tok::Eq => "`=`".into(),
            Tok::Arrow => "`->`".into(),
            Tok::Newline => "`end of line`".into(),
            Tok::Eof => "`end of file`".into(),
        }
    }
}

#[derive(Debug, Clone)]
struct Token {
    tok: Tok,
    span: SourceSpan,
}

fn lex(src: &str, errs: &mut DiagCtx) -> Vec<Token> {
    let mut out: Vec<Token> = Vec::new();
    let b = src.as_bytes();
    let push = |out: &mut Vec<Token>, tok: Tok, start: usize, end: usize| {
        out.push(Token {
            tok,
            span: (start, end - start).into(),
        });
    };

    let mut i = 0;
    while i < b.len() {
        let start = i;
        match b[i] {
            b'\n' => {
                if !matches!(
                    out.last(),
                    Some(Token {
                        tok: Tok::Newline,
                        ..
                    })
                ) {
                    push(&mut out, Tok::Newline, start, i + 1);
                }
                i += 1;
            }
            b' ' | b'\t' | b'\r' => i += 1,
            b';' => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'(' => {
                push(&mut out, Tok::LParen, start, i + 1);
                i += 1;
            }
            b')' => {
                push(&mut out, Tok::RParen, start, i + 1);
                i += 1;
            }
            b'{' => {
                push(&mut out, Tok::LBrace, start, i + 1);
                i += 1;
            }
            b'}' => {
                push(&mut out, Tok::RBrace, start, i + 1);
                i += 1;
            }
            b'[' => {
                push(&mut out, Tok::LBracket, start, i + 1);
                i += 1;
            }
            b']' => {
                push(&mut out, Tok::RBracket, start, i + 1);
                i += 1;
            }
            b',' => {
                push(&mut out, Tok::Comma, start, i + 1);
                i += 1;
            }
            b':' => {
                push(&mut out, Tok::Colon, start, i + 1);
                i += 1;
            }
            b'=' => {
                push(&mut out, Tok::Eq, start, i + 1);
                i += 1;
            }
            b'-' if b.get(i + 1) == Some(&b'>') => {
                push(&mut out, Tok::Arrow, start, i + 2);
                i += 2;
            }
            b'@' => {
                i += 1;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.')
                {
                    i += 1;
                }
                if i == start + 1 {
                    errs.emit(Diag::error(
                        "invalid global name",
                        (start, 1).into(),
                        "expected name after `@`",
                    ));
                } else {
                    push(
                        &mut out,
                        Tok::Global(src[start + 1..i].to_string()),
                        start,
                        i,
                    );
                }
            }
            b'"' => {
                i += 1;
                let mut bytes = Vec::new();
                let mut ok = true;
                loop {
                    match b.get(i) {
                        None | Some(b'\n') => {
                            errs.emit(
                                Diag::error(
                                    "unterminated string literal",
                                    (start, i - start).into(),
                                    "string starts here",
                                )
                                .help("close it with `\"` on the same line"),
                            );
                            ok = false;
                            break;
                        }
                        Some(b'"') => {
                            i += 1;
                            break;
                        }
                        Some(b'\\') => {
                            let esc_start = i;
                            i += 1;
                            let byte = match b.get(i) {
                                Some(b'n') => Some(b'\n'),
                                Some(b't') => Some(b'\t'),
                                Some(b'r') => Some(b'\r'),
                                Some(b'0') => Some(0),
                                Some(b'\\') => Some(b'\\'),
                                Some(b'"') => Some(b'"'),
                                Some(b'x') => {
                                    let hex = src
                                        .get(i + 1..i + 3)
                                        .and_then(|h| u8::from_str_radix(h, 16).ok());
                                    if hex.is_some() {
                                        i += 2;
                                    }
                                    hex
                                }
                                _ => None,
                            };
                            match byte {
                                Some(b) => {
                                    bytes.push(b);
                                    i += 1;
                                }
                                None => {
                                    errs.emit(
                                        Diag::error(
                                            "unknown escape",
                                            (esc_start, 2).into(),
                                            "unsupported escape sequence",
                                        )
                                        .help("supported: \\n, \\t, \\r, \\0, \\\\, \\\", \\xXX"),
                                    );
                                    ok = false;
                                    i += 1;
                                }
                            }
                        }
                        Some(_) => {
                            let ch = src[i..].chars().next().unwrap();
                            bytes.extend_from_slice(ch.to_string().as_bytes());
                            i += ch.len_utf8();
                        }
                    }
                }
                if ok {
                    push(&mut out, Tok::Str(bytes), start, i);
                }
            }
            b'%' => {
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                match src[start + 1..i].parse::<Value>() {
                    Ok(val) => push(&mut out, Tok::Val(val), start, i),
                    Err(_) => errs.emit(
                        Diag::error(
                            "invalid value",
                            (start, (i - start).max(1)).into(),
                            "expected `%` followed by a number",
                        )
                        .help("values are written as `%0`, `%1`, ..."),
                    ),
                };
            }
            b'-' | b'0'..=b'9' => {
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                match src[start..i].parse::<i64>() {
                    Ok(val) => push(&mut out, Tok::Int(val), start, i),
                    Err(_) => errs.emit(Diag::error(
                        "not a valid integer",
                        (start, (i - start).max(1)).into(),
                        "value is not a valid 64-bit integer",
                    )),
                }
            }
            c if c.is_ascii_alphabetic() || c == b'_' || c == b'.' => {
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.')
                {
                    i += 1;
                }
                push(&mut out, Tok::Ident(src[start..i].to_string()), start, i);
            }
            _ => {
                let ch = src[i..].chars().next().unwrap();
                errs.emit(Diag::error(
                    format!("unexpected character `{ch}`"),
                    (i, ch.len_utf8()).into(),
                    "not valid here",
                ));
                i += ch.len_utf8();
            }
        }
    }

    push(&mut out, Tok::Newline, b.len(), b.len());
    push(&mut out, Tok::Eof, b.len(), b.len());

    out
}

struct Parser<'t> {
    toks: &'t [Token],
    pos: usize,
    last_end: usize,
}

impl<'t> Parser<'t> {
    fn peek(&self) -> &Token {
        &self.toks[self.pos]
    }

    fn peek_at(&self, n: usize) -> &Token {
        &self.toks[(self.pos + n).min(self.toks.len() - 1)]
    }

    fn bump(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        self.last_end = t.span.offset() + t.span.len();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, tok: Tok) -> bool {
        if self.peek().tok == tok {
            self.bump();
            true
        } else {
            false
        }
    }

    fn kw(&self, s: &str) -> bool {
        matches!(&self.peek().tok, Tok::Ident(k) if s == k)
    }

    fn unexpected(&self, what: &str) -> Diag {
        let t = self.peek();
        Diag::error(
            format!("expected {what}"),
            t.span,
            format!("found {}", t.tok.describe()),
        )
    }

    fn expect(&mut self, tok: Tok, what: &str) -> Result<SourceSpan, Diag> {
        if self.peek().tok == tok {
            Ok(self.bump().span)
        } else {
            Err(self.unexpected(what))
        }
    }

    fn ident(&mut self, what: &str) -> Result<(String, SourceSpan), Diag> {
        match self.peek().tok.clone() {
            Tok::Ident(s) => Ok((s, self.bump().span)),
            _ => Err(self.unexpected(what)),
        }
    }

    fn value(&mut self) -> Result<Value, Diag> {
        match self.peek().tok {
            Tok::Val(val) => {
                self.bump();
                Ok(val)
            }
            _ => Err(self.unexpected("a value like `%0`")),
        }
    }

    fn int(&mut self, what: &str) -> Result<(i64, SourceSpan), Diag> {
        match self.peek().tok {
            Tok::Int(int) => Ok((int, self.bump().span)),
            _ => Err(self.unexpected(what)),
        }
    }

    fn uint(&mut self, what: &str) -> Result<u32, Diag> {
        let (i, span) = self.int(what)?;
        u32::try_from(i).map_err(|_| {
            Diag::error(
                format!("invalid {what}"),
                span,
                "must be a non-negative 32-bit integer",
            )
        })
    }

    fn ty(&mut self) -> Result<Width, Diag> {
        let (name, span) = self.ident("a type")?;
        if name == "ptr" {
            return Ok(Width::Ptr);
        }
        name.strip_prefix('i')
            .and_then(|n| n.parse::<u16>().ok())
            .filter(|&n| n > 0)
            .map(Width::Width)
            .ok_or_else(|| {
                Diag::error(format!("unknown type {name}"), span, "not a type")
                    .help("types are `ptr` or `iN`, e.g. `i32`")
            })
    }

    fn skip_newlines(&mut self) {
        while self.eat(Tok::Newline) {}
    }

    fn end_of_line(&mut self) -> Result<(), Diag> {
        match self.peek().tok {
            Tok::Newline => {
                self.bump();
                Ok(())
            }
            Tok::RBrace | Tok::Eof => Ok(()),
            _ => Err(self.unexpected("end of line")),
        }
    }

    fn recover_line(&mut self) {
        while !matches!(self.peek().tok, Tok::Newline | Tok::RBrace | Tok::Eof) {
            self.bump();
        }
        self.eat(Tok::Newline);
    }

    fn recover_block(&mut self) {
        while !matches!(self.peek().tok, Tok::RBrace | Tok::Eof) {
            self.bump();
        }
        self.eat(Tok::RBrace);
    }

    fn inst(&mut self) -> Result<IRInst, Diag> {
        let start = self.peek().span.offset();

        let result = if let Tok::Val(_) = self.peek().tok {
            let value = self.value()?;
            self.expect(Tok::Colon, "`:` after the result value")?;
            let ty_span = self.peek().span;
            let ty = self.ty()?;
            self.expect(Tok::Eq, "`=`")?;
            Some(IRResult {
                id: value,
                width: ty,
                span: Some(ty_span),
            })
        } else {
            None
        };

        let (m, mspan) = self.ident("an instruction")?;

        let binop: Option<fn(Value, Value) -> IROp> = match m.as_str() {
            "add" => Some(IROp::Add),
            "sub" => Some(IROp::Sub),
            "mul" => Some(IROp::Mul),
            "udiv" => Some(IROp::UDiv),
            "sdiv" => Some(IROp::SDiv),
            "urem" => Some(IROp::URem),
            "srem" => Some(IROp::SRem),
            "and" => Some(IROp::And),
            "or" => Some(IROp::Or),
            "xor" => Some(IROp::Xor),
            "shl" => Some(IROp::Shl),
            "ashr" => Some(IROp::AShr),
            "lshr" => Some(IROp::LShr),
            _ => None,
        };

        let unop: Option<fn(Value) -> IROp> = match m.as_str() {
            "zext" => Some(IROp::ZExt),
            "sext" => Some(IROp::SExt),
            "trunc" => Some(IROp::Trunc),
            _ => None,
        };

        let op = if let Some(make) = binop {
            let a = self.value()?;
            self.expect(Tok::Comma, "`,`")?;
            make(a, self.value()?)
        } else if let Some(make) = unop {
            make(self.value()?)
        } else {
            match m.as_str() {
                "const" => IROp::Const(self.int("a constant")?.0),
                "jmp" => IROp::Jmp(self.ident("a label")?.0),
                "icmp" => {
                    let (k, kspan) = self.ident("a icmp kind")?;
                    let kind = k.parse::<ICmpKind>().map_err(|_| {
                        Diag::error(
                            format!("unknown comparison kind `{k}`"),
                            kspan,
                            "not a comparison kind",
                        )
                        .help("expected one of: eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge")
                    })?;
                    let a = self.value()?;
                    self.expect(Tok::Comma, "`,`")?;
                    IROp::ICmp(kind, a, self.value()?)
                }
                "br" => {
                    let c = self.value()?;
                    self.expect(Tok::Comma, "`,`")?;
                    let t = self.ident("a label")?.0;
                    self.expect(Tok::Comma, "`,`")?;
                    IROp::Br(c, t, self.ident("a label")?.0)
                }
                "call" => {
                    let callee = self.ident("a function name")?.0;
                    self.expect(Tok::LParen, "`(`")?;
                    let mut args = Vec::new();
                    if !self.eat(Tok::RParen) {
                        loop {
                            args.push(self.value()?);
                            if self.eat(Tok::RParen) {
                                break;
                            }
                            self.expect(Tok::Comma, "`,`")?;
                        }
                    }
                    IROp::Call(callee, args)
                }
                "ret" => IROp::Ret(match self.peek().tok {
                    Tok::RBrace | Tok::Newline | Tok::Eof => None,
                    _ => Some(self.value()?),
                }),
                "ptradd" => {
                    let p = self.value()?;
                    self.expect(Tok::Comma, "`,`")?;
                    IROp::PtrAdd(p, self.value()?)
                }
                "globaladdr" => match self.peek().tok.clone() {
                    Tok::Global(g) => {
                        self.bump();
                        IROp::GlobalAddr(g)
                    }
                    _ => return Err(self.unexpected("a global name starting with `@`")),
                },
                "alloca" => IROp::Alloca(self.uint("an allocation size")?),
                "load" => {
                    let p = self.value()?;
                    self.expect(Tok::Comma, "`,`")?;
                    IROp::Load(p, self.uint("an offset")?)
                }
                "store" => {
                    let p = self.value()?;
                    self.expect(Tok::Comma, "`,`")?;
                    let o = self.uint("an offset")?;
                    self.expect(Tok::Comma, "`,`")?;
                    IROp::Store(p, o, self.value()?)
                }
                _ => {
                    return Err(Diag::error(
                        format!("unknown instruction {m}"),
                        mspan,
                        "unknown mnemonic",
                    ));
                }
            }
        };

        check_result_shape(&op, &result, &m, mspan)?;

        let span = SourceSpan::from((start, (self.last_end - start)));
        Ok(IRInst {
            op,
            flags: IRInstFlags {},
            meta: IRInstMeta {
                result,
                span: Some(span),
            },
        })
    }

    fn function(&mut self, errs: &mut DiagCtx) -> Result<IRFunction, Diag> {
        self.bump();
        let (name, name_span) = self.ident("a function name")?;
        self.expect(Tok::LParen, "`(`")?;
        let mut params = Vec::new();
        if !self.eat(Tok::RParen) {
            loop {
                let id = self.value()?;
                self.expect(Tok::Colon, "`:`")?;
                let start = self.pos;
                let ty = self.ty()?;
                params.push(Param {
                    id,
                    width: ty,
                    span: Some((start, self.pos - start).into()),
                });
                if self.eat(Tok::RParen) {
                    break;
                }
                self.expect(Tok::Comma, "`,`")?;
            }
        }

        let ret = if self.eat(Tok::Arrow) {
            Some(self.ty()?)
        } else {
            None
        };
        self.expect(Tok::LBrace, "`{`")?;

        let mut blocks = Vec::new();
        loop {
            self.skip_newlines();
            match self.peek().tok.clone() {
                Tok::RBrace => {
                    self.bump();
                    break;
                }
                Tok::Eof => {
                    return Err(Diag::error(
                        "unterminated function",
                        name_span,
                        "this function is never closed",
                    )
                    .help("add a closing `}`"));
                }
                Tok::Ident(label) if self.peek_at(1).tok == Tok::Colon => {
                    self.bump();
                    self.bump();
                    blocks.push(IRBlock {
                        label,
                        instructions: vec![],
                    });
                    if let Err(e) = self.end_of_line() {
                        errs.emit(e);
                        self.recover_line();
                    }
                }
                _ => {
                    let res = self.inst().and_then(|i| self.end_of_line().map(|_| i));
                    match res {
                        Ok(inst) => match blocks.last_mut() {
                            Some(b) => b.instructions.push(inst),
                            None => {
                                errs.emit(
                                    Diag::error(
                                        "instruction outside of a block",
                                        inst.meta.span.unwrap(),
                                        "no label before this",
                                    )
                                    .help("start the function body with a label, e.g. `entry:`"),
                                );
                            }
                        },
                        Err(e) => {
                            errs.emit(e);
                            self.recover_line();
                        }
                    }
                }
            }
        }

        Ok(IRFunction {
            name,
            params,
            ret,
            blocks,
            span: Some(name_span),
        })
    }

    fn extern_decl(&mut self) -> Result<IRExtern, Diag> {
        self.bump();
        if !self.kw("fn") {
            return Err(self.unexpected("`fn`"));
        }
        self.bump();
        let (name, name_span) = self.ident("a function name")?;
        self.expect(Tok::LParen, "`(`")?;
        let (mut args, mut variadic) = (Vec::new(), false);
        if !self.eat(Tok::RParen) {
            loop {
                if self.kw("...") {
                    self.bump();
                    variadic = true;
                    self.expect(Tok::RParen, "`)`")?;
                    break;
                }
                args.push(self.ty()?);
                if self.eat(Tok::RParen) {
                    break;
                }
                self.expect(Tok::Comma, "`,` or `)`")?;
            }
        }

        let ret = if self.eat(Tok::Arrow) {
            Some(self.ty()?)
        } else {
            None
        };

        self.end_of_line()?;
        Ok(IRExtern {
            name,
            args,
            ret,
            variadic,
            span: Some(name_span),
        })
    }

    fn global_decl(&mut self) -> Result<IRGlobal, Diag> {
        self.bump();
        let (mut export, mut mutable) = (false, false);
        loop {
            if self.kw("export") {
                export = true;
                self.bump();
            } else if self.kw("mut") {
                mutable = true;
                self.bump();
            } else {
                break;
            }
        }

        let name_span = self.peek().span;
        let name = match self.peek().tok.clone() {
            Tok::Global(g) => g,
            _ => return Err(self.unexpected("a global name starting with `@`")),
        };
        self.bump();

        let align = if self.kw("align") {
            self.bump();
            self.uint("alignment")?
        } else {
            1
        };
        self.expect(Tok::Eq, "`=`")?;

        let init = match self.peek().tok.clone() {
            Tok::Str(s) => {
                self.bump();
                GlobalInit::Bytes(s)
            }
            Tok::LBracket => {
                self.bump();
                let mut bytes = Vec::new();
                if !self.eat(Tok::RBracket) {
                    loop {
                        let (n, span) = self.int("a byte value")?;
                        let byte = u8::try_from(n)
                            .ok()
                            .or_else(|| i8::try_from(n).ok().map(|v| v as u8))
                            .ok_or_else(|| {
                                Diag::error("byte out of range", span, "must be in -128..=255")
                            })?;
                        bytes.push(byte);
                        if self.eat(Tok::RBracket) {
                            self.bump();
                            break;
                        }
                        self.expect(Tok::Comma, "`,` or `]`")?;
                    }
                }
                GlobalInit::Bytes(bytes)
            }
            _ if self.kw("zero") => {
                self.bump();
                GlobalInit::Zeroed(self.uint("size")?)
            }
            _ => return Err(self.unexpected("a string, `[bytes]`, or `zero N`")),
        };

        self.end_of_line()?;
        Ok(IRGlobal {
            name,
            init,
            align,
            export,
            mutable,
            span: Some(name_span),
        })
    }

    fn module(&mut self, errs: &mut DiagCtx) -> IRModule {
        let mut functions = Vec::new();
        let mut externs = Vec::new();
        let mut globals = Vec::new();
        loop {
            self.skip_newlines();
            match &self.peek().tok {
                Tok::Eof => break,
                Tok::Ident(k) if k == "fn" => match self.function(errs) {
                    Ok(func) => functions.push(func),
                    Err(e) => {
                        errs.emit(e);
                        self.recover_block();
                    }
                },
                Tok::Ident(k) if k == "extern" => match self.extern_decl() {
                    Ok(ext) => externs.push(ext),
                    Err(e) => {
                        errs.emit(e);
                        self.recover_line();
                    }
                },
                Tok::Ident(k) if k == "global" => match self.global_decl() {
                    Ok(glob) => globals.push(glob),
                    Err(e) => {
                        errs.emit(e);
                        self.recover_line();
                    }
                },
                _ => {
                    let t = self.bump();
                    errs.emit(
                        Diag::error("expected fn", t.span, "not a function definition")
                            .help("only function definitions are allowed at the top level"),
                    );
                    self.recover_block();
                }
            }
        }

        IRModule {
            functions,
            externs,
            globals,
        }
    }
}

fn check_result_shape(
    op: &IROp,
    result: &Option<IRResult>,
    m: &str,
    span: SourceSpan,
) -> Result<(), Diag> {
    let produces = match op {
        IROp::Jmp(_) | IROp::Br(..) | IROp::Ret(_) | IROp::Store(..) => Some(false),
        IROp::Call(..) => None,
        _ => Some(true),
    };
    match (produces, result.is_some()) {
        (Some(true), false) => {
            Err(
                Diag::error(format!("`{m}` produces a value"), span, "result is missing")
                    .help(format!("write it as `%N: <type> = {m} ...`")),
            )
        }
        (Some(false), true) => Err(Diag::error(
            format!("`{m}` does not produce a value"),
            span,
            "cannot be assigned",
        )
        .help("remove the `%N: <type> = ` prefix")),
        _ => Ok(()),
    }
}

pub fn parse_module(src: &str, ctx: &mut DiagCtx) -> IRModule {
    let toks = lex(src, ctx);
    let mut p = Parser {
        toks: &toks,
        pos: 0,
        last_end: 0,
    };
    p.module(ctx)
}
