use std::sync::Arc;

use miette::{Diagnostic, NamedSource, Result, SourceSpan};
use thiserror::Error;

use crate::ir::{
    ICmpKind, IRBlock, IRFunction, IRInst, IRInstFlags, IRInstMeta, IRModule, IROp, IRResult,
    Param, Value, Width,
};

#[derive(Debug, Error, Diagnostic)]
#[error("failed to parse `{name}`")]
#[diagnostic(code(ir::parse))]
pub struct ParseErrors {
    name: String,
    #[related]
    errors: Vec<ParseError>,
}

#[derive(Debug, Error, Diagnostic)]
#[error("{message}")]
pub struct ParseError {
    message: String,
    #[source_code]
    src: NamedSource<Arc<str>>,
    #[label("{label}")]
    span: SourceSpan,
    label: String,
    #[help]
    help: Option<String>,
}

struct PErr {
    message: String,
    span: SourceSpan,
    label: String,
    help: Option<String>,
}

impl PErr {
    fn new(message: impl Into<String>, span: SourceSpan, label: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span,
            label: label.into(),
            help: None,
        }
    }

    fn help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    fn into_diag(self, name: &str, src: Arc<str>) -> ParseError {
        ParseError {
            message: self.message,
            src: NamedSource::new(name, src),
            span: self.span,
            label: self.label,
            help: self.help,
        }
    }
}

type PResult<T> = Result<T, PErr>;

#[derive(Debug, Clone, PartialEq)]
#[rustfmt::skip]
enum Tok {
    Ident(String),
    Val(Value),
    Int(i64),
    LParen, RParen, LBrace, RBrace,
    Comma, Colon, Eq, Arrow,
    Newline, Eof,
}

impl Tok {
    fn describe(&self) -> String {
        match self {
            Tok::Ident(s) => format!("`{s}`"),
            Tok::Val(v) => format!("`%{v}`"),
            Tok::Int(i) => format!("`{i}`"),
            Tok::LParen => "`(`".into(),
            Tok::RParen => "`)`".into(),
            Tok::LBrace => "`{`".into(),
            Tok::RBrace => "`}`".into(),
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

fn lex(src: &str, errs: &mut Vec<PErr>) -> Vec<Token> {
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
            b'%' => {
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                match src[start + 1..i].parse::<Value>() {
                    Ok(val) => push(&mut out, Tok::Val(val), start, i),
                    Err(_) => errs.push(
                        PErr::new(
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
                    Err(_) => errs.push(PErr::new(
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
                errs.push(PErr::new(
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

    fn unexpected(&self, what: &str) -> PErr {
        let t = self.peek();
        PErr::new(
            format!("expected {what}"),
            t.span,
            format!("found {}", t.tok.describe()),
        )
    }

    fn expect(&mut self, tok: Tok, what: &str) -> PResult<SourceSpan> {
        if self.peek().tok == tok {
            Ok(self.bump().span)
        } else {
            Err(self.unexpected(what))
        }
    }

    fn ident(&mut self, what: &str) -> PResult<(String, SourceSpan)> {
        match self.peek().tok.clone() {
            Tok::Ident(s) => Ok((s, self.bump().span)),
            _ => Err(self.unexpected(what)),
        }
    }

    fn value(&mut self) -> PResult<Value> {
        match self.peek().tok {
            Tok::Val(val) => {
                self.bump();
                Ok(val)
            }
            _ => Err(self.unexpected("a value like `%0`")),
        }
    }

    fn int(&mut self, what: &str) -> PResult<(i64, SourceSpan)> {
        match self.peek().tok {
            Tok::Int(int) => Ok((int, self.bump().span)),
            _ => Err(self.unexpected(what)),
        }
    }

    fn uint(&mut self, what: &str) -> PResult<u32> {
        let (i, span) = self.int(what)?;
        u32::try_from(i).map_err(|_| {
            PErr::new(
                format!("invalid {what}"),
                span,
                "must be a non-negative 32-bit integer",
            )
        })
    }

    fn ty(&mut self) -> PResult<Width> {
        let (name, span) = self.ident("a type")?;
        if name == "ptr" {
            return Ok(Width::Ptr);
        }
        name.strip_prefix('i')
            .and_then(|n| n.parse::<u16>().ok())
            .filter(|&n| n > 0)
            .map(Width::Width)
            .ok_or_else(|| {
                PErr::new(format!("unknown type {name}"), span, "not a type")
                    .help("types are `ptr` or `iN`, e.g. `i32`")
            })
    }

    fn skip_newlines(&mut self) {
        while self.eat(Tok::Newline) {}
    }

    fn end_of_line(&mut self) -> PResult<()> {
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

    fn inst(&mut self) -> PResult<IRInst> {
        let start = self.peek().span.offset();

        let result = if let Tok::Val(_) = self.peek().tok {
            let value = self.value()?;
            self.expect(Tok::Colon, "`:` after the result value")?;
            let ty = self.ty()?;
            self.expect(Tok::Eq, "`=`")?;
            Some(IRResult {
                id: value,
                width: ty,
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
            _ => None,
        };

        let op = if let Some(make) = binop {
            let a = self.value()?;
            self.expect(Tok::Comma, "`,`")?;
            make(a, self.value()?)
        } else {
            match m.as_str() {
                "const" => IROp::Const(self.int("a constant")?.0),
                "jmp" => IROp::Jmp(self.ident("a label")?.0),
                "icmp" => {
                    let (k, kspan) = self.ident("a icmp kind")?;
                    let kind = k.parse::<ICmpKind>().map_err(|_| {
                        PErr::new(
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
                    return Err(PErr::new(
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

    fn function(&mut self, errs: &mut Vec<PErr>) -> PResult<IRFunction> {
        self.bump();
        let (name, name_span) = self.ident("a function name")?;
        self.expect(Tok::LParen, "`(`")?;
        let mut params = Vec::new();
        if !self.eat(Tok::RParen) {
            loop {
                let id = self.value()?;
                self.expect(Tok::Colon, "`:`")?;
                let ty = self.ty()?;
                params.push(Param { id, width: ty });
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
                    return Err(PErr::new(
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
                        errs.push(e);
                        self.recover_line();
                    }
                }
                _ => {
                    let res = self.inst().and_then(|i| self.end_of_line().map(|_| i));
                    match res {
                        Ok(inst) => match blocks.last_mut() {
                            Some(b) => b.instructions.push(inst),
                            None => {
                                errs.push(
                                    PErr::new(
                                        "instruction outside of a block",
                                        inst.meta.span.unwrap(),
                                        "no label before this",
                                    )
                                    .help("start the function body with a label, e.g. `entry:`"),
                                );
                            }
                        },
                        Err(e) => {
                            errs.push(e);
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
        })
    }

    fn module(&mut self, errs: &mut Vec<PErr>) -> IRModule {
        let mut functions = Vec::new();
        loop {
            self.skip_newlines();
            match &self.peek().tok {
                Tok::Eof => break,
                Tok::Ident(k) if k == "fn" => match self.function(errs) {
                    Ok(func) => functions.push(func),
                    Err(e) => {
                        errs.push(e);
                        self.recover_block();
                    }
                },
                _ => {
                    let t = self.bump();
                    errs.push(
                        PErr::new("expected fn", t.span, "not a function definition")
                            .help("only function definitions are allowed at the top level"),
                    );
                    self.recover_block();
                }
            }
        }

        IRModule { functions }
    }
}

fn check_result_shape(
    op: &IROp,
    result: &Option<IRResult>,
    m: &str,
    span: SourceSpan,
) -> PResult<()> {
    let produces = match op {
        IROp::Jmp(_) | IROp::Br(..) | IROp::Ret(_) | IROp::Store(..) => Some(false),
        IROp::Call(..) => None,
        _ => Some(true),
    };
    match (produces, result.is_some()) {
        (Some(true), false) => {
            Err(
                PErr::new(format!("`{m}` produces a value"), span, "result is missing")
                    .help(format!("write it as `%N: <type> = {m} ...`")),
            )
        }
        (Some(false), true) => Err(PErr::new(
            format!("`{m}` does not produce a value"),
            span,
            "cannot be assigned",
        )
        .help("remove the `%N: <type> = ` prefix")),
        _ => Ok(()),
    }
}

pub fn parse_module(name: &str, src: &str) -> Result<IRModule, ParseErrors> {
    let mut errs = Vec::new();
    let toks = lex(src, &mut errs);
    let mut p = Parser {
        toks: &toks,
        pos: 0,
        last_end: 0,
    };
    let module = p.module(&mut errs);

    if errs.is_empty() {
        return Ok(module);
    }

    let source: Arc<str> = Arc::from(src);
    Err(ParseErrors {
        name: name.to_string(),
        errors: errs
            .into_iter()
            .map(|e| e.into_diag(name, source.clone()))
            .collect(),
    })
}
