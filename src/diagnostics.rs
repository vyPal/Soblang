use std::sync::Arc;

use miette::{Diagnostic, NamedSource, Result, SourceSpan};
use thiserror::Error;

#[derive(Debug, Error, Diagnostic)]
#[error("compilation failed for `{name}`")]
pub struct DiagnosticReport {
    pub name: String,
    #[related]
    pub diagnostics: Vec<CompilerDiagnostic>,
}

#[derive(Debug, Error, Diagnostic)]
pub enum CompilerDiagnostic {
    #[error("{message}")]
    #[diagnostic(severity(Error))]
    Error {
        message: String,
        #[source_code]
        src: NamedSource<Arc<str>>,
        #[label("{label}")]
        span: SourceSpan,
        label: String,
        #[help]
        help: Option<String>,
    },
    #[error("{message}")]
    #[diagnostic(severity(Warning))]
    Warning {
        message: String,
        #[source_code]
        src: NamedSource<Arc<str>>,
        #[label("{label}")]
        span: SourceSpan,
        label: String,
        #[help]
        help: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Error,
    Warning,
}

pub struct Diag {
    pub level: Level,
    pub message: String,
    pub span: SourceSpan,
    pub label: String,
    pub help: Option<String>,
}

impl Diag {
    pub fn error(message: impl Into<String>, span: SourceSpan, label: impl Into<String>) -> Self {
        Self {
            level: Level::Error,
            message: message.into(),
            span,
            label: label.into(),
            help: None,
        }
    }

    pub fn warning(message: impl Into<String>, span: SourceSpan, label: impl Into<String>) -> Self {
        Self {
            level: Level::Warning,
            message: message.into(),
            span,
            label: label.into(),
            help: None,
        }
    }

    pub fn help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    fn into_miette(self, name: &str, src: Arc<str>) -> CompilerDiagnostic {
        let named_src = NamedSource::new(name, src);
        match self.level {
            Level::Error => CompilerDiagnostic::Error {
                message: self.message,
                src: named_src,
                span: self.span,
                label: self.label,
                help: self.help,
            },
            Level::Warning => CompilerDiagnostic::Warning {
                message: self.message,
                src: named_src,
                span: self.span,
                label: self.label,
                help: self.help,
            },
        }
    }
}

pub struct DiagCtx {
    name: String,
    source: Arc<str>,
    diags: Vec<Diag>,
}

impl DiagCtx {
    pub fn new(name: impl Into<String>, source: impl Into<Arc<str>>) -> Self {
        Self {
            name: name.into(),
            source: source.into(),
            diags: Vec::new(),
        }
    }

    pub fn emit(&mut self, diag: Diag) {
        self.diags.push(diag);
    }

    pub fn has_errors(&self) -> bool {
        self.diags.iter().any(|d| d.level == Level::Error)
    }

    pub fn finish(self) -> Result<Vec<CompilerDiagnostic>, DiagnosticReport> {
        if self.diags.is_empty() {
            return Ok(Vec::new());
        }

        let has_errors = self.has_errors();

        let bound_diags: Vec<_> = self
            .diags
            .into_iter()
            .map(|d| d.into_miette(&self.name, self.source.clone()))
            .collect();

        if has_errors {
            Err(DiagnosticReport {
                name: self.name,
                diagnostics: bound_diags,
            })
        } else {
            Ok(bound_diags)
        }
    }
}
