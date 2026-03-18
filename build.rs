//! Build script: tracks changes to the embedded web UI.
//!
//! The web UI is a single embedded HTML file at `static/index.html`,
//! loaded at compile time by `include_str!` in `src/web.rs`.
//! Cargo will recompile when this file changes.

fn main() {
    println!("cargo:rerun-if-changed=static/index.html");
}
