//! Build script: assembles the web UI from split source files.
//!
//! Source files:
//!   src/ui/style.css    — all CSS (no <style> tags)
//!   src/ui/app.js       — all JavaScript (no <script> tags)
//!   src/ui/template.html — HTML skeleton with {{STYLE}} and {{SCRIPT}} markers
//!
//! Output: static/index.html — the single embedded file loaded by web.rs.
//!
//! When the source files are present and valid, build.rs regenerates
//! static/index.html on every `cargo build`.  If the sources are absent the
//! existing static/index.html is left untouched so a fresh checkout still
//! compiles without running the assembler.

use std::fs;
use std::path::Path;

fn main() {
    let style_path    = Path::new("src/ui/style.css");
    let script_path   = Path::new("src/ui/app.js");
    let template_path = Path::new("src/ui/template.html");
    let output_path   = Path::new("static/index.html");

    // Re-run this build script whenever any source file changes.
    println!("cargo:rerun-if-changed=src/ui/style.css");
    println!("cargo:rerun-if-changed=src/ui/app.js");
    println!("cargo:rerun-if-changed=src/ui/template.html");

    // Only assemble when all three source files exist.
    // During a fresh checkout the files may be absent if the developer has not
    // yet split the monolithic index.html — in that case we skip silently and
    // use the pre-existing static/index.html.
    if !style_path.exists() || !script_path.exists() || !template_path.exists() {
        return;
    }

    let style    = fs::read_to_string(style_path).expect("failed to read src/ui/style.css");
    let script   = fs::read_to_string(script_path).expect("failed to read src/ui/app.js");
    let template = fs::read_to_string(template_path).expect("failed to read src/ui/template.html");

    // Only assemble when the template actually contains the markers — a stub
    // template (no markers) means the split is incomplete, skip silently.
    if !template.contains("{{STYLE}}") || !template.contains("{{SCRIPT}}") {
        return;
    }

    let html = template
        .replace("{{STYLE}}", &style)
        .replace("{{SCRIPT}}", &script);

    fs::write(output_path, html).expect("failed to write static/index.html");
}
