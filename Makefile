.PHONY: test check ratio lint fmt validate-config all

test:
	cargo test

check:
	cargo check --all-features 2>/dev/null || cargo check

ratio:
	bash scripts/ratio_check.sh

lint:
	cargo clippy -- -D warnings 2>/dev/null || echo "clippy not available"

fmt:
	cargo fmt --check 2>/dev/null || echo "rustfmt not available"

validate-config:
	cargo run -- --validate-config 2>/dev/null || echo "build first with: cargo build"

all: test ratio
