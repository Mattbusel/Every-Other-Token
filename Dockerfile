# ---- Stage 1: dependency caching with cargo-chef -------------------------
# cargo-chef computes a "recipe" (dependency manifest only) so that the heavy
# `cargo build --release` step is cached in a layer that only invalidates when
# Cargo.toml / Cargo.lock change — not when application source changes.
FROM rust:1.75-slim AS chef
RUN cargo install cargo-chef --locked
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ---- Stage 2: build dependencies (cached layer) ---------------------------
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build and cache dependencies only.
RUN cargo chef cook --release --recipe-path recipe.json
# Now copy source and build the binary. This layer re-runs only on src/ changes.
COPY . .
RUN cargo build --release

# ---- Stage 3: minimal runtime image --------------------------------------
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/every-other-token /usr/local/bin/
EXPOSE 8888
ENTRYPOINT ["every-other-token"]
CMD ["--help"]
