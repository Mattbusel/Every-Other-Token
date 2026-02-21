# Agent Coordination

## Roles
- Builder Agent: implements features, owns specific modules
- Test Agent: writes tests only, read-only on src/

## Module Ownership
- src/main.rs — CLI, transformations, providers
- src/web/ — HTTP server, SSE, embedded HTML
- tests/ — test coverage

## Rules
- Claim your module at start: "PROTOCOL ACKNOWLEDGED — claiming X"
- Do not edit modules owned by another active agent
- Run cargo test --release before committing
- Commit message format: [feat/fix/docs] description
- Push to both: git push origin main && git push origin master
