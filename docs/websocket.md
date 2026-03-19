# WebSocket Protocol

The `/ws/:code` endpoint implements a JSON message protocol for multiplayer collaboration rooms.

## Inbound messages (client -> server)

| `type` | Fields | Description |
|--------|--------|-------------|
| `set_name` | `name: string` | Set display name (max 64 chars) |
| `vote` | `transform: string`, `dir: "up"|"down"` | Vote on a transform |
| `surgery` | `token_index: number`, `new_text: string`, `old_text: string` | Edit a token |
| `chat` | `text: string`, `token_index: number` | Send a chat message |
| `record_start` | -- | Begin session recording |
| `record_stop` | -- | End session recording |
| `token` | (TokenEvent fields) | Host broadcasts a token to guests |
| `ping` | -- | Keepalive (server replies with `pong`) |

## Outbound messages (server -> client)

| `type` | Fields | Description |
|--------|--------|-------------|
| `welcome` | `participant_id`, `room_code`, `is_host` | Sent on connect |
| `participant_join` | `participant` | New participant joined |
| `participant_leave` | `participant_id` | Participant left |
| `participant_update` | `participant` | Name/color changed |
| `vote_update` | `votes` | Vote tally changed |
| `surgery` | `token_index`, `new_text`, `old_text`, `participant_id` | Token edited |
| `chat` | `text`, `token_index`, `participant_id`, `name` | Chat message |
| `record_started` / `record_stopped` | -- | Recording state changed |
| `replay_event` | (TokenEvent) | Replaying a recorded token |
| `replay_done` | -- | Replay finished |
| `stream_done` | -- | Host's LLM stream completed |
| `pong` | -- | Keepalive response |
| `error` | `message` | Error notification |
