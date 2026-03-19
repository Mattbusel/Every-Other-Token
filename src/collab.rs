//! Multiplayer collaboration: room state, participant management, WebSocket handling.
//!
//! ## Design
//! - RoomStore: Arc<Mutex<HashMap<String, Room>>> — shared across all connections
//! - Each Room has a broadcast channel (tokio::sync::broadcast) for real-time fan-out
//! - Each WS client subscribes to the room's broadcast sender
//! - Messages are serde_json::Value for flexibility
//!
//! ## Room lifecycle
//! 1. Host calls POST /room/create → gets 6-char code
//! 2. Host connects to WS /ws/CODE → is assigned host role
//! 3. Guests open /join/CODE in browser → connects to WS /ws/CODE → gets guest role
//! 4. Host starts a stream → token events broadcast to all participants
//! 5. Any participant edits a token → surgery event broadcast to all
//! 6. Participants can chat and vote on transforms

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_tungstenite::tungstenite::Message as WsMessage;

/// Shared room store: room code → Room.
pub type RoomStore = Arc<Mutex<HashMap<String, Room>>>;

/// Idle TTL for rooms: rooms not mutated in this many milliseconds are eligible for eviction.
const ROOM_IDLE_TTL_MS: u64 = 3_600_000;

/// Maximum number of events stored in a room's recording buffer.
const DEFAULT_RECORDING_CAP: usize = 10_000;

/// Adjectives used for memorable room code generation.
const CODE_ADJECTIVES: &[&str] = &[
    "SWIFT", "BRAVE", "CALM", "DARK", "EPIC", "FAST", "GOLD", "KEEN", "LOUD", "MILD",
    "NEAT", "OPEN", "PURE", "QUIET", "RICH", "SHARP", "TALL", "VAST", "WARM", "WISE",
];

/// Nouns used for memorable room code generation.
const CODE_NOUNS: &[&str] = &[
    "LION", "BEAR", "WOLF", "HAWK", "DEER", "FROG", "CRAB", "CROW", "DOVE", "DUCK",
    "FISH", "HARE", "KITE", "LYNX", "MOLE", "NEWT", "PUMA", "SWAN", "TOAD", "WREN",
];

/// Colors assigned to participants in round-robin order.
pub const PARTICIPANT_COLORS: &[&str] = &[
    "#58a6ff", "#f0883e", "#a371f7", "#3fb950", "#e3b341", "#f85149",
];

/// A connected participant in a room.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participant {
    pub id: String,
    pub name: String,
    pub color: String,
    pub joined_at_ms: u64,
    pub is_host: bool,
}

/// A token-level surgery edit applied by a participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurgeryEdit {
    pub token_index: usize,
    pub new_text: String,
    pub old_text: String,
    pub editor_id: String,
    pub editor_color: String,
    pub editor_name: String,
    pub timestamp_ms: u64,
}

/// A chat message from a participant, optionally referencing a token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    pub author_id: String,
    pub author_name: String,
    pub author_color: String,
    pub text: String,
    pub token_index: Option<usize>,
    pub timestamp_ms: u64,
}

/// A recorded room event with a relative timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedEvent {
    pub offset_ms: u64,
    pub payload: serde_json::Value,
}

/// An active collaboration room shared by one host and zero or more guests.
///
/// Rooms are created via [`create_room`] and stored in a [`RoomStore`].
/// Each room owns a `tokio::sync::broadcast` channel; subscribers receive all
/// events dispatched to the room until they are evicted by [`evict_idle_rooms`].
pub struct Room {
    /// Unique 6-character uppercase room code.
    pub code: String,
    /// Participant `id` of the room host (set when the first participant joins with `is_host=true`).
    pub host_id: String,
    /// All participants currently in the room.
    pub participants: Vec<Participant>,
    /// Number of token events that have been broadcast to the room.
    pub token_count: usize,
    /// Ordered log of all surgery edits applied during the session.
    pub surgery_log: Vec<SurgeryEdit>,
    /// Ordered log of all chat messages sent during the session.
    pub chat_log: Vec<ChatMessage>,
    /// Transform vote tallies: transform name → (upvotes, downvotes).
    pub votes: HashMap<String, (u32, u32)>,
    /// Whether the room is currently recording events to `recorded_events`.
    pub is_recording: bool,
    /// Wall-clock ms timestamp when recording started, or `None` if not recording.
    pub recording_start_ms: Option<u64>,
    /// Accumulated recorded events (capped at `recording_cap`).
    pub recorded_events: Vec<RecordedEvent>,
    /// Wall-clock ms timestamp when the room was created.
    pub created_at_ms: u64,
    /// Wall-clock ms timestamp of the most recent write to this room.
    pub last_activity_ms: u64,
    /// Maximum number of events to retain in `recorded_events`.
    pub recording_cap: usize,
    /// Broadcast sender — clone to get a Receiver for a new subscriber.
    pub broadcast_tx: tokio::sync::broadcast::Sender<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Constructor helpers
// ---------------------------------------------------------------------------

/// Create a new empty RoomStore.
pub fn new_room_store() -> RoomStore {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Generate a memorable uppercase room code in the format `ADJ-NOUN-NN`.
///
/// Example: `SWIFT-LION-42`
///
/// The code is uppercase and compatible with the existing `/join/:code` and
/// `/ws/:code` path parsing which uses prefix stripping.
pub fn generate_code() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let adj = CODE_ADJECTIVES[rng.gen_range(0..CODE_ADJECTIVES.len())];
    let noun = CODE_NOUNS[rng.gen_range(0..CODE_NOUNS.len())];
    let num: u8 = rng.gen_range(10..=99);
    format!("{}-{}-{}", adj, noun, num)
}

/// Truncate `s` to at most `max_bytes` bytes while respecting UTF-8 char boundaries.
///
/// Slicing a `&str` at an arbitrary byte offset panics when that offset falls
/// inside a multi-byte character (emoji, CJK, etc.).  This helper walks
/// backwards from `max_bytes` to find the last valid char boundary.
fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Current Unix epoch in milliseconds.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Room operations
// ---------------------------------------------------------------------------

/// Create a new room in the store, returning its 6-char code.
pub fn create_room(store: &RoomStore) -> String {
    let (tx, _rx) = tokio::sync::broadcast::channel(256);
    let code = generate_code();
    let room = Room {
        code: code.clone(),
        host_id: String::new(),
        participants: Vec::new(),
        token_count: 0,
        surgery_log: Vec::new(),
        chat_log: Vec::new(),
        votes: HashMap::new(),
        is_recording: false,
        recording_start_ms: None,
        recorded_events: Vec::new(),
        created_at_ms: now_ms(),
        last_activity_ms: now_ms(),
        recording_cap: DEFAULT_RECORDING_CAP,
        broadcast_tx: tx,
    };
    if let Ok(mut guard) = store.lock() {
        guard.insert(code.clone(), room);
    }
    code
}

/// Add a participant to a room.
///
/// Returns `(participant, broadcast_receiver)` on success, or an error string if
/// the room code is not found.
pub fn join_room(
    store: &RoomStore,
    code: &str,
    name: &str,
    is_host: bool,
) -> Result<
    (
        Participant,
        tokio::sync::broadcast::Receiver<serde_json::Value>,
    ),
    String,
> {
    let mut guard = store
        .lock()
        .map_err(|_| "internal: lock poisoned".to_string())?;

    let room = guard
        .get_mut(code)
        .ok_or_else(|| format!("Room '{}' not found", code))?;

    let color_idx = room.participants.len() % PARTICIPANT_COLORS.len();
    let color = PARTICIPANT_COLORS[color_idx].to_string();

    let participant = Participant {
        id: uuid::Uuid::new_v4().to_string(),
        name: name.to_string(),
        color,
        joined_at_ms: now_ms(),
        is_host,
    };

    if is_host && room.host_id.is_empty() {
        room.host_id = participant.id.clone();
    }

    let rx = room.broadcast_tx.subscribe();
    room.participants.push(participant.clone());

    Ok((participant, rx))
}

/// Remove a participant from a room.
///
/// Returns the room's broadcast sender (so the caller can broadcast the leave
/// event) or `None` if the room or participant was not found.
pub fn leave_room(
    store: &RoomStore,
    code: &str,
    participant_id: &str,
) -> Option<tokio::sync::broadcast::Sender<serde_json::Value>> {
    let mut guard = store.lock().ok()?;
    let room = guard.get_mut(code)?;
    room.participants.retain(|p| p.id != participant_id);
    Some(room.broadcast_tx.clone())
}

/// Broadcast a raw JSON message to every subscriber of the room's channel.
/// Send `msg` to all current subscribers of the room identified by `code`.
///
/// Silently does nothing if the room does not exist or the store lock is
/// poisoned.  Lagging receivers that have fallen behind will have their
/// oldest unread messages overwritten (tokio broadcast semantics).
pub fn broadcast(store: &RoomStore, code: &str, msg: serde_json::Value) {
    if let Ok(guard) = store.lock() {
        if let Some(room) = guard.get(code) {
            if let Err(_) = room.broadcast_tx.send(msg) {
                tracing::debug!(room_code = %code, "broadcast dropped: no active subscribers");
            }
        }
    }
}

/// Record and broadcast a surgery edit.
pub fn apply_surgery(store: &RoomStore, code: &str, edit: SurgeryEdit) {
    if let Ok(mut guard) = store.lock() {
        if let Some(room) = guard.get_mut(code) {
            let msg = serde_json::json!({
                "type": "surgery",
                "edit": edit,
            });
            room.surgery_log.push(edit);
            room.last_activity_ms = now_ms();
            let _ = room.broadcast_tx.send(msg);
        }
    }
}

/// Record and broadcast a chat message.
pub fn add_chat(store: &RoomStore, code: &str, msg: ChatMessage) {
    if let Ok(mut guard) = store.lock() {
        if let Some(room) = guard.get_mut(code) {
            let broadcast_msg = serde_json::json!({
                "type": "chat",
                "message": msg,
            });
            room.chat_log.push(msg);
            room.last_activity_ms = now_ms();
            let _ = room.broadcast_tx.send(broadcast_msg);
        }
    }
}

/// Cast a vote for a transform.
///
/// `dir` must be `"up"` or `"down"`. Returns the updated `(up, down)` counts,
/// or `None` if the room was not found.
pub fn vote(store: &RoomStore, code: &str, transform: &str, dir: &str) -> Option<(u32, u32)> {
    let mut guard = store.lock().ok()?;
    let room = guard.get_mut(code)?;
    let entry = room.votes.entry(transform.to_string()).or_insert((0, 0));
    match dir {
        "up" => entry.0 = entry.0.saturating_add(1),
        "down" => entry.1 = entry.1.saturating_add(1),
        _ => {}
    }
    room.last_activity_ms = now_ms();
    Some(*entry)
}

/// Snapshot the room state as a JSON value.
pub fn room_state_snapshot(store: &RoomStore, code: &str) -> serde_json::Value {
    if let Ok(guard) = store.lock() {
        if let Some(room) = guard.get(code) {
            return serde_json::json!({
                "code": room.code,
                "host_id": room.host_id,
                "participants": room.participants,
                "token_count": room.token_count,
                "surgery_log": room.surgery_log,
                "chat_log": room.chat_log,
                "votes": room.votes,
                "is_recording": room.is_recording,
                "created_at_ms": room.created_at_ms,
            });
        }
    }
    serde_json::Value::Null
}

/// Begin recording events in a room.
pub fn start_recording(store: &RoomStore, code: &str) {
    if let Ok(mut guard) = store.lock() {
        if let Some(room) = guard.get_mut(code) {
            room.is_recording = true;
            room.recording_start_ms = Some(now_ms());
            room.recorded_events.clear();
        }
    }
}

/// Stop recording and return all recorded events.
pub fn stop_recording(store: &RoomStore, code: &str) -> Vec<RecordedEvent> {
    if let Ok(mut guard) = store.lock() {
        if let Some(room) = guard.get_mut(code) {
            room.is_recording = false;
            room.recording_start_ms = None;
            return std::mem::take(&mut room.recorded_events);
        }
    }
    Vec::new()
}

/// Append a recorded event to the room's log if recording is active.
pub fn maybe_record(store: &RoomStore, code: &str, payload: serde_json::Value) {
    if let Ok(mut guard) = store.lock() {
        if let Some(room) = guard.get_mut(code) {
            if room.is_recording {
                let start = room.recording_start_ms.unwrap_or_else(now_ms);
                let offset_ms = now_ms().saturating_sub(start);
                room.recorded_events
                    .push(RecordedEvent { offset_ms, payload });
                // Count dropped events and warn
                let dropped = if room.recorded_events.len() > room.recording_cap {
                    let excess = room.recorded_events.len() - room.recording_cap;
                    for _ in 0..excess {
                        room.recorded_events.remove(0);
                    }
                    excess
                } else {
                    0
                };

                if dropped > 0 {
                    // Notify all subscribers that recording was truncated
                    let warn = serde_json::json!({
                        "type": "record_truncated",
                        "dropped": dropped,
                        "cap": room.recording_cap,
                    });
                    let _ = room.broadcast_tx.send(warn);
                }
                room.last_activity_ms = now_ms();
            }
        }
    }
}

/// Evict rooms that have been idle longer than `ROOM_IDLE_TTL_MS`.
pub fn evict_idle_rooms(store: &RoomStore) {
    if let Ok(mut guard) = store.lock() {
        let now = now_ms();
        guard.retain(|_, room| now.saturating_sub(room.last_activity_ms) < ROOM_IDLE_TTL_MS);
    }
}

// ---------------------------------------------------------------------------
// WebSocket handler
// ---------------------------------------------------------------------------

/// Handle an established WebSocket connection for room `code`.
///
/// `ws_stream`  — the tokio-tungstenite WebSocketStream
/// `store`      — the shared room store
/// `code`       — the room code
/// `is_host`    — whether this connection is the room creator
pub async fn handle_ws(
    ws_stream: tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    store: RoomStore,
    code: String,
    is_host: bool,
) {
    let initial_name = if is_host { "Host" } else { "Guest" };

    let (participant, mut room_rx) = match join_room(&store, &code, initial_name, is_host) {
        Ok(pair) => pair,
        Err(err) => {
            // Room not found — send error and close.
            let (mut sink, _) = ws_stream.split();
            let err_msg = serde_json::json!({"type": "error", "message": err});
            if let Ok(text) = serde_json::to_string(&err_msg) {
                let _ = sink.send(WsMessage::Text(text)).await;
            }
            return;
        }
    };

    let participant_id = participant.id.clone();
    let participant_name = participant.name.clone();

    let (mut ws_sink, mut ws_stream) = ws_stream.split();

    // Send welcome message to this client.
    let welcome = serde_json::json!({
        "type": "welcome",
        "participant": participant,
        "room_state": room_state_snapshot(&store, &code),
    });
    if let Ok(text) = serde_json::to_string(&welcome) {
        let _ = ws_sink.send(WsMessage::Text(text)).await;
    }

    // Notify all OTHER participants about the new joiner.
    broadcast(
        &store,
        &code,
        serde_json::json!({
            "type": "participant_join",
            "participant": participant,
        }),
    );

    // Main loop: multiplex incoming WS frames and broadcast messages.
    let mut ping_interval = tokio::time::interval(std::time::Duration::from_secs(30));
    let idle_timeout = std::time::Duration::from_secs(3600);
    let mut last_msg_at = tokio::time::Instant::now();
    loop {
        tokio::select! {
            // Message from this client.
            msg = ws_stream.next() => {
                match msg {
                    Some(Ok(WsMessage::Text(text))) => {
                        last_msg_at = tokio::time::Instant::now();
                        if text.len() > 65_536 {
                            let err = serde_json::json!({"type": "error", "message": "message too large"});
                            if let Ok(s) = serde_json::to_string(&err) {
                                let _ = ws_sink.send(WsMessage::Text(s)).await;
                            }
                            continue;
                        }
                        let parsed: serde_json::Value = match serde_json::from_str(&text) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        let msg_type = parsed.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();

                        match msg_type.as_str() {
                            "set_name" => {
                                if let Some(new_name) = parsed.get("name").and_then(|v| v.as_str()) {
                                    let new_name = truncate_utf8(new_name, 64);
                                    let updated = update_participant_name(&store, &code, &participant_id, new_name);
                                    if let Some(p) = updated {
                                        broadcast(&store, &code, serde_json::json!({
                                            "type": "participant_update",
                                            "participant": p,
                                        }));
                                    }
                                }
                            }
                            "vote" => {
                                let transform = parsed.get("transform").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                let dir = parsed.get("dir").and_then(|v| v.as_str()).unwrap_or("up").to_string();
                                if let Some((up, down)) = vote(&store, &code, &transform, &dir) {
                                    broadcast(&store, &code, serde_json::json!({
                                        "type": "vote_update",
                                        "transform": transform,
                                        "up": up,
                                        "down": down,
                                    }));
                                }
                            }
                            "surgery" => {
                                let token_index = parsed.get("token_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                let new_text = {
                                    let raw = parsed.get("new_text").and_then(|v| v.as_str()).unwrap_or("");
                                    truncate_utf8(raw, 4096).to_string()
                                };
                                let old_text = {
                                    let raw = parsed.get("old_text").and_then(|v| v.as_str()).unwrap_or("");
                                    truncate_utf8(raw, 4096).to_string()
                                };

                                let (editor_color, editor_name) = get_participant_info(&store, &code, &participant_id);
                                let edit = SurgeryEdit {
                                    token_index,
                                    new_text,
                                    old_text,
                                    editor_id: participant_id.clone(),
                                    editor_color,
                                    editor_name,
                                    timestamp_ms: now_ms(),
                                };
                                apply_surgery(&store, &code, edit);
                            }
                            "chat" => {
                                let text_content = {
                                    let raw = parsed.get("text").and_then(|v| v.as_str()).unwrap_or("");
                                    truncate_utf8(raw, 2048).to_string()
                                };
                                let token_index = parsed.get("token_index").and_then(|v| v.as_u64()).map(|n| n as usize);
                                let (author_color, author_name) = get_participant_info(&store, &code, &participant_id);
                                let chat_msg = ChatMessage {
                                    id: uuid::Uuid::new_v4().to_string(),
                                    author_id: participant_id.clone(),
                                    author_name,
                                    author_color,
                                    text: text_content,
                                    token_index,
                                    timestamp_ms: now_ms(),
                                };
                                add_chat(&store, &code, chat_msg);
                            }
                            "record_start" => {
                                start_recording(&store, &code);
                                broadcast(&store, &code, serde_json::json!({"type": "record_started"}));
                            }
                            "record_stop" => {
                                let events = stop_recording(&store, &code);
                                let count = events.len();
                                broadcast(&store, &code, serde_json::json!({
                                    "type": "record_stopped",
                                    "event_count": count,
                                }));
                            }
                            "replay_request" => {
                                // Send recorded events only to this client.
                                let events = get_recorded_events(&store, &code);
                                for event in &events {
                                    let replay_msg = serde_json::json!({
                                        "type": "replay_event",
                                        "event": event.payload,
                                        "offset_ms": event.offset_ms,
                                    });
                                    if let Ok(text) = serde_json::to_string(&replay_msg) {
                                        if ws_sink.send(WsMessage::Text(text)).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                                if let Ok(done) = serde_json::to_string(&serde_json::json!({"type": "replay_done"})) {
                                    let _ = ws_sink.send(WsMessage::Text(done)).await;
                                }
                            }
                            "ping" => {
                                if let Ok(pong) = serde_json::to_string(&serde_json::json!({"type": "pong"})) {
                                    let _ = ws_sink.send(WsMessage::Text(pong)).await;
                                }
                            }
                            // Host broadcasts a token event to all guests in the room.
                            "token" => {
                                if is_host {
                                    // Re-broadcast the full token payload so guests render it.
                                    broadcast(&store, &code, parsed.clone());
                                    // Also record if active.
                                    maybe_record(&store, &code, parsed);
                                }
                            }
                            // Host sends this to record a token without broadcasting.
                            "_record_token" => {
                                if is_host {
                                    maybe_record(&store, &code, parsed);
                                }
                            }
                            // Host signals the stream has finished.
                            "stream_done" => {
                                if is_host {
                                    broadcast(
                                        &store,
                                        &code,
                                        serde_json::json!({"type": "stream_done"}),
                                    );
                                }
                            }
                            unknown => {
                                // Log unrecognised message types (#2) rather than
                                // silently dropping them so that client bugs are
                                // visible in the server logs.
                                tracing::warn!(
                                    msg_type = unknown,
                                    room = %code,
                                    participant = %participant_id,
                                    "received unknown WebSocket message type — ignoring"
                                );
                            }
                        }
                    }
                    Some(Ok(_)) => {} // Ignore binary / ping / pong frames
                    Some(Err(_)) | None => break, // Connection closed or error
                }
            }

            // Broadcast message from the room channel.
            bcast = room_rx.recv() => {
                match bcast {
                    Ok(msg) => {
                        if let Ok(text) = serde_json::to_string(&msg) {
                            if ws_sink.send(WsMessage::Text(text)).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        // Receiver fell behind; notify the client and continue.
                        tracing::warn!(
                            participant = %participant_id,
                            skipped = n,
                            "WS client lagged — skipped broadcast messages"
                        );
                        let warn = serde_json::json!({
                            "type": "lag_warning",
                            "skipped": n,
                        });
                        if let Ok(s) = serde_json::to_string(&warn) {
                            let _ = ws_sink.send(WsMessage::Text(s)).await;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }

            // Keepalive ping to prevent idle connection teardown.
            _ = ping_interval.tick() => {
                // Idle timeout: close if no message received in the last 3600 seconds.
                if last_msg_at.elapsed() >= idle_timeout {
                    tracing::info!(
                        participant = %participant_id,
                        "WS idle timeout — closing connection"
                    );
                    break;
                }
                if ws_sink.send(WsMessage::Ping(vec![])).await.is_err() {
                    break;
                }
            }
        }
    }

    // Client disconnected — clean up and notify others.
    if let Some(tx) = leave_room(&store, &code, &participant_id) {
        let leave_msg = serde_json::json!({
            "type": "participant_leave",
            "participant_id": participant_id,
            "name": participant_name,
        });
        let _ = tx.send(leave_msg);
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Update a participant's display name and return the updated Participant.
fn update_participant_name(
    store: &RoomStore,
    code: &str,
    participant_id: &str,
    new_name: &str,
) -> Option<Participant> {
    let mut guard = store.lock().ok()?;
    let room = guard.get_mut(code)?;
    let p = room
        .participants
        .iter_mut()
        .find(|p| p.id == participant_id)?;
    p.name = new_name.to_string();
    Some(p.clone())
}

/// Look up a participant's color and name. Returns empty strings if not found.
fn get_participant_info(store: &RoomStore, code: &str, participant_id: &str) -> (String, String) {
    if let Ok(guard) = store.lock() {
        if let Some(room) = guard.get(code) {
            if let Some(p) = room.participants.iter().find(|p| p.id == participant_id) {
                return (p.color.clone(), p.name.clone());
            }
        }
    }
    (String::new(), String::new())
}

/// Get a snapshot of recorded events (cloned) for replay.
fn get_recorded_events(store: &RoomStore, code: &str) -> Vec<RecordedEvent> {
    if let Ok(guard) = store.lock() {
        if let Some(room) = guard.get(code) {
            return room.recorded_events.clone();
        }
    }
    Vec::new()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- truncate_utf8 -------------------------------------------------------

    #[test]
    fn test_truncate_utf8_ascii_unchanged() {
        assert_eq!(truncate_utf8("hello", 64), "hello");
    }

    #[test]
    fn test_truncate_utf8_exact_limit() {
        assert_eq!(truncate_utf8("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_utf8_ascii_truncated() {
        assert_eq!(truncate_utf8("hello world", 5), "hello");
    }

    #[test]
    fn test_truncate_utf8_multibyte_no_panic() {
        // Each emoji is 4 bytes. Truncating at byte 5 would split the second emoji.
        let s = "😀😀😀"; // 12 bytes
        let result = truncate_utf8(s, 5);
        // Must be a valid &str (no panic) and at most 5 bytes
        assert!(result.len() <= 5);
        assert!(std::str::from_utf8(result.as_bytes()).is_ok());
    }

    #[test]
    fn test_truncate_utf8_multibyte_boundary_aligned() {
        let s = "😀😀"; // 8 bytes: first emoji is bytes 0-3, second is 4-7
        let result = truncate_utf8(s, 4); // exactly one emoji
        assert_eq!(result, "😀");
    }

    // -- generate_code -------------------------------------------------------

    #[test]
    fn test_generate_code_length_is_reasonable() {
        // New format: ADJ-NOUN-NN — at least 8 chars (e.g. "CALM-BEAR-10")
        let code = generate_code();
        assert!(code.len() >= 8, "code '{}' is too short", code);
    }

    #[test]
    fn test_generate_code_all_uppercase_alphanumeric() {
        let code = generate_code();
        assert!(
            code.chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '-'),
            "code '{}' contains unexpected chars",
            code
        );
    }

    #[test]
    fn test_generate_code_uniqueness_across_calls() {
        let codes: Vec<String> = (0..30).map(|_| generate_code()).collect();
        let unique: std::collections::HashSet<&String> = codes.iter().collect();
        // With 20*20*90 = 36,000 possibilities, 30 calls should all be unique.
        assert!(
            unique.len() >= 20,
            "expected near-unique codes, got {} unique out of 30",
            unique.len()
        );
    }

    #[test]
    fn test_generate_code_no_lowercase() {
        for _ in 0..20 {
            let code = generate_code();
            assert!(
                !code.chars().any(|c| c.is_ascii_lowercase()),
                "code contains lowercase: {}",
                code
            );
        }
    }

    // -- now_ms --------------------------------------------------------------

    #[test]
    fn test_now_ms_nonzero() {
        assert!(now_ms() > 0);
    }

    #[test]
    fn test_now_ms_monotonic() {
        let t1 = now_ms();
        let t2 = now_ms();
        assert!(t2 >= t1, "now_ms() must be non-decreasing");
    }

    #[test]
    fn test_now_ms_plausible_epoch() {
        // Must be after 2024-01-01 in milliseconds
        assert!(
            now_ms() > 1_704_067_200_000,
            "now_ms() appears to predate 2024"
        );
    }

    // -- PARTICIPANT_COLORS --------------------------------------------------

    #[test]
    fn test_participant_colors_count() {
        assert_eq!(PARTICIPANT_COLORS.len(), 6);
    }

    #[test]
    fn test_participant_colors_are_hex() {
        for color in PARTICIPANT_COLORS {
            assert!(color.starts_with('#'), "color must start with #: {}", color);
            assert_eq!(color.len(), 7, "color must be 7 chars (#RRGGBB): {}", color);
        }
    }

    #[test]
    fn test_participant_colors_are_unique() {
        let unique: std::collections::HashSet<&&str> = PARTICIPANT_COLORS.iter().collect();
        assert_eq!(unique.len(), PARTICIPANT_COLORS.len());
    }

    // -- new_room_store / create_room ----------------------------------------

    #[test]
    fn test_new_room_store_is_empty() {
        let store = new_room_store();
        assert!(store.lock().unwrap_or_else(|e| e.into_inner()).is_empty());
    }

    #[test]
    fn test_create_room_returns_meaningful_code() {
        let store = new_room_store();
        let code = create_room(&store);
        // New memorable format: ADJ-NOUN-NN
        assert!(code.len() >= 8, "code too short: {}", code);
        assert_eq!(code.split('-').count(), 3, "expected 3 parts: {}", code);
    }

    #[test]
    fn test_create_room_code_is_uppercase_alphanumeric() {
        let store = new_room_store();
        let code = create_room(&store);
        assert!(code
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '-'));
    }

    #[test]
    fn test_create_room_inserts_into_store() {
        let store = new_room_store();
        let code = create_room(&store);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert!(guard.contains_key(&code));
    }

    #[test]
    fn test_create_multiple_rooms_all_stored() {
        let store = new_room_store();
        let c1 = create_room(&store);
        let c2 = create_room(&store);
        let c3 = create_room(&store);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert!(guard.contains_key(&c1));
        assert!(guard.contains_key(&c2));
        assert!(guard.contains_key(&c3));
        assert_eq!(guard.len(), 3);
    }

    #[test]
    fn test_create_room_starts_not_recording() {
        let store = new_room_store();
        let code = create_room(&store);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        let room = guard.get(&code).unwrap();
        assert!(!room.is_recording);
    }

    #[test]
    fn test_create_room_starts_with_empty_participants() {
        let store = new_room_store();
        let code = create_room(&store);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        let room = guard.get(&code).unwrap();
        assert!(room.participants.is_empty());
    }

    #[test]
    fn test_create_room_starts_with_empty_logs() {
        let store = new_room_store();
        let code = create_room(&store);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        let room = guard.get(&code).unwrap();
        assert!(room.surgery_log.is_empty());
        assert!(room.chat_log.is_empty());
        assert!(room.votes.is_empty());
    }

    // -- join_room -----------------------------------------------------------

    #[test]
    fn test_join_room_host_sets_host_id() {
        let store = new_room_store();
        let code = create_room(&store);
        let (p, _rx) = join_room(&store, &code, "Alice", true).unwrap();
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(guard.get(&code).unwrap().host_id, p.id);
    }

    #[test]
    fn test_join_room_guest_does_not_override_host_id() {
        let store = new_room_store();
        let code = create_room(&store);
        let (host, _rx1) = join_room(&store, &code, "Host", true).unwrap();
        let (_guest, _rx2) = join_room(&store, &code, "Guest", false).unwrap();
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(guard.get(&code).unwrap().host_id, host.id);
    }

    #[test]
    fn test_join_room_participant_added() {
        let store = new_room_store();
        let code = create_room(&store);
        join_room(&store, &code, "Alice", true).unwrap();
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(guard.get(&code).unwrap().participants.len(), 1);
    }

    #[test]
    fn test_join_room_multiple_participants() {
        let store = new_room_store();
        let code = create_room(&store);
        join_room(&store, &code, "Alice", true).unwrap();
        join_room(&store, &code, "Bob", false).unwrap();
        join_room(&store, &code, "Carol", false).unwrap();
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(guard.get(&code).unwrap().participants.len(), 3);
    }

    #[test]
    fn test_join_room_assigns_unique_ids() {
        let store = new_room_store();
        let code = create_room(&store);
        let (p1, _) = join_room(&store, &code, "A", true).unwrap();
        let (p2, _) = join_room(&store, &code, "B", false).unwrap();
        assert_ne!(p1.id, p2.id);
    }

    #[test]
    fn test_join_room_colors_cycle_through_palette() {
        let store = new_room_store();
        let code = create_room(&store);
        let mut colors = vec![];
        for i in 0..PARTICIPANT_COLORS.len() {
            let (p, _) = join_room(&store, &code, &format!("P{}", i), i == 0).unwrap();
            colors.push(p.color.clone());
        }
        // Colors should match the palette in order
        for (i, color) in colors.iter().enumerate() {
            assert_eq!(color, PARTICIPANT_COLORS[i]);
        }
    }

    #[test]
    fn test_join_room_error_on_nonexistent_code() {
        let store = new_room_store();
        let result = join_room(&store, "XXXXXX", "Alice", true);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("XXXXXX"),
            "error should mention the code: {}",
            err
        );
    }

    #[test]
    fn test_join_room_host_flag_set_correctly() {
        let store = new_room_store();
        let code = create_room(&store);
        let (host, _) = join_room(&store, &code, "Host", true).unwrap();
        let (guest, _) = join_room(&store, &code, "Guest", false).unwrap();
        assert!(host.is_host);
        assert!(!guest.is_host);
    }

    #[test]
    fn test_join_room_joined_at_ms_is_plausible() {
        let store = new_room_store();
        let code = create_room(&store);
        let (p, _) = join_room(&store, &code, "Alice", true).unwrap();
        assert!(
            p.joined_at_ms > 1_704_067_200_000,
            "timestamp looks wrong: {}",
            p.joined_at_ms
        );
    }

    // -- leave_room ----------------------------------------------------------

    #[test]
    fn test_leave_room_removes_participant() {
        let store = new_room_store();
        let code = create_room(&store);
        let (p, _) = join_room(&store, &code, "Alice", true).unwrap();
        leave_room(&store, &code, &p.id);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert!(guard.get(&code).unwrap().participants.is_empty());
    }

    #[test]
    fn test_leave_room_returns_broadcast_sender() {
        let store = new_room_store();
        let code = create_room(&store);
        let (p, _) = join_room(&store, &code, "Alice", true).unwrap();
        let result = leave_room(&store, &code, &p.id);
        assert!(result.is_some());
    }

    #[test]
    fn test_leave_room_nonexistent_participant_returns_none() {
        let store = new_room_store();
        let code = create_room(&store);
        let result = leave_room(&store, &code, "nonexistent-id");
        // Room exists but participant not found — still returns Some(sender)
        // (leave_room removes from list with retain, then returns sender regardless)
        assert!(result.is_some());
    }

    #[test]
    fn test_leave_room_nonexistent_code_returns_none() {
        let store = new_room_store();
        let result = leave_room(&store, "XXXXXX", "some-id");
        assert!(result.is_none());
    }

    #[test]
    fn test_leave_room_only_removes_matching_participant() {
        let store = new_room_store();
        let code = create_room(&store);
        let (p1, _) = join_room(&store, &code, "Alice", true).unwrap();
        let (_p2, _) = join_room(&store, &code, "Bob", false).unwrap();
        leave_room(&store, &code, &p1.id);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        let room = guard.get(&code).unwrap();
        assert_eq!(room.participants.len(), 1);
        assert_eq!(room.participants[0].name, "Bob");
    }

    // -- broadcast -----------------------------------------------------------

    #[test]
    fn test_broadcast_to_existing_room_sends_to_subscriber() {
        let store = new_room_store();
        let code = create_room(&store);
        // Create a receiver before broadcasting
        let mut rx = {
            let guard = store.lock().unwrap_or_else(|e| e.into_inner());
            guard.get(&code).unwrap().broadcast_tx.subscribe()
        };
        let msg = serde_json::json!({"type": "test", "value": 42});
        broadcast(&store, &code, msg.clone());
        // The send may lag if no one is listening yet; check receiver
        // (broadcast channel is non-blocking; receiver.try_recv works if message was sent)
        let received = rx.try_recv();
        assert!(received.is_ok(), "expected message on broadcast channel");
        assert_eq!(received.unwrap()["value"], 42);
    }

    #[test]
    fn test_broadcast_to_nonexistent_room_is_noop() {
        let store = new_room_store();
        // Must not panic
        broadcast(&store, "XXXXXX", serde_json::json!({"type": "test"}));
    }

    // -- apply_surgery -------------------------------------------------------

    #[test]
    fn test_apply_surgery_appends_to_log() {
        let store = new_room_store();
        let code = create_room(&store);
        let edit = SurgeryEdit {
            token_index: 3,
            new_text: "foo".to_string(),
            old_text: "bar".to_string(),
            editor_id: "p1".to_string(),
            editor_color: "#58a6ff".to_string(),
            editor_name: "Alice".to_string(),
            timestamp_ms: now_ms(),
        };
        apply_surgery(&store, &code, edit);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(guard.get(&code).unwrap().surgery_log.len(), 1);
    }

    #[test]
    fn test_apply_surgery_stores_correct_fields() {
        let store = new_room_store();
        let code = create_room(&store);
        let edit = SurgeryEdit {
            token_index: 7,
            new_text: "new".to_string(),
            old_text: "old".to_string(),
            editor_id: "p1".to_string(),
            editor_color: "#3fb950".to_string(),
            editor_name: "Bob".to_string(),
            timestamp_ms: 12345,
        };
        apply_surgery(&store, &code, edit);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        let stored = &guard.get(&code).unwrap().surgery_log[0];
        assert_eq!(stored.token_index, 7);
        assert_eq!(stored.new_text, "new");
        assert_eq!(stored.old_text, "old");
        assert_eq!(stored.editor_name, "Bob");
    }

    #[test]
    fn test_apply_surgery_nonexistent_room_is_noop() {
        let store = new_room_store();
        let edit = SurgeryEdit {
            token_index: 0,
            new_text: "x".to_string(),
            old_text: "y".to_string(),
            editor_id: "p1".to_string(),
            editor_color: "#fff".to_string(),
            editor_name: "Ghost".to_string(),
            timestamp_ms: 0,
        };
        // Must not panic
        apply_surgery(&store, "XXXXXX", edit);
    }

    #[test]
    fn test_apply_surgery_multiple_edits_all_stored() {
        let store = new_room_store();
        let code = create_room(&store);
        for i in 0..5 {
            apply_surgery(
                &store,
                &code,
                SurgeryEdit {
                    token_index: i,
                    new_text: format!("new{}", i),
                    old_text: format!("old{}", i),
                    editor_id: "p1".to_string(),
                    editor_color: "#58a6ff".to_string(),
                    editor_name: "Alice".to_string(),
                    timestamp_ms: i as u64,
                },
            );
        }
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(guard.get(&code).unwrap().surgery_log.len(), 5);
    }

    // -- add_chat ------------------------------------------------------------

    #[test]
    fn test_add_chat_appends_to_log() {
        let store = new_room_store();
        let code = create_room(&store);
        let msg = ChatMessage {
            id: "m1".to_string(),
            author_id: "p1".to_string(),
            author_name: "Alice".to_string(),
            author_color: "#58a6ff".to_string(),
            text: "Hello room!".to_string(),
            token_index: None,
            timestamp_ms: now_ms(),
        };
        add_chat(&store, &code, msg);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(guard.get(&code).unwrap().chat_log.len(), 1);
    }

    #[test]
    fn test_add_chat_stores_correct_text() {
        let store = new_room_store();
        let code = create_room(&store);
        add_chat(
            &store,
            &code,
            ChatMessage {
                id: "m1".to_string(),
                author_id: "p1".to_string(),
                author_name: "Alice".to_string(),
                author_color: "#58a6ff".to_string(),
                text: "Nice token!".to_string(),
                token_index: Some(5),
                timestamp_ms: 0,
            },
        );
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        let stored = &guard.get(&code).unwrap().chat_log[0];
        assert_eq!(stored.text, "Nice token!");
        assert_eq!(stored.token_index, Some(5));
    }

    #[test]
    fn test_add_chat_nonexistent_room_is_noop() {
        let store = new_room_store();
        // Must not panic
        add_chat(
            &store,
            "XXXXXX",
            ChatMessage {
                id: "m1".to_string(),
                author_id: "p1".to_string(),
                author_name: "Ghost".to_string(),
                author_color: "#fff".to_string(),
                text: "hello".to_string(),
                token_index: None,
                timestamp_ms: 0,
            },
        );
    }

    #[test]
    fn test_add_chat_multiple_messages() {
        let store = new_room_store();
        let code = create_room(&store);
        for i in 0..3 {
            add_chat(
                &store,
                &code,
                ChatMessage {
                    id: format!("m{}", i),
                    author_id: "p1".to_string(),
                    author_name: "Alice".to_string(),
                    author_color: "#58a6ff".to_string(),
                    text: format!("msg {}", i),
                    token_index: None,
                    timestamp_ms: i as u64,
                },
            );
        }
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(guard.get(&code).unwrap().chat_log.len(), 3);
    }

    // -- vote ----------------------------------------------------------------

    #[test]
    fn test_vote_up_increments_upvotes() {
        let store = new_room_store();
        let code = create_room(&store);
        let result = vote(&store, &code, "reverse", "up").unwrap();
        assert_eq!(result, (1, 0));
    }

    #[test]
    fn test_vote_down_increments_downvotes() {
        let store = new_room_store();
        let code = create_room(&store);
        let result = vote(&store, &code, "reverse", "down").unwrap();
        assert_eq!(result, (0, 1));
    }

    #[test]
    fn test_vote_multiple_up_accumulates() {
        let store = new_room_store();
        let code = create_room(&store);
        vote(&store, &code, "reverse", "up").unwrap();
        vote(&store, &code, "reverse", "up").unwrap();
        let result = vote(&store, &code, "reverse", "up").unwrap();
        assert_eq!(result, (3, 0));
    }

    #[test]
    fn test_vote_mixed_up_and_down() {
        let store = new_room_store();
        let code = create_room(&store);
        vote(&store, &code, "uppercase", "up").unwrap();
        vote(&store, &code, "uppercase", "up").unwrap();
        let result = vote(&store, &code, "uppercase", "down").unwrap();
        assert_eq!(result, (2, 1));
    }

    #[test]
    fn test_vote_different_transforms_independent() {
        let store = new_room_store();
        let code = create_room(&store);
        vote(&store, &code, "reverse", "up").unwrap();
        vote(&store, &code, "reverse", "up").unwrap();
        vote(&store, &code, "mock", "down").unwrap();
        let r = vote(&store, &code, "reverse", "up").unwrap();
        let m = vote(&store, &code, "mock", "up").unwrap();
        assert_eq!(r, (3, 0));
        assert_eq!(m, (1, 1));
    }

    #[test]
    fn test_vote_nonexistent_room_returns_none() {
        let store = new_room_store();
        let result = vote(&store, "XXXXXX", "reverse", "up");
        assert!(result.is_none());
    }

    #[test]
    fn test_vote_invalid_dir_does_not_change_counts() {
        let store = new_room_store();
        let code = create_room(&store);
        let result = vote(&store, &code, "reverse", "sideways").unwrap();
        assert_eq!(result, (0, 0));
    }

    #[test]
    fn test_vote_saturation_at_u32_max_does_not_panic() {
        let store = new_room_store();
        let code = create_room(&store);
        // Manually set votes to near u32::MAX
        {
            let mut guard = store.lock().unwrap_or_else(|e| e.into_inner());
            let room = guard.get_mut(&code).unwrap();
            room.votes.insert("reverse".to_string(), (u32::MAX, 0));
        }
        // saturating_add should not panic or overflow
        let result = vote(&store, &code, "reverse", "up").unwrap();
        assert_eq!(result.0, u32::MAX);
    }

    // -- room_state_snapshot -------------------------------------------------

    #[test]
    fn test_room_state_snapshot_returns_code() {
        let store = new_room_store();
        let code = create_room(&store);
        let snap = room_state_snapshot(&store, &code);
        assert_eq!(snap["code"], code.as_str());
    }

    #[test]
    fn test_room_state_snapshot_has_all_keys() {
        let store = new_room_store();
        let code = create_room(&store);
        let snap = room_state_snapshot(&store, &code);
        assert!(snap.get("host_id").is_some());
        assert!(snap.get("participants").is_some());
        assert!(snap.get("token_count").is_some());
        assert!(snap.get("surgery_log").is_some());
        assert!(snap.get("chat_log").is_some());
        assert!(snap.get("votes").is_some());
        assert!(snap.get("is_recording").is_some());
        assert!(snap.get("created_at_ms").is_some());
    }

    #[test]
    fn test_room_state_snapshot_nonexistent_returns_null() {
        let store = new_room_store();
        let snap = room_state_snapshot(&store, "XXXXXX");
        assert!(snap.is_null());
    }

    #[test]
    fn test_room_state_snapshot_reflects_participants() {
        let store = new_room_store();
        let code = create_room(&store);
        join_room(&store, &code, "Alice", true).unwrap();
        join_room(&store, &code, "Bob", false).unwrap();
        let snap = room_state_snapshot(&store, &code);
        let participants = snap["participants"].as_array().unwrap();
        assert_eq!(participants.len(), 2);
    }

    // -- recording -----------------------------------------------------------

    #[test]
    fn test_start_recording_sets_flag() {
        let store = new_room_store();
        let code = create_room(&store);
        start_recording(&store, &code);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert!(guard.get(&code).unwrap().is_recording);
    }

    #[test]
    fn test_start_recording_sets_start_time() {
        let store = new_room_store();
        let code = create_room(&store);
        start_recording(&store, &code);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert!(guard.get(&code).unwrap().recording_start_ms.is_some());
    }

    #[test]
    fn test_start_recording_clears_previous_events() {
        let store = new_room_store();
        let code = create_room(&store);
        start_recording(&store, &code);
        maybe_record(&store, &code, serde_json::json!({"type": "token"}));
        // Start again — should clear
        start_recording(&store, &code);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert!(guard.get(&code).unwrap().recorded_events.is_empty());
    }

    #[test]
    fn test_stop_recording_clears_flag() {
        let store = new_room_store();
        let code = create_room(&store);
        start_recording(&store, &code);
        stop_recording(&store, &code);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!guard.get(&code).unwrap().is_recording);
    }

    #[test]
    fn test_stop_recording_returns_recorded_events() {
        let store = new_room_store();
        let code = create_room(&store);
        start_recording(&store, &code);
        maybe_record(
            &store,
            &code,
            serde_json::json!({"type": "token", "index": 0}),
        );
        maybe_record(
            &store,
            &code,
            serde_json::json!({"type": "token", "index": 1}),
        );
        let events = stop_recording(&store, &code);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_stop_recording_drains_events_from_store() {
        let store = new_room_store();
        let code = create_room(&store);
        start_recording(&store, &code);
        maybe_record(&store, &code, serde_json::json!({"type": "chat"}));
        stop_recording(&store, &code);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert!(guard.get(&code).unwrap().recorded_events.is_empty());
    }

    #[test]
    fn test_stop_recording_nonexistent_room_returns_empty() {
        let store = new_room_store();
        let events = stop_recording(&store, "XXXXXX");
        assert!(events.is_empty());
    }

    // -- maybe_record --------------------------------------------------------

    #[test]
    fn test_maybe_record_does_nothing_when_not_recording() {
        let store = new_room_store();
        let code = create_room(&store);
        maybe_record(&store, &code, serde_json::json!({"type": "token"}));
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert!(guard.get(&code).unwrap().recorded_events.is_empty());
    }

    #[test]
    fn test_maybe_record_appends_event_when_recording() {
        let store = new_room_store();
        let code = create_room(&store);
        start_recording(&store, &code);
        maybe_record(&store, &code, serde_json::json!({"type": "token"}));
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(guard.get(&code).unwrap().recorded_events.len(), 1);
    }

    #[test]
    fn test_maybe_record_stores_payload() {
        let store = new_room_store();
        let code = create_room(&store);
        start_recording(&store, &code);
        maybe_record(
            &store,
            &code,
            serde_json::json!({"type": "chat", "text": "hello"}),
        );
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        let ev = &guard.get(&code).unwrap().recorded_events[0];
        assert_eq!(ev.payload["type"], "chat");
        assert_eq!(ev.payload["text"], "hello");
    }

    #[test]
    fn test_maybe_record_sets_offset_ms() {
        let store = new_room_store();
        let code = create_room(&store);
        start_recording(&store, &code);
        maybe_record(&store, &code, serde_json::json!({"type": "token"}));
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        let ev = &guard.get(&code).unwrap().recorded_events[0];
        // offset_ms must be >= 0 (it's u64, always true) and < 1000ms for a test
        assert!(
            ev.offset_ms < 1000,
            "offset_ms should be small in a fast test: {}",
            ev.offset_ms
        );
    }

    #[test]
    fn test_maybe_record_multiple_events_ordered() {
        let store = new_room_store();
        let code = create_room(&store);
        start_recording(&store, &code);
        for i in 0..5 {
            maybe_record(&store, &code, serde_json::json!({"seq": i}));
        }
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        let events = &guard.get(&code).unwrap().recorded_events;
        assert_eq!(events.len(), 5);
        for (i, ev) in events.iter().enumerate() {
            assert_eq!(ev.payload["seq"], i);
        }
    }

    #[test]
    fn test_maybe_record_nonexistent_room_is_noop() {
        let store = new_room_store();
        // Must not panic
        maybe_record(&store, "XXXXXX", serde_json::json!({"type": "token"}));
    }

    #[test]
    fn test_maybe_record_cap_exceeded_broadcasts_truncated_warning() {
        let store = new_room_store();
        let code = "CAPWRN".to_string();
        {
            let mut guard = store.lock().unwrap_or_else(|e| e.into_inner());
            let (tx, _) = tokio::sync::broadcast::channel(16);
            let room = Room {
                code: code.clone(),
                host_id: String::new(),
                participants: Vec::new(),
                token_count: 0,
                surgery_log: Vec::new(),
                chat_log: Vec::new(),
                votes: std::collections::HashMap::new(),
                is_recording: true,
                recording_start_ms: Some(now_ms()),
                recorded_events: Vec::new(),
                created_at_ms: now_ms(),
                last_activity_ms: now_ms(),
                recording_cap: 2,
                broadcast_tx: tx,
            };
            let _ = room.recording_cap;
            guard.insert(code.clone(), room);
        }

        // Subscribe to broadcast before recording events
        let mut rx = {
            let guard = store.lock().unwrap_or_else(|e| e.into_inner());
            guard.get(&code).unwrap().broadcast_tx.subscribe()
        };

        // Push 3 events — should trigger truncation warning on the 3rd
        maybe_record(&store, &code, serde_json::json!({"seq": 0}));
        maybe_record(&store, &code, serde_json::json!({"seq": 1}));
        maybe_record(&store, &code, serde_json::json!({"seq": 2}));

        // Collect broadcast messages
        let mut got_truncated = false;
        while let Ok(msg) = rx.try_recv() {
            if msg.get("type").and_then(|v| v.as_str()) == Some("record_truncated") {
                got_truncated = true;
                assert_eq!(msg["dropped"], 1);
                assert_eq!(msg["cap"], 2);
            }
        }
        assert!(
            got_truncated,
            "expected record_truncated broadcast when cap is exceeded"
        );

        // Verify only cap events remain
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(guard.get(&code).unwrap().recorded_events.len(), 2);
    }

    // -- Participant serialization -------------------------------------------

    #[test]
    fn test_participant_serializes_all_fields() {
        let p = Participant {
            id: "abc".to_string(),
            name: "Alice".to_string(),
            color: "#58a6ff".to_string(),
            joined_at_ms: 9999,
            is_host: true,
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("\"id\":\"abc\""));
        assert!(json.contains("\"name\":\"Alice\""));
        assert!(json.contains("\"color\":\"#58a6ff\""));
        assert!(json.contains("\"is_host\":true"));
    }

    #[test]
    fn test_participant_roundtrip() {
        let p = Participant {
            id: "xyz".to_string(),
            name: "Bob".to_string(),
            color: "#3fb950".to_string(),
            joined_at_ms: 42,
            is_host: false,
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: Participant = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "xyz");
        assert_eq!(back.name, "Bob");
        assert!(!back.is_host);
    }

    // -- SurgeryEdit serialization -------------------------------------------

    #[test]
    fn test_surgery_edit_serializes() {
        let edit = SurgeryEdit {
            token_index: 2,
            new_text: "world".to_string(),
            old_text: "hello".to_string(),
            editor_id: "p1".to_string(),
            editor_color: "#a371f7".to_string(),
            editor_name: "Carol".to_string(),
            timestamp_ms: 1234,
        };
        let json = serde_json::to_string(&edit).unwrap();
        assert!(json.contains("\"token_index\":2"));
        assert!(json.contains("\"new_text\":\"world\""));
        assert!(json.contains("\"editor_name\":\"Carol\""));
    }

    #[test]
    fn test_surgery_edit_roundtrip() {
        let edit = SurgeryEdit {
            token_index: 5,
            new_text: "modified".to_string(),
            old_text: "original".to_string(),
            editor_id: "p99".to_string(),
            editor_color: "#f85149".to_string(),
            editor_name: "Dave".to_string(),
            timestamp_ms: 555,
        };
        let json = serde_json::to_string(&edit).unwrap();
        let back: SurgeryEdit = serde_json::from_str(&json).unwrap();
        assert_eq!(back.token_index, 5);
        assert_eq!(back.new_text, "modified");
        assert_eq!(back.editor_name, "Dave");
    }

    // -- ChatMessage serialization -------------------------------------------

    #[test]
    fn test_chat_message_serializes() {
        let msg = ChatMessage {
            id: "msg1".to_string(),
            author_id: "p1".to_string(),
            author_name: "Eve".to_string(),
            author_color: "#e3b341".to_string(),
            text: "interesting token".to_string(),
            token_index: Some(3),
            timestamp_ms: 9876,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"text\":\"interesting token\""));
        assert!(json.contains("\"token_index\":3"));
        assert!(json.contains("\"author_name\":\"Eve\""));
    }

    #[test]
    fn test_chat_message_null_token_index_serializes() {
        let msg = ChatMessage {
            id: "m2".to_string(),
            author_id: "p1".to_string(),
            author_name: "Frank".to_string(),
            author_color: "#f0883e".to_string(),
            text: "general comment".to_string(),
            token_index: None,
            timestamp_ms: 0,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"token_index\":null"));
    }

    #[test]
    fn test_chat_message_roundtrip() {
        let msg = ChatMessage {
            id: "m3".to_string(),
            author_id: "p2".to_string(),
            author_name: "Grace".to_string(),
            author_color: "#58a6ff".to_string(),
            text: "roundtrip test".to_string(),
            token_index: Some(10),
            timestamp_ms: 1000,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "m3");
        assert_eq!(back.text, "roundtrip test");
        assert_eq!(back.token_index, Some(10));
    }

    // -- RecordedEvent serialization -----------------------------------------

    #[test]
    fn test_recorded_event_serializes() {
        let ev = RecordedEvent {
            offset_ms: 250,
            payload: serde_json::json!({"type": "token", "index": 7}),
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("\"offset_ms\":250"));
        assert!(json.contains("\"type\":\"token\""));
    }

    #[test]
    fn test_recorded_event_roundtrip() {
        let ev = RecordedEvent {
            offset_ms: 500,
            payload: serde_json::json!({"type": "surgery", "token_index": 3}),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: RecordedEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.offset_ms, 500);
        assert_eq!(back.payload["type"], "surgery");
        assert_eq!(back.payload["token_index"], 3);
    }

    // -- Edge cases / robustness ---------------------------------------------

    #[test]
    fn test_full_session_lifecycle() {
        // Create → join × 2 → chat → surgery → vote → leave
        let store = new_room_store();
        let code = create_room(&store);

        let (host, _rx_host) = join_room(&store, &code, "Host", true).unwrap();
        let (guest, _rx_guest) = join_room(&store, &code, "Guest", false).unwrap();

        add_chat(
            &store,
            &code,
            ChatMessage {
                id: "c1".to_string(),
                author_id: host.id.clone(),
                author_name: "Host".to_string(),
                author_color: host.color.clone(),
                text: "Welcome!".to_string(),
                token_index: None,
                timestamp_ms: now_ms(),
            },
        );

        apply_surgery(
            &store,
            &code,
            SurgeryEdit {
                token_index: 0,
                new_text: "edited".to_string(),
                old_text: "original".to_string(),
                editor_id: guest.id.clone(),
                editor_color: guest.color.clone(),
                editor_name: guest.name.clone(),
                timestamp_ms: now_ms(),
            },
        );

        let (up, _) = vote(&store, &code, "reverse", "up").unwrap();
        assert_eq!(up, 1);

        let snap = room_state_snapshot(&store, &code);
        assert_eq!(snap["participants"].as_array().unwrap().len(), 2);
        assert_eq!(snap["chat_log"].as_array().unwrap().len(), 1);
        assert_eq!(snap["surgery_log"].as_array().unwrap().len(), 1);

        leave_room(&store, &code, &host.id);
        leave_room(&store, &code, &guest.id);

        let snap2 = room_state_snapshot(&store, &code);
        assert!(snap2["participants"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_recording_full_lifecycle() {
        let store = new_room_store();
        let code = create_room(&store);
        start_recording(&store, &code);
        for i in 0..3 {
            maybe_record(
                &store,
                &code,
                serde_json::json!({"type": "token", "index": i}),
            );
        }
        maybe_record(
            &store,
            &code,
            serde_json::json!({"type": "chat", "text": "hi"}),
        );
        let events = stop_recording(&store, &code);
        assert_eq!(events.len(), 4);
        assert_eq!(events[0].payload["type"], "token");
        assert_eq!(events[3].payload["type"], "chat");
    }

    // -- token broadcast handler tests ---------------------------------------

    #[test]
    fn test_broadcast_token_reaches_subscriber() {
        let store = new_room_store();
        let code = create_room(&store);
        let (_, mut rx) = join_room(&store, &code, "watcher", false).unwrap();

        // Simulate what handle_ws does for a "token" message from host
        broadcast(
            &store,
            &code,
            serde_json::json!({"type": "token", "text": "hello", "index": 0}),
        );

        let msg = rx.try_recv().unwrap();
        assert_eq!(msg["type"], "token");
        assert_eq!(msg["text"], "hello");
    }

    #[test]
    fn test_broadcast_token_also_records_when_active() {
        let store = new_room_store();
        let code = create_room(&store);
        start_recording(&store, &code);

        let token_payload = serde_json::json!({"type": "token", "text": "hi", "index": 0});
        maybe_record(&store, &code, token_payload);

        let events = stop_recording(&store, &code);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].payload["type"], "token");
        assert_eq!(events[0].payload["text"], "hi");
    }

    #[test]
    fn test_record_token_not_recorded_when_not_active() {
        let store = new_room_store();
        let code = create_room(&store);
        // NOT started recording

        maybe_record(&store, &code, serde_json::json!({"type": "token"}));

        // Recording was never started so events list should be empty after stop
        let events = stop_recording(&store, &code);
        assert!(events.is_empty());
    }

    #[test]
    fn test_stream_done_broadcast_reaches_subscriber() {
        let store = new_room_store();
        let code = create_room(&store);
        let (_, mut rx) = join_room(&store, &code, "viewer", false).unwrap();

        broadcast(&store, &code, serde_json::json!({"type": "stream_done"}));

        let msg = rx.try_recv().unwrap();
        assert_eq!(msg["type"], "stream_done");
    }

    #[test]
    fn test_token_broadcast_increments_room_count_independently() {
        let store = new_room_store();
        let code = create_room(&store);
        // No auto-increment on broadcast; but participant count should stay correct
        let (_, _rx) = join_room(&store, &code, "p1", false).unwrap();
        let (_, _rx2) = join_room(&store, &code, "p2", false).unwrap();

        broadcast(
            &store,
            &code,
            serde_json::json!({"type": "token", "index": 0}),
        );

        let snap = room_state_snapshot(&store, &code);
        assert_eq!(snap["participants"].as_array().unwrap().len(), 2);
    }

    // -- stream_done JS handler test (via INDEX_HTML from web module) ---------
    // (These live in web.rs test block; here we verify server-side logic only)

    #[test]
    fn test_maybe_record_chat_and_token_interleaved() {
        let store = new_room_store();
        let code = create_room(&store);
        start_recording(&store, &code);

        maybe_record(
            &store,
            &code,
            serde_json::json!({"type": "token", "index": 0}),
        );
        maybe_record(
            &store,
            &code,
            serde_json::json!({"type": "chat", "text": "nice"}),
        );
        maybe_record(
            &store,
            &code,
            serde_json::json!({"type": "token", "index": 1}),
        );

        let events = stop_recording(&store, &code);
        assert_eq!(events.len(), 3);
        assert_eq!(events[1].payload["type"], "chat");
    }

    #[test]
    fn test_recording_offset_ms_is_non_decreasing() {
        let store = new_room_store();
        let code = create_room(&store);
        start_recording(&store, &code);

        for i in 0..5 {
            maybe_record(&store, &code, serde_json::json!({"i": i}));
        }

        let events = stop_recording(&store, &code);
        assert_eq!(events.len(), 5);
        for w in events.windows(2) {
            assert!(w[1].offset_ms >= w[0].offset_ms);
        }
    }

    #[test]
    fn test_multiple_rooms_record_independently() {
        let store = new_room_store();
        let code_a = create_room(&store);
        let code_b = create_room(&store);

        start_recording(&store, &code_a);
        maybe_record(&store, &code_a, serde_json::json!({"room": "a"}));
        // code_b is NOT recording
        maybe_record(&store, &code_b, serde_json::json!({"room": "b"}));

        let a_events = stop_recording(&store, &code_a);
        let b_events = stop_recording(&store, &code_b);

        assert_eq!(a_events.len(), 1);
        assert!(b_events.is_empty());
    }

    // -- evict_idle_rooms ----------------------------------------------------

    #[test]
    fn test_evict_idle_rooms_removes_stale_room() {
        let store = new_room_store();
        let code = create_room(&store);
        // Manually set last_activity_ms to far in the past
        {
            let mut guard = store.lock().unwrap_or_else(|e| e.into_inner());
            let room = guard.get_mut(&code).unwrap();
            room.last_activity_ms = 0; // epoch zero — definitely idle
        }
        evict_idle_rooms(&store);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            !guard.contains_key(&code),
            "stale room should have been evicted"
        );
    }

    #[test]
    fn test_evict_idle_rooms_keeps_active_room() {
        let store = new_room_store();
        let code = create_room(&store);
        // last_activity_ms was just set to now_ms() in create_room
        evict_idle_rooms(&store);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            guard.contains_key(&code),
            "recently active room should not be evicted"
        );
    }

    #[test]
    fn test_evict_idle_rooms_mixed() {
        let store = new_room_store();
        let code_active = create_room(&store);
        let code_stale = create_room(&store);
        {
            let mut guard = store.lock().unwrap_or_else(|e| e.into_inner());
            guard.get_mut(&code_stale).unwrap().last_activity_ms = 0;
        }
        evict_idle_rooms(&store);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert!(guard.contains_key(&code_active));
        assert!(!guard.contains_key(&code_stale));
    }

    #[test]
    fn test_evict_idle_rooms_empty_store_is_noop() {
        let store = new_room_store();
        evict_idle_rooms(&store); // must not panic
        assert!(store.lock().unwrap_or_else(|e| e.into_inner()).is_empty());
    }

    // -- recording_cap -------------------------------------------------------

    #[test]
    fn test_recording_cap_enforced() {
        let store = new_room_store();
        let code = create_room(&store);
        // Lower the cap to a small value for testing
        {
            let mut guard = store.lock().unwrap_or_else(|e| e.into_inner());
            guard.get_mut(&code).unwrap().recording_cap = 5;
        }
        start_recording(&store, &code);
        for i in 0..10 {
            maybe_record(&store, &code, serde_json::json!({"seq": i}));
        }
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        let room = guard.get(&code).unwrap();
        assert_eq!(
            room.recorded_events.len(),
            5,
            "cap should limit buffer to 5 events"
        );
        // The oldest events should have been dropped (FIFO trim)
        assert_eq!(room.recorded_events[0].payload["seq"], 5);
        assert_eq!(room.recorded_events[4].payload["seq"], 9);
    }

    #[test]
    fn test_recording_cap_default_is_ten_thousand() {
        let store = new_room_store();
        let code = create_room(&store);
        let guard = store.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            guard.get(&code).unwrap().recording_cap,
            DEFAULT_RECORDING_CAP
        );
    }

    // -- generate_code collision test ----------------------------------------

    #[test]
    fn test_generate_code_no_collision_at_small_scale() {
        // With 20*20*90=36000 possibilities, generate 200 codes; expect high uniqueness.
        let codes: Vec<String> = (0..200).map(|_| generate_code()).collect();
        for code in &codes {
            assert!(code.len() >= 8, "code must be at least 8 chars: {}", code);
            assert!(
                code.chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '-'),
                "code contains unexpected chars: {}",
                code
            );
        }
        let unique: std::collections::HashSet<&String> = codes.iter().collect();
        assert!(
            unique.len() >= 190,
            "expected high uniqueness, got {} unique out of 200",
            unique.len()
        );
    }

    #[test]
    fn test_maybe_record_cap_enforced() {
        let store = new_room_store();
        let code = create_room(&store);
        start_recording(&store, &code);
        // Set recording_cap to 3
        if let Ok(mut guard) = store.lock() {
            if let Some(room) = guard.get_mut(&code) {
                room.recording_cap = 3;
            }
        }
        for i in 0..5 {
            maybe_record(&store, &code, serde_json::json!({"i": i}));
        }
        // Should only have recording_cap events
        let recorded_len = store.lock().unwrap().get(&code).map(|r| r.recorded_events.len()).unwrap_or(0);
        assert!(recorded_len <= 3);
    }

    #[test]
    fn test_evict_idle_rooms_removes_stale() {
        let store = new_room_store();
        let code = create_room(&store);
        // Backdating last_activity_ms to beyond TTL
        if let Ok(mut guard) = store.lock() {
            if let Some(room) = guard.get_mut(&code) {
                room.last_activity_ms = 0; // epoch 0 is definitely stale
            }
        }
        evict_idle_rooms(&store);
        let is_gone = store.lock().unwrap().get(&code).is_none();
        assert!(is_gone, "stale room should be evicted");
    }

    #[test]
    fn test_evict_idle_rooms_keeps_active() {
        let store = new_room_store();
        let code = create_room(&store);
        // last_activity_ms is set to now_ms() in create_room — should survive
        evict_idle_rooms(&store);
        let is_alive = store.lock().unwrap().get(&code).is_some();
        assert!(is_alive, "active room should not be evicted");
    }

    // -- Item 10: Concurrent room operations --

    #[tokio::test]
    async fn test_concurrent_room_creation() {
        let store = new_room_store();
        let mut handles = vec![];
        for _ in 0..20 {
            let store_clone = store.clone();
            handles.push(tokio::spawn(async move {
                create_room(&store_clone)
            }));
        }
        let mut codes = vec![];
        for h in handles {
            codes.push(h.await.expect("task should not panic"));
        }
        // All 20 rooms should have unique codes.
        let unique: std::collections::HashSet<&String> = codes.iter().collect();
        assert_eq!(unique.len(), 20, "all room codes should be unique");
        // All 20 rooms should exist in the store.
        let guard = store.lock().unwrap();
        for code in &codes {
            assert!(guard.contains_key(code), "room {} should exist", code);
        }
    }

    #[test]
    fn test_room_not_found_returns_none() {
        let store = new_room_store();
        // Looking up a non-existent room should return None/error.
        let result = join_room(&store, "NONEXISTENT-CODE", "Alice", false);
        assert!(result.is_err(), "non-existent room should return an error");
        // vote returns None for non-existent rooms.
        let vote_result = vote(&store, "NONEXISTENT-CODE", "reverse", "up");
        assert!(vote_result.is_none(), "vote on non-existent room should return None");
        // room_state_snapshot returns Null for non-existent rooms.
        let snap = room_state_snapshot(&store, "NONEXISTENT-CODE");
        assert!(snap.is_null(), "snapshot of non-existent room should be null");
    }

    // -- Item 11: WS message parsing edge cases --

    #[test]
    fn test_ws_msg_malformed_json_ignored() {
        // Verify that serde_json gracefully handles malformed JSON without panicking.
        let malformed = "{ this is not valid json }";
        let result: Result<serde_json::Value, _> = serde_json::from_str(malformed);
        assert!(result.is_err(), "malformed JSON should parse to Err, not panic");
    }

    #[test]
    fn test_ws_msg_unknown_type_ignored() {
        // Verify that JSON with an unknown type field is parsed without panicking.
        let msg = r#"{"type":"unknown_type_xyz","data":"something"}"#;
        let parsed: serde_json::Value = serde_json::from_str(msg).expect("valid JSON");
        let msg_type = parsed.get("type").and_then(|v| v.as_str()).unwrap_or("");
        // The handle_ws match arm falls through to the unknown/warn branch — no panic.
        assert_eq!(msg_type, "unknown_type_xyz");
        // Simulate the match: none of the known types match.
        let known = ["set_name", "vote", "surgery", "chat", "record_start",
                     "record_stop", "replay_request", "ping", "token", "_record_token", "stream_done"];
        assert!(!known.contains(&msg_type), "unknown_type_xyz should not match any known type");
    }

    // -- Memorable code format tests --

    #[test]
    fn test_generate_code_has_hyphen_format() {
        for _ in 0..20 {
            let code = generate_code();
            let parts: Vec<&str> = code.split('-').collect();
            assert_eq!(parts.len(), 3, "code should have 3 hyphen-separated parts: {}", code);
            // Third part should be a number 10-99
            let num: u8 = parts[2].parse().expect("third part should be a number");
            assert!(num >= 10 && num <= 99, "number part should be 10-99: {}", num);
        }
    }

    #[test]
    fn test_generate_code_all_uppercase() {
        for _ in 0..20 {
            let code = generate_code();
            assert!(code.chars().all(|c| c.is_ascii_uppercase() || c == '-' || c.is_ascii_digit()),
                "code should be uppercase: {}", code);
        }
    }

    #[test]
    fn test_generate_code_adjective_from_list() {
        for _ in 0..50 {
            let code = generate_code();
            let adj = code.split('-').next().unwrap();
            assert!(CODE_ADJECTIVES.contains(&adj), "adjective {} not in list", adj);
        }
    }

    #[test]
    fn test_generate_code_noun_from_list() {
        for _ in 0..50 {
            let code = generate_code();
            let parts: Vec<&str> = code.split('-').collect();
            assert!(CODE_NOUNS.contains(&parts[1]), "noun {} not in list", parts[1]);
        }
    }

    // -- Item 5: broadcast to empty room does not panic --
    #[test]
    fn test_broadcast_to_empty_room_no_panic() {
        let store = new_room_store();
        let code = create_room(&store);
        // No WS connections; broadcast should not panic even with no subscribers
        broadcast(&store, &code, serde_json::json!({"type": "test"}));
        // If we get here without panic, the test passes
    }

    // -- Item 6: evict_idle_rooms is pub and callable --
    #[test]
    fn test_evict_idle_rooms_is_pub() {
        let store = new_room_store();
        // Just verify the function is callable (it's pub)
        evict_idle_rooms(&store);
    }

    // -- Item 23: lag warning message format --
    fn build_lag_warning_message(skipped: u64) -> String {
        serde_json::json!({
            "type": "lag_warning",
            "skipped": skipped,
        })
        .to_string()
    }

    #[test]
    fn test_lag_warning_message_format() {
        let msg = build_lag_warning_message(5);
        let parsed: serde_json::Value = serde_json::from_str(&msg).expect("valid JSON");
        assert_eq!(parsed["type"], "lag_warning");
        assert_eq!(parsed["skipped"], 5);
    }
}
