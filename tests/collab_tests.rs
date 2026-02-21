//! Tests for the collab module — room management, participant tracking,
//! surgery edits, chat, voting, recording, and code generation.

use every_other_token::collab::*;

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

#[test]
fn test_generate_code_length() {
    assert_eq!(generate_code().len(), 6);
}

#[test]
fn test_generate_code_uppercase_alphanumeric() {
    let code = generate_code();
    assert!(code.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()));
}

#[test]
fn test_generate_code_is_unique() {
    let codes: std::collections::HashSet<String> = (0..100).map(|_| generate_code()).collect();
    assert!(codes.len() > 90);
}

#[test]
fn test_generate_code_no_lowercase() {
    for _ in 0..50 {
        let code = generate_code();
        assert!(!code.chars().any(|c| c.is_ascii_lowercase()));
    }
}

#[test]
fn test_generate_code_no_special_chars() {
    for _ in 0..50 {
        let code = generate_code();
        assert!(code.chars().all(|c| c.is_alphanumeric()));
    }
}

// ---------------------------------------------------------------------------
// now_ms
// ---------------------------------------------------------------------------

#[test]
fn test_now_ms_is_reasonable() {
    let ms = now_ms();
    // After 2023-11-01
    assert!(ms > 1_700_000_000_000);
}

#[test]
fn test_now_ms_increases() {
    let a = now_ms();
    std::thread::sleep(std::time::Duration::from_millis(2));
    let b = now_ms();
    assert!(b >= a);
}

// ---------------------------------------------------------------------------
// Room creation
// ---------------------------------------------------------------------------

#[test]
fn test_create_room_returns_6_char_code() {
    let store = new_room_store();
    let code = create_room(&store);
    assert_eq!(code.len(), 6);
}

#[test]
fn test_create_room_is_stored() {
    let store = new_room_store();
    let code = create_room(&store);
    let guard = store.lock().unwrap();
    assert!(guard.contains_key(&code));
}

#[test]
fn test_create_multiple_rooms() {
    let store = new_room_store();
    let c1 = create_room(&store);
    let c2 = create_room(&store);
    let guard = store.lock().unwrap();
    assert!(guard.contains_key(&c1));
    assert!(guard.contains_key(&c2));
    // Codes are very unlikely to collide; if they do this test may flake
    // but the probability is 1/(36^6) ≈ 10^-9 per pair.
}

#[test]
fn test_create_room_initial_participant_count_zero() {
    let store = new_room_store();
    let code = create_room(&store);
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert!(room.participants.is_empty());
}

#[test]
fn test_create_room_host_id_empty_initially() {
    let store = new_room_store();
    let code = create_room(&store);
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert!(room.host_id.is_empty());
}

#[test]
fn test_create_room_not_recording_initially() {
    let store = new_room_store();
    let code = create_room(&store);
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert!(!room.is_recording);
}

// ---------------------------------------------------------------------------
// Join room
// ---------------------------------------------------------------------------

#[test]
fn test_join_room_success() {
    let store = new_room_store();
    let code = create_room(&store);
    let result = join_room(&store, &code, "Alice", true);
    assert!(result.is_ok());
}

#[test]
fn test_join_room_wrong_code_returns_error() {
    let store = new_room_store();
    let result = join_room(&store, "XXXXXX", "Alice", false);
    assert!(result.is_err());
}

#[test]
fn test_join_room_assigns_participant_name() {
    let store = new_room_store();
    let code = create_room(&store);
    let (p, _rx) = join_room(&store, &code, "Alice", true).unwrap();
    assert_eq!(p.name, "Alice");
}

#[test]
fn test_join_room_host_flag_set() {
    let store = new_room_store();
    let code = create_room(&store);
    let (p, _rx) = join_room(&store, &code, "Host", true).unwrap();
    assert!(p.is_host);
}

#[test]
fn test_join_room_guest_flag_not_host() {
    let store = new_room_store();
    let code = create_room(&store);
    let _ = join_room(&store, &code, "Host", true).unwrap();
    let (p2, _rx2) = join_room(&store, &code, "Guest", false).unwrap();
    assert!(!p2.is_host);
}

#[test]
fn test_join_room_assigns_color() {
    let store = new_room_store();
    let code = create_room(&store);
    let (p, _rx) = join_room(&store, &code, "Alice", true).unwrap();
    assert!(!p.color.is_empty());
    assert!(PARTICIPANT_COLORS.contains(&p.color.as_str()));
}

#[test]
fn test_join_room_multiple_participants_get_different_colors() {
    let store = new_room_store();
    let code = create_room(&store);
    let (p1, _r1) = join_room(&store, &code, "Alice", true).unwrap();
    let (p2, _r2) = join_room(&store, &code, "Bob", false).unwrap();
    assert_ne!(p1.color, p2.color);
}

#[test]
fn test_join_room_color_wraps_after_all_colors_used() {
    let store = new_room_store();
    let code = create_room(&store);
    let n = PARTICIPANT_COLORS.len();
    for i in 0..n {
        let _ = join_room(&store, &code, &format!("P{}", i), false).unwrap();
    }
    // Next participant wraps to color 0
    let (p_wrap, _rx) = join_room(&store, &code, "Wrap", false).unwrap();
    assert_eq!(p_wrap.color, PARTICIPANT_COLORS[0]);
}

#[test]
fn test_join_room_participant_stored() {
    let store = new_room_store();
    let code = create_room(&store);
    let (p, _rx) = join_room(&store, &code, "Alice", true).unwrap();
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert!(room.participants.iter().any(|x| x.id == p.id));
}

#[test]
fn test_join_room_sets_host_id() {
    let store = new_room_store();
    let code = create_room(&store);
    let (p, _rx) = join_room(&store, &code, "Host", true).unwrap();
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert_eq!(room.host_id, p.id);
}

#[test]
fn test_join_room_participant_has_uuid() {
    let store = new_room_store();
    let code = create_room(&store);
    let (p, _rx) = join_room(&store, &code, "Alice", true).unwrap();
    // UUID v4 is 36 chars with dashes
    assert_eq!(p.id.len(), 36);
}

// ---------------------------------------------------------------------------
// Leave room
// ---------------------------------------------------------------------------

#[test]
fn test_leave_room_removes_participant() {
    let store = new_room_store();
    let code = create_room(&store);
    let (p, _rx) = join_room(&store, &code, "Alice", true).unwrap();
    leave_room(&store, &code, &p.id);
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert!(!room.participants.iter().any(|x| x.id == p.id));
}

#[test]
fn test_leave_room_wrong_code_returns_none() {
    let store = new_room_store();
    let result = leave_room(&store, "XXXXXX", "any-id");
    assert!(result.is_none());
}

#[test]
fn test_leave_room_nonexistent_participant_ok() {
    let store = new_room_store();
    let code = create_room(&store);
    // Should not panic or error even if the participant doesn't exist.
    let tx = leave_room(&store, &code, "nonexistent-id");
    assert!(tx.is_some()); // room exists so sender is returned
}

#[test]
fn test_leave_room_returns_broadcast_tx() {
    let store = new_room_store();
    let code = create_room(&store);
    let (p, _rx) = join_room(&store, &code, "Alice", true).unwrap();
    let tx = leave_room(&store, &code, &p.id);
    assert!(tx.is_some());
}

// ---------------------------------------------------------------------------
// Surgery edits
// ---------------------------------------------------------------------------

fn make_edit(index: usize, new_text: &str, old_text: &str) -> SurgeryEdit {
    SurgeryEdit {
        token_index: index,
        new_text: new_text.to_string(),
        old_text: old_text.to_string(),
        editor_id: "editor-1".to_string(),
        editor_color: "#58a6ff".to_string(),
        editor_name: "Alice".to_string(),
        timestamp_ms: now_ms(),
    }
}

#[test]
fn test_apply_surgery_adds_to_log() {
    let store = new_room_store();
    let code = create_room(&store);
    apply_surgery(&store, &code, make_edit(0, "world", "World"));
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert_eq!(room.surgery_log.len(), 1);
    assert_eq!(room.surgery_log[0].new_text, "world");
}

#[test]
fn test_apply_surgery_multiple_edits() {
    let store = new_room_store();
    let code = create_room(&store);
    apply_surgery(&store, &code, make_edit(0, "hello", "Hello"));
    apply_surgery(&store, &code, make_edit(1, "world", "World"));
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert_eq!(room.surgery_log.len(), 2);
}

#[test]
fn test_apply_surgery_preserves_old_text() {
    let store = new_room_store();
    let code = create_room(&store);
    apply_surgery(&store, &code, make_edit(3, "new", "old"));
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert_eq!(room.surgery_log[0].old_text, "old");
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

fn make_chat(text: &str, token_index: Option<usize>) -> ChatMessage {
    ChatMessage {
        id: uuid::Uuid::new_v4().to_string(),
        author_id: "author-1".to_string(),
        author_name: "Alice".to_string(),
        author_color: "#58a6ff".to_string(),
        text: text.to_string(),
        token_index,
        timestamp_ms: now_ms(),
    }
}

#[test]
fn test_add_chat_message_stored() {
    let store = new_room_store();
    let code = create_room(&store);
    add_chat(&store, &code, make_chat("hello", None));
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert_eq!(room.chat_log.len(), 1);
    assert_eq!(room.chat_log[0].text, "hello");
}

#[test]
fn test_chat_with_token_index() {
    let store = new_room_store();
    let code = create_room(&store);
    add_chat(&store, &code, make_chat("interesting token", Some(5)));
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert_eq!(room.chat_log[0].token_index, Some(5));
}

#[test]
fn test_chat_without_token_index() {
    let store = new_room_store();
    let code = create_room(&store);
    add_chat(&store, &code, make_chat("general comment", None));
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert!(room.chat_log[0].token_index.is_none());
}

#[test]
fn test_add_multiple_chat_messages() {
    let store = new_room_store();
    let code = create_room(&store);
    add_chat(&store, &code, make_chat("one", None));
    add_chat(&store, &code, make_chat("two", None));
    add_chat(&store, &code, make_chat("three", None));
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert_eq!(room.chat_log.len(), 3);
}

// ---------------------------------------------------------------------------
// Voting
// ---------------------------------------------------------------------------

#[test]
fn test_vote_up_increments_up_count() {
    let store = new_room_store();
    let code = create_room(&store);
    let result = vote(&store, &code, "reverse", "up");
    assert_eq!(result, Some((1, 0)));
}

#[test]
fn test_vote_down_increments_down_count() {
    let store = new_room_store();
    let code = create_room(&store);
    let result = vote(&store, &code, "reverse", "down");
    assert_eq!(result, Some((0, 1)));
}

#[test]
fn test_vote_multiple_ups() {
    let store = new_room_store();
    let code = create_room(&store);
    vote(&store, &code, "reverse", "up");
    vote(&store, &code, "reverse", "up");
    let result = vote(&store, &code, "reverse", "up");
    assert_eq!(result, Some((3, 0)));
}

#[test]
fn test_vote_mixed_up_and_down() {
    let store = new_room_store();
    let code = create_room(&store);
    vote(&store, &code, "reverse", "up");
    vote(&store, &code, "reverse", "up");
    vote(&store, &code, "reverse", "down");
    let result = vote(&store, &code, "reverse", "up");
    assert_eq!(result, Some((3, 1)));
}

#[test]
fn test_vote_multiple_transforms() {
    let store = new_room_store();
    let code = create_room(&store);
    vote(&store, &code, "reverse", "up");
    vote(&store, &code, "uppercase", "up");
    vote(&store, &code, "uppercase", "up");
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert_eq!(room.votes.get("reverse"), Some(&(1, 0)));
    assert_eq!(room.votes.get("uppercase"), Some(&(2, 0)));
}

#[test]
fn test_vote_on_nonexistent_room_returns_none() {
    let store = new_room_store();
    let result = vote(&store, "XXXXXX", "reverse", "up");
    assert!(result.is_none());
}

#[test]
fn test_vote_unknown_dir_no_change() {
    let store = new_room_store();
    let code = create_room(&store);
    let result = vote(&store, &code, "reverse", "sideways");
    assert_eq!(result, Some((0, 0)));
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

#[test]
fn test_start_recording_sets_flag() {
    let store = new_room_store();
    let code = create_room(&store);
    start_recording(&store, &code);
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert!(room.is_recording);
}

#[test]
fn test_start_recording_sets_start_ms() {
    let store = new_room_store();
    let code = create_room(&store);
    start_recording(&store, &code);
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert!(room.recording_start_ms.is_some());
}

#[test]
fn test_stop_recording_clears_flag() {
    let store = new_room_store();
    let code = create_room(&store);
    start_recording(&store, &code);
    stop_recording(&store, &code);
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert!(!room.is_recording);
}

#[test]
fn test_stop_recording_clears_start_ms() {
    let store = new_room_store();
    let code = create_room(&store);
    start_recording(&store, &code);
    stop_recording(&store, &code);
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert!(room.recording_start_ms.is_none());
}

#[test]
fn test_maybe_record_appends_when_recording() {
    let store = new_room_store();
    let code = create_room(&store);
    start_recording(&store, &code);
    maybe_record(&store, &code, serde_json::json!({"type": "token", "text": "hello"}));
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert_eq!(room.recorded_events.len(), 1);
}

#[test]
fn test_maybe_record_ignores_when_not_recording() {
    let store = new_room_store();
    let code = create_room(&store);
    maybe_record(&store, &code, serde_json::json!({"type": "token", "text": "hello"}));
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert_eq!(room.recorded_events.len(), 0);
}

#[test]
fn test_stop_recording_returns_events() {
    let store = new_room_store();
    let code = create_room(&store);
    start_recording(&store, &code);
    maybe_record(&store, &code, serde_json::json!({"type": "token", "text": "a"}));
    maybe_record(&store, &code, serde_json::json!({"type": "token", "text": "b"}));
    let events = stop_recording(&store, &code);
    assert_eq!(events.len(), 2);
}

#[test]
fn test_stop_recording_clears_events_from_room() {
    let store = new_room_store();
    let code = create_room(&store);
    start_recording(&store, &code);
    maybe_record(&store, &code, serde_json::json!({"type": "token"}));
    stop_recording(&store, &code);
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert!(room.recorded_events.is_empty());
}

#[test]
fn test_start_recording_clears_previous_events() {
    let store = new_room_store();
    let code = create_room(&store);
    start_recording(&store, &code);
    maybe_record(&store, &code, serde_json::json!({"type": "token"}));
    stop_recording(&store, &code);
    // Start again — previous events should be gone.
    start_recording(&store, &code);
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    assert!(room.recorded_events.is_empty());
}

#[test]
fn test_recorded_event_has_offset_ms() {
    let store = new_room_store();
    let code = create_room(&store);
    start_recording(&store, &code);
    std::thread::sleep(std::time::Duration::from_millis(2));
    maybe_record(&store, &code, serde_json::json!({"type": "token"}));
    let guard = store.lock().unwrap();
    let room = guard.get(&code).unwrap();
    // offset_ms should be ≥ 0 (approximately 2ms but timing is inexact in CI)
    assert!(room.recorded_events[0].offset_ms < 10_000);
}

// ---------------------------------------------------------------------------
// Room state snapshot
// ---------------------------------------------------------------------------

#[test]
fn test_room_state_snapshot_contains_code() {
    let store = new_room_store();
    let code = create_room(&store);
    let snap = room_state_snapshot(&store, &code);
    assert_eq!(snap["code"].as_str(), Some(code.as_str()));
}

#[test]
fn test_room_state_snapshot_contains_participants_array() {
    let store = new_room_store();
    let code = create_room(&store);
    let snap = room_state_snapshot(&store, &code);
    assert!(snap["participants"].is_array());
}

#[test]
fn test_room_state_snapshot_participant_count() {
    let store = new_room_store();
    let code = create_room(&store);
    let _ = join_room(&store, &code, "Alice", true).unwrap();
    let _ = join_room(&store, &code, "Bob", false).unwrap();
    let snap = room_state_snapshot(&store, &code);
    assert_eq!(snap["participants"].as_array().unwrap().len(), 2);
}

#[test]
fn test_room_state_snapshot_nonexistent_room_is_null() {
    let store = new_room_store();
    let snap = room_state_snapshot(&store, "XXXXXX");
    assert!(snap.is_null());
}

#[test]
fn test_room_state_snapshot_has_surgery_log() {
    let store = new_room_store();
    let code = create_room(&store);
    apply_surgery(&store, &code, make_edit(0, "new", "old"));
    let snap = room_state_snapshot(&store, &code);
    assert_eq!(snap["surgery_log"].as_array().unwrap().len(), 1);
}

#[test]
fn test_room_state_snapshot_has_votes() {
    let store = new_room_store();
    let code = create_room(&store);
    vote(&store, &code, "reverse", "up");
    let snap = room_state_snapshot(&store, &code);
    assert!(snap["votes"].is_object());
}

// ---------------------------------------------------------------------------
// broadcast (smoke test — just verify no panic with no subscribers)
// ---------------------------------------------------------------------------

#[test]
fn test_broadcast_no_subscribers_does_not_panic() {
    let store = new_room_store();
    let code = create_room(&store);
    // No panic even when there are no active subscribers.
    broadcast(&store, &code, serde_json::json!({"type": "ping"}));
}

#[test]
fn test_broadcast_nonexistent_room_does_not_panic() {
    let store = new_room_store();
    broadcast(&store, "XXXXXX", serde_json::json!({"type": "ping"}));
}

// ---------------------------------------------------------------------------
// Participant colors constant
// ---------------------------------------------------------------------------

#[test]
fn test_participant_colors_not_empty() {
    assert!(!PARTICIPANT_COLORS.is_empty());
}

#[test]
fn test_participant_colors_are_hex() {
    for color in PARTICIPANT_COLORS {
        assert!(color.starts_with('#'), "Color {} doesn't start with #", color);
        assert_eq!(color.len(), 7, "Color {} isn't 7 chars", color);
    }
}
