# Security Specification - BovinoVision Security Suite

## Data Invariants
1. **User Ownership**: All cattle records must contain a `userId` field matching the creator's verified Firebase Auth UID (`request.auth.uid`). A user cannot create, view, modify, or delete another user's records.
2. **Profile Isolation**: Veterinarian user profiles must be mapped strictly by their UID `/users/{userId}` where `{userId}` must match `request.auth.uid`. No user can edit someone else's profile.
3. **Value Range Enforcement**: Cattle records must contain validated zootecnic bounds:
   - Body condition score (`score`) must be a number between `1.0` and `5.0`.
   - Live weight (`weight`) must be a positive number up to `1500` kg.
   - Confidence levels (`aiConfidence`) and Fat levels (`fatProgress`) must be percentages within physical limits.
4. **Finite Verdicts**: The slaughtered verdict on a record must strictly equal either `"APTO PARA ABATE"` or `"NÃO APTO"`.
5. **No Spoofing**: Users cannot spoof creation timestamps or update tracking info.

---

## The "Dirty Dozen" Payloads
These payloads attempt to bypass authorization, inject corrupt data, or escalate credentials:

1. **Identity Spoofing - Create another user's profile**
   - Path: `/users/hacker-id`
   - Payload: `{ "uid": "victim-id", "email": "victim@domain.com", "name": "Fake Profile" }`
   - Expected Output: `PERMISSION_DENIED`
2. **Identity Spoofing - Write record with guest UID**
   - Path: `/records/rec-001`
   - Payload: `{ "id": "rec-001", "userId": "victim-id", "score": 3.5, "weight": 520, "verdict": "APTO PARA ABATE" }`
   - Expected Output: `PERMISSION_DENIED`
3. **Data Poisoning - Negative range condition score**
   - Path: `/records/rec-002`
   - Payload: `{ "id": "rec-002", "userId": "hacker-id", "score": -1.2, "weight": 440, "verdict": "NÃO APTO" }`
   - Expected Output: `PERMISSION_DENIED`
4. **Data Poisoning - Colossal condition score**
   - Path: `/records/rec-003`
   - Payload: `{ "id": "rec-003", "userId": "hacker-id", "score": 999.0, "weight": 440, "verdict": "NÃO APTO" }`
   - Expected Output: `PERMISSION_DENIED`
5. **Data Poisoning - Negative weight**
   - Path: `/records/rec-004`
   - Payload: `{ "id": "rec-004", "userId": "hacker-id", "score": 3.2, "weight": -50.5, "verdict": "NÃO APTO" }`
   - Expected Output: `PERMISSION_DENIED`
6. **State Shortcutting - Arbitrary verdict value injection**
   - Path: `/records/rec-005`
   - Payload: `{ "id": "rec-005", "userId": "hacker-id", "score": 3.2, "weight": 450, "verdict": "IMEDIATAMENTE VENDA" }`
   - Expected Output: `PERMISSION_DENIED`
7. **Resource Poisoning - Gigantic string injection in ID field**
   - Path: `/records/rec_huge_id_poisoning_attack_string_with_excessive_characters_over_one_hundred_and_twenty_eight_or_more`
   - Payload: `{ "id": "valid", "userId": "hacker-id", "score": 3.0, "weight": 450, "verdict": "NÃO APTO" }`
   - Expected Output: `PERMISSION_DENIED`
8. **PII Blanket Leak - Read user configuration without authentication**
   - Path: `/users/some-user`
   - Action: `get`
   - Expected Output: `PERMISSION_DENIED`
9. **Cross-user Reading - Unauthorized list queries on other's data**
   - Path: `/records`
   - Action: `list` (without matching user query parameters)
   - Expected Output: `PERMISSION_DENIED`
10. **Malicious Ownership Shifting - Changing userId on records**
    - Path: `/records/rec-owned`
    - Update Payload: `{ "userId": "victim-id" }` (affectedKeys bypass attempt)
    - Expected Output: `PERMISSION_DENIED`
11. **Malicious Timestamp Spoofing**
    - Path: `/records/rec-008`
    - Payload: `{ "id": "rec-008", "userId": "hacker-id", "score": 3.0, "weight": 400, "verdict": "NÃO APTO", "createdAt": "2020-01-01" }`
    - Expected Output: `PERMISSION_DENIED`
12. **Malicious Privilege Escalation - Altering profile admin indicator**
    - Path: `/users/hacker-id`
    - Update Payload: `{ "isAdmin": true }`
    - Expected Output: `PERMISSION_DENIED`

---

## Test Verification Suite (firestore.rules.test.ts)
A test validation runner script mapped under the test setup block to assert continuous fortress-level access bounds.
