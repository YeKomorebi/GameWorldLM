# Spatial Failure Report

Stored raw predictions; no repairs or retraining.

| Metric | base | lora |
| --- | ---: | ---: |
| json_parse_rate | 14.29% | 100.00% |
| schema_valid_rate | 14.29% | 100.00% |
| spatial_validator_pass_rate | 0.00% | 21.43% |
| object_count_accuracy | 14.29% | 100.00% |
| all_expectations_pass_rate | 0.00% | 21.43% |
| boundary_accuracy | 14.29% | 100.00% |
| overlap_free_rate | 0.00% | 28.57% |
| task_constraint_pass_rate | N/A | N/A |
| near_accuracy | 0.00% | 75.00% |
| inside_accuracy | 0.00% | 66.67% |
| connected_to_accuracy | 0.00% | 57.14% |

## base: 14 samples

| Category | Issue occurrences | Affected samples |
| --- | ---: | ---: |
| overlap | 12 | 2 |
| out_of_bounds | 0 | 0 |
| near_violation | 0 | 0 |
| inside_violation | 1 | 1 |
| connected_to_violation | 1 | 1 |
| invalid_reference | 0 | 0 |
| other | 12 | 12 |

Original error codes: `{"blocked_entity": 8, "json_parse": 12, "not_connected": 1, "not_inside": 1, "overlap": 4}`.

Relation passes / slots: `{"near": {"passed": 0, "total": 4}, "inside": {"passed": 0, "total": 3}, "connected_to": {"passed": 0, "total": 7}}`.

### Exact failures

- `913ef8f442136763c8208dd4279ff9341cf85e17e6cd1a297e8fa5421a79cb40`: json_parse: Expecting ',' delimiter: line 1 column 253 (char 252)
- `913ef8f442136763c8208dd4279ff9341cf85e17e6cd1a297e8fa5421a79cb40` prompt requirements: Missing 1 required near relation slot(s)
- `d295a130bd1809fdce8ba66dc93101e188af59b2cbf45205c5ad54816a75a749`: json_parse: Expecting ',' delimiter: line 1 column 257 (char 256)
- `d295a130bd1809fdce8ba66dc93101e188af59b2cbf45205c5ad54816a75a749` prompt requirements: Missing 1 required connected_to relation slot(s)
- `80e0c14ed541b85ad30f62ab67f7821e68be484b23d5d0c307a333fcff05dc7c`: json_parse: Expecting ',' delimiter: line 1 column 268 (char 267)
- `a4a5fa053878e50a16c9d62bc615e6f4c130de269935b19930a3d58e0eb5dd8d`: not_connected: ('house_001', 'road_001') must share a footprint edge; overlap: Same-layer overlap: house_001, ruin_001; blocked_entity: Entity intersects structure: house_001, npc_001; blocked_entity: Entity intersects structure: house_001, monster_001; blocked_entity: Entity intersects structure: house_001, monster_002; blocked_entity: Entity intersects structure: ruin_001, monster_002; blocked_entity: Entity intersects structure: ruin_002, monster_003; blocked_entity: Entity intersects structure: ruin_003, monster_004; overlap: Same-layer overlap: vehicle_001, npc_003
- `87fcb381ffc4d97eb2a8c5a8a5d2a32ffc6c95da576d90453a286276439dfdc0`: json_parse: Expecting ',' delimiter: line 1 column 266 (char 265)
- `87fcb381ffc4d97eb2a8c5a8a5d2a32ffc6c95da576d90453a286276439dfdc0` prompt requirements: Missing 1 required near relation slot(s)
- `c014c5903ed24eade059c0f29aec51ad9f791fd872632402230aafeee4f13676`: json_parse: Expecting ',' delimiter: line 1 column 258 (char 257)
- `c014c5903ed24eade059c0f29aec51ad9f791fd872632402230aafeee4f13676` prompt requirements: Missing 1 required connected_to relation slot(s)
- `b229d65cf1d6d1fae45f3f47f4dd410ca23f795b763e51f5799559103177f2b9`: json_parse: Expecting ',' delimiter: line 1 column 266 (char 265)
- `b229d65cf1d6d1fae45f3f47f4dd410ca23f795b763e51f5799559103177f2b9` prompt requirements: Missing 1 required inside relation slot(s)
- `3a30367a62b6f93bb9a2b9b1401c52f60c92bb2ae891820c180806de6a01b393`: json_parse: Expecting property name enclosed in double quotes: line 1 column 257 (char 256)
- `3a30367a62b6f93bb9a2b9b1401c52f60c92bb2ae891820c180806de6a01b393` prompt requirements: Missing 1 required connected_to relation slot(s)
- `af8b5cc83c7e5fc9a4715692db8e23660d3121b2305fe62717ee984ef331584d`: not_inside: house_001 must fit within npc_001; overlap: Same-layer overlap: house_001, rock_001; blocked_entity: Entity intersects structure: house_001, npc_001; blocked_entity: Entity intersects structure: house_001, monster_002; overlap: Same-layer overlap: house_002, rock_004
- `b07906da16cedef4365e2320b9034bfbf6ad764cf6dafa247453b5d292379fe7`: json_parse: Expecting ',' delimiter: line 1 column 255 (char 254)
- `ff4a224c0339354a723ac6bbba55c785c64ed634e14daaf13ef503a7d7a8c97e`: json_parse: Expecting property name enclosed in double quotes: line 1 column 252 (char 251)
- `ff4a224c0339354a723ac6bbba55c785c64ed634e14daaf13ef503a7d7a8c97e` prompt requirements: Missing 1 required inside relation slot(s)
- `880783635a788e4c09eba25d982d386b1a74bbd9428db4a27c70005979e82262`: json_parse: Expecting property name enclosed in double quotes: line 1 column 238 (char 237)
- `880783635a788e4c09eba25d982d386b1a74bbd9428db4a27c70005979e82262` prompt requirements: Missing 2 required connected_to relation slot(s)
- `52abb770596cc6be356d11977520a1abd24fabfe4fe770b6d33d235375571ed1`: json_parse: Expecting ',' delimiter: line 1 column 273 (char 272)
- `52abb770596cc6be356d11977520a1abd24fabfe4fe770b6d33d235375571ed1` prompt requirements: Missing 1 required near relation slot(s)
- `c45599860ad17c13deacd9857ab674e1925272d6a6b006743fba99840cfb17aa`: json_parse: Expecting ',' delimiter: line 1 column 236 (char 235)
- `c45599860ad17c13deacd9857ab674e1925272d6a6b006743fba99840cfb17aa` prompt requirements: Missing 1 required near relation slot(s); Missing 1 required connected_to relation slot(s)

Full prompts, generated JSON and validator results: `base_spatial_samples.jsonl`.

## lora: 14 samples

| Category | Issue occurrences | Affected samples |
| --- | ---: | ---: |
| overlap | 18 | 10 |
| out_of_bounds | 0 | 0 |
| near_violation | 0 | 0 |
| inside_violation | 1 | 1 |
| connected_to_violation | 3 | 2 |
| invalid_reference | 0 | 0 |
| other | 0 | 0 |

Original error codes: `{"blocked_entity": 11, "not_connected": 3, "not_inside": 1, "overlap": 7}`.

Relation passes / slots: `{"near": {"passed": 3, "total": 4}, "inside": {"passed": 2, "total": 3}, "connected_to": {"passed": 4, "total": 7}}`.

### Exact failures

- `913ef8f442136763c8208dd4279ff9341cf85e17e6cd1a297e8fa5421a79cb40`: overlap: Same-layer overlap: tower_1, portal_1; blocked_entity: Entity intersects structure: tower_1, npc_1
- `d295a130bd1809fdce8ba66dc93101e188af59b2cbf45205c5ad54816a75a749`: overlap: Same-layer overlap: castle_2, crystal_3
- `a4a5fa053878e50a16c9d62bc615e6f4c130de269935b19930a3d58e0eb5dd8d`: blocked_entity: Entity intersects structure: ruin_1, monster_1; blocked_entity: Entity intersects structure: ruin_4, npc_2
- `87fcb381ffc4d97eb2a8c5a8a5d2a32ffc6c95da576d90453a286276439dfdc0`: blocked_entity: Entity intersects structure: tower_1, npc_1; overlap: Same-layer overlap: library_3, library_4
- `c014c5903ed24eade059c0f29aec51ad9f791fd872632402230aafeee4f13676`: overlap: Same-layer overlap: ruin_2, crystal_2
- `b229d65cf1d6d1fae45f3f47f4dd410ca23f795b763e51f5799559103177f2b9`: not_inside: npc_1 must fit within ruin_1; overlap: Same-layer overlap: dune_2, road_1
- `3a30367a62b6f93bb9a2b9b1401c52f60c92bb2ae891820c180806de6a01b393`: blocked_entity: Entity intersects structure: tower_1, vehicle_1; blocked_entity: Entity intersects structure: tower_1, npc_1; blocked_entity: Entity intersects structure: tower_2, npc_3
- `b07906da16cedef4365e2320b9034bfbf6ad764cf6dafa247453b5d292379fe7`: blocked_entity: Entity intersects structure: house_1, npc_1
- `ff4a224c0339354a723ac6bbba55c785c64ed634e14daaf13ef503a7d7a8c97e`: overlap: Same-layer overlap: library_3, portal_3; blocked_entity: Entity intersects structure: library_3, npc_3
- `880783635a788e4c09eba25d982d386b1a74bbd9428db4a27c70005979e82262`: not_connected: ('road_1', 'bridge_1') must share a footprint edge; not_connected: ('road_2', 'bridge_1') must share a footprint edge
- `c45599860ad17c13deacd9857ab674e1925272d6a6b006743fba99840cfb17aa`: not_connected: ('road_1', 'castle_1') must share a footprint edge; overlap: Same-layer overlap: castle_1, rock_3; blocked_entity: Entity intersects structure: castle_1, guard_1; blocked_entity: Entity intersects structure: castle_1, guard_2
- `c45599860ad17c13deacd9857ab674e1925272d6a6b006743fba99840cfb17aa` prompt requirements: Missing 1 required near relation slot(s)

Full prompts, generated JSON and validator results: `lora_spatial_samples.jsonl`.

## Metric definitions

- denominator: all samples, including invalid JSON, schema failures and inference errors
- object_count_accuracy: fraction with every explicitly expected type count correct
- parsing: strict raw JSON, no code-fence stripping, repair or substring extraction
- relation_accuracy: sum(valid unique predicted edges) / sum(max(predicted edges + missing explicitly requested edges, required minimum)). Invalid JSON contributes required slots but zero passes. Zero total is N/A. near/connected are symmetric; inside is directional. Duplicate edges add failures. Geometry only: an edge may pass while collision or boundary checks fail.
- legacy_relation_scope: v0.4 prompts specify type minimums, not endpoint IDs; new benchmark uses required_relations to additionally penalize wrong endpoints.
- boundary_accuracy: fraction of ALL samples with schema-valid worlds, unique IDs and every footprint within its declared map; map matching is a separate expectation
- overlap_free_rate: fraction of ALL samples with schema-valid worlds, unique IDs, no overlap or blocked_entity; canonical layer and explicit containment rules apply
- failure_categories: issue occurrences and unique sample counts are reported separately; blocked_entity maps to overlap; parse/schema/inference errors to other. Missing requested relations affect metrics, not canonical validator semantics.
- task_constraint_pass_rate: among tasks with required_objects: full spatial and expectation pass plus exact IDs/types/sizes, required edges and requested boundary contact. N/A for legacy v0.4 samples without these annotations.
