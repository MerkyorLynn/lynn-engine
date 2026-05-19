# P196 Qwen3.5-9B W4A8 Structured Content Gate

Date: 2026-05-19

## Result

R6000 ran the 70-prompt hard structured set with chat-template semantics:

| Mode | Pass | Pass Rate | Mean Decode TPS | Interpretation |
|---|---:|---:|---:|---|
| W4A16 reference | 63/70 | 90.00% | 60.49 | The 70-prompt set is harder than the current 9B profile |
| W4A8 gate/up fake-quant | 64/70 | 91.43% | 60.19 | No regression versus W4A16 |
| W4A8 full fake-quant | 64/70 | 91.43% | 57.81 | No regression versus W4A16 |

Raw report:

```text
/root/autodl-tmp/reports/qwen35_9b/p196_qwen35_9b_w4a8_structured_content_gate_20260519_1718_p196_chat70.json
```

## Decision

This is **not** a default-promotion GREEN, because W4A16 itself does not pass
70/70 on this hard set.

It is an important relative result:

```text
W4A8_RELATIVE_NO_REGRESSION_ABSOLUTE_AMBER
```

W4A8 did not damage structured content quality on this gate. The next blocker
is not quality; it is implementing a real native FP8-active dense FFN path that
preserves this behavior and improves speed.

## Failed Prompt Classes

The common failures are concentrated in hard exact-format prompts:

- `python_normalize_city_code_only`
- `yaml_openapi_request_body`
- `json_h_nested_tool_array`
- `json_h_response_schema_with_enum`
- `json_h_array_of_objects_sorted`
- `json_h_array_only_no_object_wrap`
- `list_h_json_strings_array`

The W4A8 modes do not introduce a new broad failure pattern.

## Policy

- Keep W4A16 as the safe NVIDIA fallback.
- Treat W4A8 as a speed candidate only after a native FP8-active resident path
  passes this content gate.
- Do not use exact-token parity alone to reject W4A8 on 9B; use content gates
  and quality benchmarks as the release signal.
