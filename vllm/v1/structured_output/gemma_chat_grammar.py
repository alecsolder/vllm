# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gemma4 chat-format tool-call grammar generator for llguidance.

Generates a Lark grammar string (with the ``%llguidance {}`` header) that
constrains a Gemma4 model's output to the Gemma4 tool-call envelope::

    <|tool_call>call:NAME{key:value,key:<|"|>strvalue<|"|>}<tool_call|>

The grammar refers to Gemma4 special tokens by their integer token IDs, so
constraint enforcement works at the token level via ``llguidance.LLMatcher``.

The generated grammar supports one or more tool calls, required and optional
parameters in any order, nested objects and arrays, local ``$ref`` definitions,
unions, enums, optional reasoning blocks, natural-language fallback, and a JSON
Schema response branch.

Input: list of OpenAI-style tool definitions, each shaped like::

    {
        "type": "function",
        "function": {
            "name": "...",
            "parameters": {  # JSON Schema
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        },
    }

Output: a Lark grammar string ready for ``llguidance.LLMatcher``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# Gemma4 special-token IDs. Keep these synchronized with the model tokenizer;
# callers may provide an alternate ``Gemma4SpecialTokens`` mapping.
_TOOL_CALL_START_ID = 48  # <|tool_call>
_TOOL_CALL_END_ID = 49  # <tool_call|>
_QUOTE_ID = 52  # <|"|>
_CHANNEL_START_ID = 100  # <|channel>  (thinking-block open)
_CHANNEL_END_ID = 101  # <channel|>  (thinking-block close)
_TURN_END_ID = 106  # <turn|>  (terminator for natural-language refusal branch)
_EOS_ID = 1  # <eos>  (Gemma4 generation_config eos)
_TOOL_RESPONSE_START_ID = 50  # <|tool_response>  (post-call sentinel)

# Full set of Gemma4 reserved special-token IDs — EVERY ``added_tokens`` entry
# in tokenizer.json with ``"special": true`` (all 24). A freeform string value's
# interior forbids this WHOLE set (the ``no_special_token`` content category), so
# a converged value can never commit ANY reserved sentinel (pad/eos/bos/unk/
# closer/channel/turn/multimodal/…) and leave the string open/unterminated. The
# IDs are scattered across the low control block (0-106) AND the high multimodal
# block (255999-258884), so they are enumerated EXPLICITLY — never a numeric
# range: Gemma byte-fallback tokens (``<0xHH>``, IDs 238-493) are ordinary string
# content and must stay legal. ``test_all_special_token_ids_match_tokenizer``
# re-derives this set from the live tokenizer and fails loudly if a future
# release renumbers / adds / removes any.
_ALL_SPECIAL_IDS: tuple[int, ...] = (
    0,
    1,
    2,
    3,
    4,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    98,
    100,
    101,
    105,
    106,
    255999,
    256000,
    258880,
    258881,
    258882,
    258883,
    258884,
)


@dataclass(frozen=True)
class Gemma4SpecialTokens:
    """Gemma4 special-token IDs used by the tool-call grammar.

    Kept as a dataclass so an alternate Gemma4 tokenizer can supply remapped
    IDs without changing the grammar builder.
    """

    tool_call_start: int = _TOOL_CALL_START_ID
    tool_call_end: int = _TOOL_CALL_END_ID
    quote: int = _QUOTE_ID
    channel_start: int = _CHANNEL_START_ID
    channel_end: int = _CHANNEL_END_ID
    turn_end: int = _TURN_END_ID
    eos: int = _EOS_ID
    tool_response_start: int = _TOOL_RESPONSE_START_ID
    # The complete reserved special-token set (NOT just the named role tokens
    # above) — forbidden inside freeform string values. Default = the stock
    # Gemma4 tokenizer's 24 special IDs; override alongside a remapped tokenizer.
    all_special_ids: tuple[int, ...] = _ALL_SPECIAL_IDS


# JSON Schema primitive type -> regex pattern (no surrounding slashes).
# Strings are handled separately because they're framed in Gemma4 by the
# special ``<|"|>`` quote token, not by ASCII double-quotes.
_PRIMITIVE_PATTERNS: dict[str, str] = {
    "integer": r"-?(?:0|[1-9][0-9]*)",
    "number": r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?",
    "boolean": r"(?:true|false)",
}

# BFCL (and some other tool-calling datasets) ship schemas using Python
# type names instead of canonical JSON Schema types.  Normalize them so
# the rest of the generator only has to dispatch on JSON Schema names.
# Without this alias map, a ``"type": "float"`` schema falls through the
# primitive-pattern lookup and gets the string-quote wrapper — forcing
# the model to emit ``<|"|>1e-9<|"|>`` for a numeric parameter, which
# the BFCL evaluator then flags as a ``value_error:string`` mismatch.
_TYPE_ALIASES: dict[str, str] = {
    "float": "number",
    "int": "integer",
    "bool": "boolean",
    "dict": "object",
    "tuple": "array",
}


def _normalize_type(raw_type: Any) -> str | list[str]:
    """Map BFCL-style Python type names to canonical JSON Schema types.

    Accepts a single string or a list (JSON Schema union type).  ``"null"``
    is preserved unchanged; unknown types pass through unchanged so the
    caller's default-handling (string fallback) still applies.
    """
    if isinstance(raw_type, list):
        return [_TYPE_ALIASES.get(t, t) for t in raw_type]
    return _TYPE_ALIASES.get(raw_type, raw_type)


def _resolve_ref(ref: str, root_schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Resolve a local JSON-Schema ``$ref`` against the root schema.

    Handles intra-document refs of the form ``#/$defs/Name`` or
    ``#/definitions/Name`` (and deeper pointer paths). Returns the
    referenced sub-schema dict, or ``None`` if it can't be resolved
    (non-local ref, missing root, or broken pointer) so the caller can
    fall back gracefully.
    """
    if not isinstance(ref, str) or not ref.startswith("#/") or root_schema is None:
        return None
    node: Any = root_schema
    for part in ref[2:].split("/"):
        # JSON-pointer unescaping (~1 -> /, ~0 -> ~).
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node if isinstance(node, dict) else None


def _sanitize_rule_name(name: str) -> str:
    """Convert a tool/parameter name to a valid Lark rule identifier.

    Lark rule names must be lowercase ASCII identifiers; uppercase reads
    as a terminal.  Non-alphanumerics collapse to underscores.
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()


def _escape_lark_regex(s: str) -> str:
    """Escape a literal string for embedding in a Lark ``/.../`` regex.

    ``re.escape`` neutralizes regex metacharacters (the precedent used by
    :func:`_build_tool_rules` for tool/param names), but it does NOT escape
    the forward slash, which terminates a Lark regex literal — so we escape
    ``/`` to ``\\/`` on top. Enum/const values (e.g. units like ``"km/h"``)
    are more likely to contain ``/`` than tool names, hence the extra step.
    """
    return re.escape(s).replace("/", r"\/")


def _enum_member_literal(value: Any) -> str | None:
    """Lark-escaped regex body for ONE enum/const member (no frame).

    Returns ``None`` for a non-primitive member (object/array/null) that
    cannot be a literal — the caller skips those so an enum with structured
    members degrades to the type pattern rather than wrongly rejecting them.
    ``bool`` is checked before ``int`` (it is an ``int`` subclass) and
    rendered lowercase to match JSON/Gemma4 output — ``str(True)`` would emit
    the off-distribution ``"True"``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _escape_lark_regex(str(value))
    if isinstance(value, str):
        return _escape_lark_regex(value)
    return None


def _enum_quote_framed(pschema: dict[str, Any], values: list[Any]) -> bool:
    """Whether an enum/const value is Gemma4 quote-framed (string) vs bare.

    Decided by the declared ``type`` (``string`` → framed; integer/number/
    boolean → bare). When ``type`` is absent or non-decisive (object/array/
    unknown), inferred from the python type of the members — bare only when
    every non-null member is an int/number/bool primitive.
    """
    raw_type = _normalize_type(pschema.get("type"))
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        raw_type = non_null[0] if non_null else None
    if raw_type in _PRIMITIVE_PATTERNS:
        return False
    if raw_type == "string":
        return True
    prims = [v for v in values if v is not None]
    return not (prims and all(isinstance(v, (bool, int, float)) for v in prims))


def _build_param_value_expr(
    pschema: dict[str, Any],
    quote_id: int,
    rule_prefix: str,
    root_schema: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Build the Lark fragment for a single parameter's value.

    Thin shim that delegates to :func:`_build_value_expr`. Caller
    supplies a unique ``rule_prefix`` so nested helper rule names
    don't collide across parameters. ``root_schema`` (the tool's full
    ``parameters`` schema) is threaded through for ``$ref`` resolution.

    NOTE: replaces the previous ``%json <schema>`` implementation,
    which forced model output to standard ASCII JSON. Gemma4 emits
    object args in a NATIVE compact syntax (``<|"|>`` quotes, bareword
    keys, comma-separated pairs). Masking that syntax into ASCII JSON pushes
    generation off-distribution and can produce low-probability malformed output.
    """
    return _build_value_expr(pschema, quote_id, rule_prefix, root_schema)


def _build_value_expr(
    pschema: dict[str, Any],
    quote_id: int,
    rule_prefix: str,
    root_schema: dict[str, Any] | None = None,
    _depth: int = 0,
) -> tuple[str, list[str]]:
    """Recursive Gemma4-native encoder for any JSON-Schema value type.

    Returns ``(value_expr, extra_rule_defs)``. ``rule_prefix`` namespaces
    helper rule names (so nested objects don't collide). ``root_schema``
    is the tool's full ``parameters`` schema, threaded through so local
    ``$ref`` (``#/$defs/...``) can be resolved; ``_depth`` guards against
    cyclic ``$ref`` chains.

    Gemma4 native arg syntax::

        primitive (int/number/boolean): plain literal (123, 1.5, true)
        string:                          <|"|>value<|"|>   (Gemma4 quote token id 52)
        object: {key1:value1,key2:value2} (bareword keys, comma-separated)
        array:  [v0,v1,...] (comma-separated elements)

    This DIFFERS from standard JSON, so llguidance's ``%json`` sub-parser
    (ASCII-only) must NOT be used.
    """
    # Cyclic-$ref / runaway-recursion guard. Beyond this depth, fall back
    # to the permissive open_value rule (any Gemma4-native value) so we
    # never hang the grammar build; correctness for pathological schemas
    # is not worth an unbounded expansion.
    if _depth > 8:
        return "open_value", []

    # --- Resolve a local $ref against the root schema's $defs/definitions.
    # The referenced node replaces this schema, then we recurse (the ref
    # target may itself be an object/array/anyOf). Merge any sibling keys
    # (JSON Schema allows $ref alongside other keywords) with ref winning.
    if isinstance(pschema, dict) and "$ref" in pschema:
        resolved = _resolve_ref(pschema["$ref"], root_schema)
        if resolved is not None:
            merged = {k: v for k, v in pschema.items() if k != "$ref"}
            merged.update(resolved)
            return _build_value_expr(
                merged, quote_id, rule_prefix, root_schema, _depth + 1
            )
        # Unresolvable ref → permissive fallback rather than a string.
        return "open_value", []

    # --- enum / const: compile the allowed values as a literal regex
    # alternation so the veto can reject an out-of-enum but type-valid value
    # (e.g. ``unit="kelvin"`` for ``enum:["celsius","fahrenheit"]``) — the #1
    # value-corruption lever. Placed before the anyOf/type dispatch so
    # enum/const take precedence over the declared ``type``.
    #
    # Shape mirrors the live emit: a STRING value keeps its quote-token frame
    # with the alternation replacing the free interior
    # (``<|"|> /(a|b)/ <|"|>``); a PRIMITIVE value is a bare ``/(a|b)/`` (cf.
    # ``open_primitive``). ``const`` is a single-element enum (one alt, no
    # parens). Value-interior only — does NOT change reject-not-repair, so a
    # stubborn out-of-enum value is suppressed/truncated, never rewritten;
    # the net accuracy gain is empirical.
    #
    # Non-primitive members (object/array/null) are skipped (can't be a
    # literal); if NONE remain we fall through to the normal type dispatch so
    # a structured enum degrades gracefully instead of crashing the build.
    # String enums are enforced literally. Some datasets use illustrative rather
    # than exhaustive enums, so this is an intentional prototype tradeoff.
    if isinstance(pschema, dict) and ("const" in pschema or "enum" in pschema):
        if "const" in pschema:
            enum_values: list[Any] = [pschema["const"]]
        else:
            raw_enum = pschema.get("enum")
            enum_values = list(raw_enum) if isinstance(raw_enum, list) else []
        literals: list[str] = []
        for member in enum_values:
            lit = _enum_member_literal(member)
            if lit is not None and lit not in literals:
                literals.append(lit)
        if literals:
            framed = _enum_quote_framed(pschema, enum_values)
            if framed and literals == [""]:
                # Sole empty-string member: bare quote pair (no empty // regex).
                return f"<[{quote_id}]> <[{quote_id}]>", []
            alts = "|".join(literals)
            regex = f"/{alts}/" if len(literals) == 1 else f"/({alts})/"
            if framed:
                return f"<[{quote_id}]> {regex} <[{quote_id}]>", []
            return regex, []

    # --- anyOf / oneOf: build an alternation over the (non-null) branches.
    # This is what makes array-of-object params legal: when `items` is
    # `{"anyOf":[{"$ref":"#/$defs/Passenger"},{"type":"object",...}]}` the
    # object branch contributes the `{...}` sub-rule, so the object-open
    # `{` becomes a grammar-legal array element (previously masked, forcing
    # an empty array). Each branch gets a distinct sub-prefix so their
    # helper rule names (perm/params) don't collide.
    branches = pschema.get("anyOf") or pschema.get("oneOf")
    if isinstance(branches, list) and branches:
        # Build each (non-null) branch, tracking whether it's an object-typed
        # value so we can collapse redundant object branches below.
        built: list[tuple[str, list[str]]] = []  # (expr, extras)
        has_open_object = False
        for i, branch in enumerate(branches):
            if not isinstance(branch, dict):
                continue
            # Skip explicit null branches — Gemma4 args don't emit `null`
            # for an absent optional; the param simply isn't present.
            if branch.get("type") == "null":
                continue
            b_expr, b_extras = _build_value_expr(
                branch, quote_id, f"{rule_prefix}_alt{i}", root_schema, _depth + 1
            )
            if b_expr == "open_object":
                has_open_object = True
            built.append((b_expr, b_extras))

        # Lexer-ambiguity collapse: when an open-object branch is present
        # (``{additionalProperties:true, type:object}``), it accepts ANY
        # object — so the union with any strict-object branch (a ``$ref`` to
        # a fixed-property object) is just "any object". Keeping both is not
        # only redundant, it BREAKS parsing: the strict branch's contextual
        # key terminals (e.g. ``/date:/``) shadow ``open_object``'s generic
        # ``open_key`` lexeme, so once the lexer commits a shared leading key
        # the open-object branch dies and an extra key (``destination``) is
        # rejected even though the schema admits it. Drop strict-object branches
        # when an open-object branch covers them; keep ``open_object`` plus any
        # non-object branches (strings/numbers/arrays).
        def _is_object_branch(expr: str) -> bool:
            return expr == "open_object" or expr.startswith("/\\{")

        alt_exprs: list[str] = []
        alt_extras: list[str] = []
        seen_exprs: set[str] = set()
        for b_expr, b_extras in built:
            if (
                has_open_object
                and _is_object_branch(b_expr)
                and b_expr != "open_object"
            ):
                # Subsumed by the open-object branch — drop it (and its
                # now-unused helper rules) to avoid the lexer conflict.
                continue
            if b_expr not in seen_exprs:
                seen_exprs.add(b_expr)
                alt_exprs.append(b_expr)
                alt_extras.extend(b_extras)
        if alt_exprs:
            if len(alt_exprs) == 1:
                return alt_exprs[0], alt_extras
            return "( " + " | ".join(alt_exprs) + " )", alt_extras
        # All branches null/invalid → permissive fallback.
        return "open_value", []

    raw_type = _normalize_type(pschema.get("type", "string"))
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        primitives = [t for t in non_null if t in _PRIMITIVE_PATTERNS]
        if primitives:
            ptype = primitives[0]
        elif non_null:
            ptype = non_null[0]
        else:
            ptype = "string"
    else:
        ptype = raw_type

    if ptype == "object":
        return _build_object_body(pschema, rule_prefix, quote_id, root_schema, _depth)

    if ptype == "array":
        items_schema = pschema.get("items") or {}
        if not isinstance(items_schema, dict):
            items_schema = {}
        item_prefix = f"{rule_prefix}_item"
        item_expr, item_extras = _build_value_expr(
            items_schema, quote_id, item_prefix, root_schema, _depth + 1
        )
        # Array brackets + element separator tolerate the whitespace the model
        # emits — ``[12, 15, 18]`` (space token) and the merged ``, -`` token in
        # ``[33.4484, -112.074]``. Production output allows full ``\s*``
        # around array commas/brackets as plain regex TERMINALS, and the past
        # CUDA-crash caveat was in the OBJECT perm rules (parametric bit-flag
        # rules), NOT arrays — so ``\s*`` here is whitespace-safe + lower per-token
        # mask cost. (The recursive free ``open_array`` keeps the narrow
        # space/newline class — it can't be one regex because of nested objects.)
        array_expr = (
            f"/\\[\\s*/ ( {item_expr} ( /\\s*,\\s*/ {item_expr} )* )? /\\s*\\]/"
        )
        return array_expr, item_extras

    if ptype in _PRIMITIVE_PATTERNS:
        return f"/{_PRIMITIVE_PATTERNS[ptype]}/", []

    # String (default): Gemma4 quote-token framing. The interior admits only
    # content tokens via the shared ``no_special_token`` category (defined once
    # in :func:`generate_gemma4_chat_grammar`) — every Gemma4 control sentinel
    # (eos/closer/channel/turn/tool_response/quote) is excluded so a converged
    # value can never commit a sentinel mid-string and leave the call open.
    q = quote_id
    return f"<[{q}]> no_special_token* <[{q}]>", []


def _build_object_body(
    schema: dict[str, Any],
    rule_prefix: str,
    quote_id: int,
    root_schema: dict[str, Any] | None = None,
    _depth: int = 0,
) -> tuple[str, list[str]]:
    """Build a Gemma4-native object body Lark fragment.

    Output shape::

        /\\{\\n*/ <params_rule> /\\n*\\}/

    where ``<params_rule>`` uses llguidance parametric ``perm`` rules
    (see ``llguidance/docs/parametric.md``) — same pattern as the
    top-level tool-call params (:func:`_build_tool_rules`) but
    scoped to ``rule_prefix`` so nested objects/array-elements get
    independent perm state.

    ``_depth`` is threaded into the per-property value recursion so the
    :func:`_build_value_expr` depth guard fires across object boundaries —
    without this, a cyclic ``$ref`` (e.g. ``Node.children: [Node]``) would
    reset the counter at every object level and recurse forever at
    grammar-build time.
    """
    properties = schema.get("properties") or {}
    items: list[tuple[str, dict[str, Any]]] = [
        (k, v) for k, v in properties.items() if isinstance(v, dict)
    ]

    if not items:
        return "open_object", []

    required_set = set(schema.get("required") or [])
    required_items = [(k, v) for k, v in items if k in required_set]
    optional_items = [(k, v) for k, v in items if k not in required_set]
    sorted_items = required_items + optional_items
    n_required = len(required_items)

    perm_name = f"{rule_prefix}_perm"
    params_name = f"{rule_prefix}_params"

    param_kv: list[tuple[str, str]] = []
    extras: list[str] = []
    for idx, (k, v) in enumerate(sorted_items):
        sub_prefix = f"{rule_prefix}_p{idx}_{_sanitize_rule_name(k)}"
        value_expr, sub_extras = _build_value_expr(
            v, quote_id, sub_prefix, root_schema, _depth + 1
        )
        param_kv.append((re.escape(k), value_expr))
        extras.extend(sub_extras)

    first_alts = [
        f"/{ek}:/ {ve} {perm_name}::0x{1 << idx:x}"
        for idx, (ek, ve) in enumerate(param_kv)
    ]
    if n_required == 0:
        params_rhs = '"" | ' + " | ".join(first_alts)
    else:
        params_rhs = " | ".join(first_alts)
    params_rule = f"{params_name}: {params_rhs}"

    empty_alt = f'""  %if is_ones([0:{n_required}])' if n_required > 0 else '""'
    perm_alts: list[str] = [empty_alt]
    for idx, (ek, ve) in enumerate(param_kv):
        # Pair separator tolerates a space too (``{a:1, b:2}``) — same NARROW
        # ``[ \n]*`` class as the array rule; never broad ``\s*`` here.
        perm_alts.append(
            f"/,[ \\n]*{ek}:/ {ve} {perm_name}::set_bit({idx})  %if bit_clear({idx})"
        )
    perm_rule = f"{perm_name}::_  :  " + "\n         |  ".join(perm_alts)

    body_expr = f"/\\{{\\n*/ {params_name} /\\n*\\}}/"
    return body_expr, [params_rule, perm_rule, *extras]


def _build_tool_rules(
    tool_name: str,
    safe: str,
    properties: dict[str, dict[str, Any]],
    required_set: set[str],
    tokens: Gemma4SpecialTokens,
    root_schema: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Build the body rule + helper rules for one tool.

    Returns ``(body_rhs, helper_rules)`` where ``body_rhs`` is the RHS
    of ``tc_<safe>_body`` and ``helper_rules`` is a list of additional
    top-level rule definitions (params rule + perm rule, when needed).

    No params::

        <[TC_START]> /call:NAME\\{\\}/ <[TC_END]>

    One or more params (any-order, any-subset-with-all-required) — uses
    llguidance's parametric ``perm`` rules (see
    ``llguidance/docs/parametric.md``)::

        <[TC_START]> /call:NAME\\{/ tc_NAME_params /\\}/ <[TC_END]>
        tc_NAME_params: "" | /k0:/ v0 tc_NAME_perm::0x1
                       | /k1:/ v1 tc_NAME_perm::0x2 | ...
        tc_NAME_perm::_  :  ""  %if is_ones([0:n_required])
                         |  /,k0:/ v0 tc_NAME_perm::set_bit(0)  %if bit_clear(0)
                         |  /,k1:/ v1 tc_NAME_perm::set_bit(1)  %if bit_clear(1)
                         |  ...

    ``params`` allows the model to emit either zero params (only legal
    when no params are required), or one "first" param (no leading
    comma) followed by any number of subsequent params (each with a
    leading comma) — the bit-flag invariant inside ``perm`` enforces
    that no param appears twice and that all ``n_required`` required
    bits end set.  Required params are placed first in the bit order
    so the empty-alt's ``is_ones([0:n_required])`` check is the
    completion criterion.
    """
    escaped_name = re.escape(tool_name)

    # Filter out malformed schema entries up-front.
    items: list[tuple[str, dict[str, Any]]] = [
        (pname, pschema)
        for pname, pschema in properties.items()
        if isinstance(pschema, dict)
    ]

    # NOTE: tool-call bodies are pure CALL UNITS — they do NOT consume the
    # trailing stop token. The terminal stop-token class is appended ONCE by
    # the ``tool_calls_seq`` rule in :func:`generate_gemma4_chat_grammar`
    # after one-or-more back-to-back call units, so the grammar admits
    # PARALLEL tool calls in a single turn (the prior per-body stop token
    # forbade a 2nd call).
    if not items:
        body = (
            f"<[{tokens.tool_call_start}]> "
            f"/call:{escaped_name}\\{{\\}}/ "
            f"<[{tokens.tool_call_end}]>"
        )
        return body, []

    # Sort required params before optional ones so the perm rule's
    # is_ones([0:n_required]) completion check lines up with bit 0..k-1.
    required_items = [(p, s) for p, s in items if p in required_set]
    optional_items = [(p, s) for p, s in items if p not in required_set]
    sorted_items = required_items + optional_items
    n_required = len(required_items)

    perm_name = f"tc_{safe}_perm"
    params_name = f"tc_{safe}_params"

    # Build per-param (key-literal + value-expression) once; reused in
    # both the first-param alts and the perm-rule alts.  Each composite
    # (object/array) param contributes one or more helper rule
    # definitions (params + perm rules per object level).
    param_kv: list[tuple[str, str]] = []  # (escaped_pname, value_expr)
    extra_value_rules: list[str] = []
    for idx, (pname, pschema) in enumerate(sorted_items):
        sub_prefix = f"tc_{safe}_p{idx}_{_sanitize_rule_name(pname)}"
        value_expr, extras = _build_param_value_expr(
            pschema, tokens.quote, sub_prefix, root_schema
        )
        param_kv.append((re.escape(pname), value_expr))
        extra_value_rules.extend(extras)

    # outer params rule: first-param alternatives (no leading comma)
    first_alts = [
        f"/{esc_pname}:/ {value_expr} {perm_name}::0x{1 << idx:x}"
        for idx, (esc_pname, value_expr) in enumerate(param_kv)
    ]
    if n_required == 0:
        # All optional — empty params allowed.
        params_rhs = '"" | ' + " | ".join(first_alts)
    else:
        params_rhs = " | ".join(first_alts)
    params_rule = f"{params_name}: {params_rhs}"

    # perm rule: empty alt (gated by completion criterion) + per-param
    # subsequent-emit alts (each gated by "this param not yet set").
    empty_alt = f'""  %if is_ones([0:{n_required}])' if n_required > 0 else '""'
    perm_alts: list[str] = [empty_alt]
    for idx, (esc_pname, value_expr) in enumerate(param_kv):
        # Allow optional whitespace after the comma; Gemma4 emits ``,\n``
        # (token 107) between params under thinking mode AND a plain space
        # ``{a:1, b:2}``. Constrained to the narrow ``[ \n]*`` class (space
        # + newline only — NOT ``\s*``) to minimize lexer state and llguidance
        # lookahead cost and avoid backend instability from a much broader
        # whitespace expression.
        perm_alts.append(
            f"/,[ \\n]*{esc_pname}:/ {value_expr} "
            f"{perm_name}::set_bit({idx})  %if bit_clear({idx})"
        )
    perm_rule = f"{perm_name}::_  :  " + "\n         |  ".join(perm_alts)

    # Allow ONE optional newline immediately after ``{`` and before
    # ``}``.  Matches what Gemma4 actually emits in thinking mode
    # (``{\n a:1, …\n}``); narrower than ``\s*`` to keep lexer state
    # bounded and avoid backend instability from a broader expression.
    body = (
        f"<[{tokens.tool_call_start}]> "
        f"/call:{escaped_name}\\{{\\n*/ "
        f"{params_name} "
        f"/\\n*\\}}/ "
        f"<[{tokens.tool_call_end}]>"
    )
    return body, [params_rule, perm_rule, *extra_value_rules]


def generate_gemma4_chat_grammar(
    tools: list[dict[str, Any]] | None = None,
    tokens: Gemma4SpecialTokens | None = None,
    allow_natural_language: bool = False,
    allow_thinking: bool = False,
    json_schema: str | dict[str, Any] | None = None,
) -> str:
    """Generate a unified Lark grammar for a Gemma4 chat response.

    The grammar can constrain the model to: (a) one of N tool-call
    envelopes, (b) a JSON-schema-constrained response body, (c) a free
    natural-language refusal, or any combination thereof — composed
    inside an optional ``<|channel>...<channel|>`` reasoning prefix.

    Args:
        tools: Optional list of OpenAI-style tool definitions.  Each
            element is a dict with a ``"function"`` sub-dict carrying
            ``"name"`` and ``"parameters"`` (JSON Schema), or a bare
            function dict.  If empty / None, no tool-call branch is
            emitted.
        tokens: Override the Gemma4 special-token IDs.  Default uses the
            stock Gemma4 values (48, 49, 52, 100, 101, 106).
        allow_natural_language: If True (and ``json_schema`` is None),
            allow the model to emit a natural-language refusal /
            clarification, terminated by ``<turn|>``.  Mutually
            exclusive with ``json_schema``: when both are set, the
            schema wins silently (schema replaces the nl branch
            semantically).
        allow_thinking: If True, wrap the body in an optional
            ``<|channel>thought\\n...<channel|>`` reasoning prefix.
            Use this for Gemma4 deployments with
            ``chat_template_kwargs.enable_thinking=true``.
        json_schema: Optional JSON schema (dict or pre-serialized
            string).  When set, the generator adds a ``schema_body``
            branch that matches ``%json <schema>`` followed by a stop
            token.  The model may emit a schema-valid JSON response
            instead of (or in addition to, when ``tools`` is also set)
            a tool call.

    Returns:
        A Lark grammar string starting with ``%llguidance {}``.  Pass to
        ``llguidance.LLMatcher.grammar_from_lark(...)`` to compile.

    Raises:
        ValueError: If no body branch would be emitted (i.e., ``tools``
            empty/None, ``json_schema`` None, ``allow_natural_language``
            False).
    """
    has_tools = bool(tools)
    has_schema = json_schema is not None
    if not has_tools and not has_schema and not allow_natural_language:
        raise ValueError(
            "At least one of `tools`, `json_schema`, or "
            "`allow_natural_language` must be set."
        )

    # A JSON-schema response replaces the unconstrained natural-language branch.
    effective_allow_nl = allow_natural_language and not has_schema

    tokens = tokens or Gemma4SpecialTokens()

    # Trailing stop-token class: the grammar must consume one of <eos>,
    # <turn|>, or <|tool_response> after the LAST tool-call body so the
    # engine's finish_reason transitions to "tool_calls" via a natural stop
    # rather than the matcher's accepting state. Appended ONCE by
    # ``tool_calls_seq`` (after 1+ call units) — NOT per body — so parallel
    # tool calls are admitted in a single turn.
    stop_set = (
        f"(<[{tokens.eos}]> | <[{tokens.turn_end}]> | <[{tokens.tool_response_start}]>)"
    )

    tool_alts: list[str] = []
    rule_defs: list[str] = []
    seen_safe: set[str] = set()

    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if not isinstance(fn, dict) or "name" not in fn:
            raise ValueError(
                f"Tool definition missing 'function.name' (or 'name'): {tool!r}"
            )
        name = fn["name"]
        params_spec = fn.get("parameters") or {}
        properties = params_spec.get("properties") or {}
        required = set(params_spec.get("required") or [])

        safe = _sanitize_rule_name(name)
        if safe in seen_safe:
            raise ValueError(
                f"Tool name {name!r} collides with another tool after "
                f"sanitization (both map to {safe!r})."
            )
        seen_safe.add(safe)

        body_name = f"tc_{safe}_body"
        tool_alts.append(body_name)
        body_rhs, helpers = _build_tool_rules(
            name, safe, properties, required, tokens, params_spec
        )
        rule_defs.append(f"{body_name}: {body_rhs}")
        rule_defs.extend(helpers)

    # Preserve tool-first ordering in the response alternatives.
    body_alts: list[str] = []
    if tool_alts:
        # Parallel-tool-call sequence: ONE-or-more back-to-back call-unit
        # bodies, then exactly ONE terminal stop token. Back-to-back with NO
        # separator between units — matches chat_template.jinja:244 which
        # emits ``}<tool_call|><|tool_call>call:`` (a body's closing
        # <tool_call|> immediately followed by the next body's opening
        # <|tool_call>), i.e. each call unit is ``<|tool_call> call:NAME{...}
        # <tool_call|>`` and concatenation needs nothing between them.
        # Keep the boundary exact. If the model is observed emitting newlines
        # between calls, add only a narrow ``/\n*/`` expression here; broad
        # whitespace substantially increases parser state.
        seq_inner = " | ".join(tool_alts)
        rule_defs.append(f"tool_calls_seq: ( {seq_inner} )+ {stop_set}")
        body_alts.append("tool_calls_seq")

    # Natural-language preamble THEN tool call(s) in the SAME turn.
    # Without this branch the grammar is prose-XOR-call: an assistant turn is
    # EITHER a ``tool_calls_seq`` (opens immediately at ``<|tool_call>``) OR an
    # ``nl_refusal_body`` (prose terminated by ``<turn|>``/``<eos>``). That
    # forces "narrate-not-act": the model announces "I'll transfer you" then
    # cannot emit the call that same turn. This branch admits 1+ preamble
    # tokens (SAME exclusion set as ``nl_refusal_body`` — excludes
    # tool-call delimiters, quote delimiters, turn delimiters, and channel
    # delimiters, so
    # the prose run can't swallow the call boundary (opener OR closer), the value
    # delimiter, a terminator, or open a thinking block) FOLLOWED BY a full
    # ``tool_calls_seq``.
    #   • ``+`` (not ``*``) guarantees >=1 preamble token, so this can never
    #     shadow the bare ``tool_calls_seq`` branch (which starts AT
    #     ``<|tool_call>`` with no leading prose) — unambiguous on token 1.
    #   • the prose set excludes ``<|tool_call>``(48), so the prose->call
    #     boundary is the first token-48 the model emits; and it excludes
    #     ``<turn|>``/``<eos>``, which is exactly what disambiguates this
    #     branch from ``nl_refusal_body`` (prose then 106/1, no call).
    #   • it also excludes the closer ``<tool_call|>``(49) and value delimiter
    #     ``<|"|>``(52). Natural language uses ordinary quote characters rather
    #     than token 52, so these exclusions prevent a malformed call body from
    #     being accepted as prose.
    # Gated on tools present AND nl allowed (a forced ``tool_choice`` already
    # disallows prose, so preamble stays disallowed there too — behavior
    # preserved). Ordered AFTER ``tool_calls_seq`` and BEFORE
    # ``nl_refusal_body`` in the start alternation.
    if tool_alts and effective_allow_nl:
        rule_defs.append(
            "nl_then_calls: "
            f"<[^{tokens.tool_call_start},{tokens.tool_call_end},{tokens.quote},"
            f"{tokens.turn_end},{tokens.channel_start},{tokens.eos}]>+ "
            "tool_calls_seq"
        )
        body_alts.append("nl_then_calls")

    if effective_allow_nl:
        # Natural-language refusal / clarification: any non-tool-call,
        # non-terminator tokens followed by <turn|>.  Excludes BOTH the
        # tool-call opener <|tool_call>(48), closer <tool_call|>(49), AND the
        # value-delimiter <|"|>(52) so the matcher can't ambiguously straddle the
        # tool-call branch. Natural language uses ordinary quote characters rather
        # than token 52, so a malformed call body cannot validate as prose. Also
        # excludes <|channel> so this branch cannot open a thinking block, which is
        # handled separately by the optional prefix.
        nl_rule_name = "nl_refusal_body"
        rule_defs.append(
            f"{nl_rule_name}: "
            f"<[^{tokens.tool_call_start},{tokens.tool_call_end},{tokens.quote},"
            f"{tokens.turn_end},{tokens.channel_start},{tokens.eos}]>* "
            f"(<[{tokens.turn_end}]> | <[{tokens.eos}]>)"
        )
        body_alts.append(nl_rule_name)

    if has_schema:
        # JSON-schema-constrained response body.  ``%json <schema>``
        # invokes llguidance's JSON sub-parser; ``/\n*/`` tolerates the
        # newline the model often emits between the closing ``}`` and
        # the stop token. The stop set is just ``<eos> | <turn|>`` because
        # ``<|tool_response>`` is specific to the tool-call flow.
        if isinstance(json_schema, str):
            schema_str = json_schema
        else:
            schema_str = json.dumps(json_schema, separators=(",", ":"))
        schema_rule_name = "schema_body"
        rule_defs.append(
            f"{schema_rule_name}: %json {schema_str} /\\n*/ "
            f"(<[{tokens.eos}]> | <[{tokens.turn_end}]>)"
        )
        body_alts.append(schema_rule_name)

    body_rhs = " | ".join(body_alts)

    if allow_thinking:
        # Optional thinking block: <|channel>thought\n…<channel|>.
        # The body inside accepts any tokens except the channel-close
        # token, so the matcher unambiguously knows when the block ends.
        rule_defs.append(
            f"thinking_block: <[{tokens.channel_start}]> "
            f"<[^{tokens.channel_end}]>* <[{tokens.channel_end}]>"
        )
        start_rule = f"start: thinking_block? ({body_rhs})"
    else:
        start_rule = f"start: {body_rhs}"

    # Shared open-object encoding (recursive). Always emitted; harmless
    # overhead (~7 rule defs) when unused. Used by ``_build_object_body``
    # for schemas with no declared ``properties`` or with open-ended
    # ``additionalProperties`` shapes.
    #
    # Key character set is conservative: alphanumeric/underscore bareword
    # OR Gemma4-quoted string (covers keys with spaces, punctuation).
    # Value supports the same Gemma4-native primitives the rest of the
    # grammar emits.
    q = tokens.quote

    # Content-token category for freeform string interiors: every token that is
    # NOT a Gemma4 reserved special token. A freeform value frames as
    # ``<[quote]> no_special_token* <[quote]>`` so a converged value can only
    # ever commit ordinary content — never a reserved sentinel (pad/eos/bos/unk/
    # closer/channel/turn/quote/multimodal/…) that would truncate the string
    # mid-value and leave the call unterminated/unparseable. The exclusion is the
    # FULL tokenizer special-id set (``tokens.all_special_ids`` — single
    # source of truth; an ID remap stays in sync) UNIONED with the quote id
    # (``tokens.quote``), NOT only the few named role tokens. The quote is
    # unioned in explicitly so the interior can never emit the close-quote and
    # the outer ``<[quote]>`` frame stays the SOLE close-quote terminator —
    # even for a remapped tokenizer whose ``all_special_ids`` omits its quote.
    # For the stock tokenizer 52 is already in the set, so the union is a no-op
    # and the emitted grammar is byte-identical. Named IDs only — Gemma byte-
    # fallback tokens (``<0xHH>``) stay legal string content (no range excl.).
    sentinel_ids = ",".join(
        str(t) for t in sorted(set(tokens.all_special_ids) | {tokens.quote})
    )
    rule_defs.extend(
        [
            "open_value: open_primitive | open_string | open_array | open_object",
            (
                r"open_primitive: /-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
                r"(?:[eE][+-]?[0-9]+)?/ | /(?:true|false|null)/"
            ),
            f"open_string: <[{q}]> no_special_token* <[{q}]>",
            r"open_array: /\[/ ( open_value ( /,[ \n]*/ open_value )* )? /\]/",
            r"open_object: /\{\n*/ ( open_pair ( /,[ \n]*/ open_pair )* )? /\n*\}/",
            "open_pair: open_key /:/ open_value",
            f"open_key: /[a-zA-Z_][a-zA-Z0-9_]*/ | <[{q}]> no_special_token* <[{q}]>",
            f"no_special_token: <[^{sentinel_ids}]>",
        ]
    )

    return f"%llguidance {{}}\n{start_rule}\n" + "\n".join(rule_defs) + "\n"


# Backward-compatibility alias.  The function was originally narrowly
# named for the tool-call use case; it now generates the full chat
# response grammar.  External callers should prefer
# ``generate_gemma4_chat_grammar``.
generate_gemma4_tool_grammar = generate_gemma4_chat_grammar
