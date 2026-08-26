# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gemma4 chat-format tool-call grammar generator for llguidance.

Generates a Lark grammar string (with the ``%llguidance {}`` header) that
constrains a Gemma4 model's output to the Gemma4 tool-call envelope::

    <|tool_call>call:NAME{key:value,key:<|"|>strvalue<|"|>}<tool_call|>

The grammar refers to Gemma4 special tokens by their integer token IDs, so
constraint enforcement works at the token level via ``llguidance.LLMatcher``.

The generated grammar supports multiple tools, parallel calls, optional
parameters, nested objects and arrays, local ``$ref`` values, optional
reasoning prefixes, and natural-language or JSON-schema response branches.

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

# Gemma4 special-token IDs.  Verified against the Gemma4 tokenizer's
# ``added_tokens``. The test suite re-derives these from the tokenizer
# at runtime and will fail loudly if a future Gemma4 tokenizer release
# renumbers any of them.
_TOOL_CALL_START_ID = 48  # <|tool_call>
_TOOL_CALL_END_ID = 49  # <tool_call|>
_QUOTE_ID = 52  # <|"|>
_CHANNEL_START_ID = 100  # <|channel>  (thinking-block open)
_CHANNEL_END_ID = 101  # <channel|>  (thinking-block close)
_TURN_END_ID = 106  # <turn|>  (terminator for natural-language refusal branch)
_EOS_ID = 1  # <eos>  (Gemma4 generation_config eos)
_TOOL_RESPONSE_START_ID = 50  # <|tool_response>  (post-call sentinel)


@dataclass(frozen=True)
class Gemma4SpecialTokens:
    """Gemma4 special-token IDs used by the tool-call grammar.

    A constraint-token-IDs holder specialized to the
    Gemma4 envelope (no ``<|start|>``/``<|message|>``/``<|eom|>``/``to=``).
    A dataclass so an alternate Gemma4 tokenizer with different IDs can be
    supplied without refactoring call sites.
    """

    tool_call_start: int = _TOOL_CALL_START_ID
    tool_call_end: int = _TOOL_CALL_END_ID
    quote: int = _QUOTE_ID
    channel_start: int = _CHANNEL_START_ID
    channel_end: int = _CHANNEL_END_ID
    turn_end: int = _TURN_END_ID
    eos: int = _EOS_ID
    tool_response_start: int = _TOOL_RESPONSE_START_ID


# JSON Schema primitive type -> regex pattern (no surrounding slashes).
# Strings are handled separately because they're framed in Gemma4 by the
# special ``<|"|>`` quote token, not by ASCII double-quotes.
_PRIMITIVE_PATTERNS: dict[str, str] = {
    "integer": r"-?(?:0|[1-9][0-9]*)",
    "number": r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?",
    "boolean": r"(?:true|false)",
}

# Some tool schemas use Python type names instead of canonical JSON Schema
# names. Normalize those aliases before building value rules.
_TYPE_ALIASES: dict[str, str] = {
    "float": "number",
    "int": "integer",
    "bool": "boolean",
    "dict": "object",
    "tuple": "array",
}


def _normalize_type(raw_type: Any) -> str | list[str]:
    """Map Python-style type names to canonical JSON Schema types.

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

    Gemma4 emits object arguments in a native compact syntax (``<|"|>``
    quotes, bareword keys, and comma-separated pairs), so the grammar follows
    that format rather than standard JSON.
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

        primitive: plain literal (123, 1.5, true)
        string:    <|"|>value<|"|> (Gemma4 quote token ID 52)
        object:    {key1:value1,key2:value2}
        array:     [v0,v1,...]

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

    # --- anyOf / oneOf: build an alternation over the (non-null) branches.
    # This is what makes array-of-object params legal: when `items` is
    # `{"anyOf":[{"$ref":"#/$defs/Passenger"},{"type":"object",...}]}` the
    # object branch contributes the `{...}` sub-rule, so the object-open
    # `{` becomes a grammar-legal array element. Each branch gets a distinct
    # sub-prefix so their helper rule names (perm/params) don't collide.
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
        # rejected — even though the schema admits it (an extra-key case
        # with overlapping leading keys). Drop strict-object branches
        # when an open-object branch covers them; keep ``open_object`` plus
        # any non-object branches (strings/numbers/arrays).
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
        array_expr = f"/\\[/ ( {item_expr} ( /,/ {item_expr} )* )? /\\]/"
        return array_expr, item_extras

    if ptype in _PRIMITIVE_PATTERNS:
        return f"/{_PRIMITIVE_PATTERNS[ptype]}/", []

    # String (default): Gemma4 quote-token framing.
    q = quote_id
    return f"<[{q}]> <[^{q}]>* <[{q}]>", []


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
        perm_alts.append(
            f"/,\\n*{ek}:/ {ve} {perm_name}::set_bit({idx})  %if bit_clear({idx})"
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
        # Allow ONE optional newline after the comma; Gemma4 emits
        # ``,\n`` (token 107) between params under thinking mode.
        # Restrict whitespace to newlines to keep lexer state bounded.
        perm_alts.append(
            f"/,\\n*{esc_pname}:/ {value_expr} "
            f"{perm_name}::set_bit({idx})  %if bit_clear({idx})"
        )
    perm_rule = f"{perm_name}::_  :  " + "\n         |  ".join(perm_alts)

    # Allow newlines immediately after ``{`` and before ``}``, matching the
    # formatting Gemma4 may emit in thinking mode while keeping lexer state
    # bounded.
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

    # A JSON schema replaces the unconstrained natural-language branch.
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

    # Preserve tool-first ordering in the body alternatives.
    body_alts: list[str] = []
    if tool_alts:
        # Parallel-tool-call sequence: ONE-or-more back-to-back call-unit
        # bodies, then exactly ONE terminal stop token. Back-to-back with NO
        # separator between units — matches chat_template.jinja:244 which
        # emits ``}<tool_call|><|tool_call>call:`` (a body's closing
        # <tool_call|> immediately followed by the next body's opening
        # <|tool_call>), i.e. each call unit is ``<|tool_call> call:NAME{...}
        # <tool_call|>`` and concatenation needs nothing between them.
        # Avoid broad whitespace matching between units because it expands
        # lexer state substantially.
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
    # ``<|tool_call>``,``<turn|>``,``<|channel>``,``<eos>`` so the prose run
    # can't swallow the call boundary, a terminator, or open a thinking block)
    # FOLLOWED BY a full ``tool_calls_seq``.
    #   • ``+`` (not ``*``) guarantees >=1 preamble token, so this can never
    #     shadow the bare ``tool_calls_seq`` branch (which starts AT
    #     ``<|tool_call>`` with no leading prose) — unambiguous on token 1.
    #   • the prose set excludes ``<|tool_call>``(48), so the prose->call
    #     boundary is the first token-48 the model emits; and it excludes
    #     ``<turn|>``/``<eos>``, which is exactly what disambiguates this
    #     branch from ``nl_refusal_body`` (prose then 106/1, no call).
    # Gated on tools present AND nl allowed (a forced ``tool_choice`` already
    # disallows prose, so preamble stays disallowed there too — behavior
    # preserved). Ordered AFTER ``tool_calls_seq`` and BEFORE
    # ``nl_refusal_body`` in the start alternation.
    if tool_alts and effective_allow_nl:
        rule_defs.append(
            "nl_then_calls: "
            f"<[^{tokens.tool_call_start},{tokens.turn_end},"
            f"{tokens.channel_start},{tokens.eos}]>+ "
            "tool_calls_seq"
        )
        body_alts.append("nl_then_calls")

    if effective_allow_nl:
        # Natural-language refusal / clarification: any non-tool-call,
        # non-terminator tokens followed by <turn|>.  Excludes
        # <|tool_call>=tool_call_start so the matcher can't ambiguously
        # straddle the tool-call branch.  Also excludes <|channel> so
        # this branch can't open a thinking block (handled separately
        # by the optional prefix).
        nl_rule_name = "nl_refusal_body"
        rule_defs.append(
            f"{nl_rule_name}: "
            f"<[^{tokens.tool_call_start},{tokens.turn_end},"
            f"{tokens.channel_start},{tokens.eos}]>* "
            f"(<[{tokens.turn_end}]> | <[{tokens.eos}]>)"
        )
        body_alts.append(nl_rule_name)

    if has_schema:
        # JSON-schema-constrained response body.  ``%json <schema>``
        # invokes llguidance's JSON sub-parser; ``/\n*/`` tolerates the
        # newline the model often emits between the closing ``}`` and
        # the stop token. The stop set is ``<eos> | <turn|>`` because
        # ``<|tool_response>`` is tool-flow-specific.
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
    # for any schema with no declared ``properties`` (e.g. a ``cards``
    # map, any future ``additionalProperties`` shape).
    #
    # Key character set is conservative: alphanumeric/underscore bareword
    # OR Gemma4-quoted string (covers keys with spaces, punctuation).
    # Value supports the same Gemma4-native primitives the rest of the
    # grammar emits.
    q = tokens.quote
    rule_defs.extend(
        [
            "open_value: open_primitive | open_string | open_array | open_object",
            (
                r"open_primitive: /-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
                r"(?:[eE][+-]?[0-9]+)?/ | /(?:true|false|null)/"
            ),
            f"open_string: <[{q}]> <[^{q}]>* <[{q}]>",
            r"open_array: /\[/ ( open_value ( /,/ open_value )* )? /\]/",
            r"open_object: /\{\n*/ ( open_pair ( /,\n*/ open_pair )* )? /\n*\}/",
            "open_pair: open_key /:/ open_value",
            f"open_key: /[a-zA-Z_][a-zA-Z0-9_]*/ | <[{q}]> <[^{q}]>* <[{q}]>",
        ]
    )

    return f"%llguidance {{}}\n{start_rule}\n" + "\n".join(rule_defs) + "\n"


# Backward-compatibility alias.  The function was originally narrowly
# named for the tool-call use case; it now generates the full chat
# response grammar.  External callers should prefer
# ``generate_gemma4_chat_grammar``.
generate_gemma4_tool_grammar = generate_gemma4_chat_grammar
