from copy import deepcopy, copy
from collections import defaultdict
from typing import Dict, Any, Generic, List, Tuple, Optional, Set
from ..grammar import NonTerminal, TagNonTerminal, Rule, TagTerminal
from ..lexer import Token, TagToken, LexerThread
from ..common import ParserCallbacks
from ..rule_analyzer import RuleAnalyzer

from .grammar_analysis import StateMap
from .lalr_analysis import Shift, ParseTableBase, StateT
from ..exceptions import UnexpectedToken

###{standalone
MAX = 1<<30
class ParseConf(Generic[StateT]):
    __slots__ = 'parse_table', 'callbacks', 'start', 'start_state', 'end_state', 'states'

    parse_table: ParseTableBase[StateT]
    callbacks: ParserCallbacks
    start: str

    start_state: StateT
    end_state: StateT
    states: Dict[StateT, Dict[str, tuple]]

    def __init__(self, parse_table: ParseTableBase[StateT], callbacks: ParserCallbacks, start: str):
        self.parse_table = parse_table

        self.start_state = self.parse_table.start_states[start]
        self.end_state = self.parse_table.end_states[start]
        self.states = self.parse_table.states

        self.callbacks = callbacks
        self.start = start

class TagParseConf(Generic[StateT]):
    __slots__ = 'parse_table', 'callbacks', 'start', 'start_state', 'end_state', 'states', 'tags', 'rules'

    parse_table: ParseTableBase[StateT]
    callbacks: ParserCallbacks
    start: str

    start_state: StateT
    end_state: StateT
    states: Dict[StateT, Dict[str, tuple]]

    tags: List[Optional[str]]
    rules: List[Rule]

    def __init__(self, parse_table: ParseTableBase[StateT], callbacks: ParserCallbacks, start: str, tags: List[Optional[str]], rules: List[Rule]):
        self.parse_table = parse_table

        self.start_state = self.parse_table.start_states[start]
        self.end_state = self.parse_table.end_states[start]
        self.states = self.parse_table.states

        self.callbacks = callbacks
        self.start = start
        self.tags = tags
        self.rules = rules


class ParserState(Generic[StateT]):
    __slots__ = 'parse_conf', 'lexer', 'state_stack', 'value_stack'

    parse_conf: ParseConf[StateT]
    lexer: LexerThread
    state_stack: List[StateT]
    value_stack: list

    def __init__(self, parse_conf: ParseConf[StateT], lexer: LexerThread, state_stack=None, value_stack=None):
        self.parse_conf = parse_conf
        self.lexer = lexer
        self.state_stack = state_stack or [self.parse_conf.start_state]
        self.value_stack = value_stack or []

    @property
    def position(self) -> StateT:
        return self.state_stack[-1]

    # Necessary for match_examples() to work
    def __eq__(self, other) -> bool:
        if not isinstance(other, ParserState):
            return NotImplemented
        return len(self.state_stack) == len(other.state_stack) and self.position == other.position

    def __copy__(self):
        return self.copy()

    def copy(self, deepcopy_values=True) -> 'ParserState[StateT]':
        return type(self)(
            self.parse_conf,
            self.lexer, # XXX copy
            copy(self.state_stack),
            deepcopy(self.value_stack) if deepcopy_values else copy(self.value_stack),
        )

    def feed_token(self, token: Token, is_end=False) -> Any:
        state_stack = self.state_stack
        value_stack = self.value_stack
        states = self.parse_conf.states
        end_state = self.parse_conf.end_state
        callbacks = self.parse_conf.callbacks

        while True:
            state = state_stack[-1]
            try:
                action, arg = states[state][token.type]
            except KeyError:
                expected = {s for s in states[state].keys() if s.isupper()}
                raise UnexpectedToken(token, expected, state=self, interactive_parser=None)

            assert arg != end_state

            if action is Shift:
                # shift once and return
                assert not is_end
                state_stack.append(arg)
                value_stack.append(token if token.type not in callbacks else callbacks[token.type](token))
                return
            else:
                # reduce+shift as many times as necessary
                rule = arg
                size = len(rule.expansion)
                if size:
                    s = value_stack[-size:]
                    del state_stack[-size:]
                    del value_stack[-size:]
                else:
                    s = []

                value = callbacks[rule](s) if callbacks else s

                _action, new_state = states[state_stack[-1]][rule.origin.name]
                assert _action is Shift
                state_stack.append(new_state)
                value_stack.append(value)

                if is_end and state_stack[-1] == end_state:
                    return value_stack[-1]

class TagParserState(ParserState[StateT]):

    def __init__(
        self,
        parse_conf: TagParseConf[StateT],
        lexer: LexerThread,
        state_stack=None,
        value_stack=None,
        span_stack=None,
        reduce_events=None,
    ):
        self.parse_conf : TagParseConf[StateT] = parse_conf
        self.lexer = lexer
        self.state_stack = state_stack or [(self.parse_conf.start_state, 0)]
        self.value_stack = value_stack or []
        self.span_stack = span_stack or []
        self.reduce_events = reduce_events or []
        
        self.last_idx = 0
        self.map_cache = dict()
        self.ra_cache = dict()

    @property
    def position(self) -> StateT:
        return self.state_stack[-1][0]

    def copy(self, deepcopy_values=True) -> 'TagParserState[StateT]':
        return type(self)(
            self.parse_conf,
            self.lexer,
            copy(self.state_stack),
            deepcopy(self.value_stack) if deepcopy_values else copy(self.value_stack),
            deepcopy(self.span_stack) if deepcopy_values else copy(self.span_stack),
            deepcopy(self.reduce_events) if deepcopy_values else copy(self.reduce_events),
        )

    def reduce_event_count(self):
        return len(self.reduce_events)

    def get_reduce_events_since(self, index):
        return self.reduce_events[index:]

    def get_active_tagged_expr_contexts(self, tags=("TYPE", "NOTYPE")):
        """Return open tagged expression spans from the actual parser stack.

        A parser state can contain LR items for alternatives that were possible
        at that earlier point but are not on the concrete path taken by the
        current token prefix.  Only report a tagged expression when the stack
        suffix above that state can still reduce to the tagged expression
        symbol, matching the path-sensitive tag checks used by
        ``get_nth_last_token_tag``.
        """

        target_tags = set(tags)
        contexts = []
        token_counts = [count for _state, count in self.state_stack]
        prefix_counts = []
        total = 0
        for count in token_counts:
            total += count
            prefix_counts.append(total)

        current_end = None
        if self.span_stack:
            current_end = self.span_stack[-1][1]

        stack_len = len(self.state_stack)
        for stack_idx, (state, _count) in enumerate(self.state_stack):
            states = state
            if isinstance(states, int):
                states = self.parse_conf.parse_table.idx_to_state[states]

            token_count_before = prefix_counts[stack_idx]
            start = 0
            if token_count_before > 0 and token_count_before - 1 < len(self.span_stack):
                start = self.span_stack[token_count_before - 1][1]
            if start is None:
                continue

            end = current_end if current_end is not None else start
            if end is None or end <= start:
                continue

            seen = set()
            suffix_idx = stack_len - stack_idx - 2
            for item in states:
                ptr = item.index
                if ptr >= len(item.rule.expansion):
                    continue
                sym = item.rule.expansion[ptr]
                tag = getattr(sym, "tag", None)
                if tag not in target_tags:
                    continue
                name = getattr(sym, "name", str(sym))
                if "expr" not in name:
                    continue
                if not self.can_reduce(name, suffix_idx):
                    continue
                if self._tagged_symbol_already_completed(stack_idx, name):
                    continue
                key = (name, tag, start, end)
                if key in seen:
                    continue
                seen.add(key)
                contexts.append(
                    {
                        "symbol": name,
                        "tag": tag,
                        "start": start,
                        "end": end,
                    }
                )

        contexts.sort(key=lambda item: (item["start"], item["end"]))
        return contexts

    def _repr_symbol_name_at_stack_index(self, stack_idx):
        if stack_idx < 0 or stack_idx >= len(self.state_stack):
            return None
        reverse_idx = len(self.state_stack) - stack_idx - 1
        state_map = self.get_state_map_index_of(reverse_idx)
        repr_sym = state_map.repr_sym
        if repr_sym is None:
            return None
        return getattr(repr_sym, "name", str(repr_sym))

    def _tagged_symbol_already_completed(self, stack_idx, symbol_name):
        """Return true when this tagged symbol is below later parser input.

        LR stacks keep the state before a shifted nonterminal until the parent
        rule reduces.  If the stack element immediately above that state already
        represents the tagged expression and there are more elements after it,
        the expression has ended and the later symbols belong to the surrounding
        construct, not to an active expression prefix.
        """

        completed_idx = stack_idx + 1
        if completed_idx >= len(self.state_stack) - 1:
            return False
        return self._repr_symbol_name_at_stack_index(completed_idx) == symbol_name

    def _token_span(self, token):
        start = getattr(token, "start_pos", None)
        end = getattr(token, "end_pos", None)
        if start is None or end is None:
            return (None, None)
        return (start, end)

    def _join_spans(self, spans):
        starts = [start for start, _ in spans if start is not None]
        ends = [end for _, end in spans if end is not None]
        if not starts or not ends:
            return (None, None)
        return (min(starts), max(ends))

    def _origin_tagged_expr_children(self, rule, parent_state, spans):
        start, end = self._join_spans(spans)
        if start is None or end is None:
            return []

        state_items = parent_state
        if isinstance(state_items, int):
            state_items = self.parse_conf.parse_table.idx_to_state[state_items]

        children = []
        seen = set()
        origin_name = getattr(rule.origin, "name", str(rule.origin))
        for item in state_items:
            ptr = item.index
            if ptr >= len(item.rule.expansion):
                continue
            sym = item.rule.expansion[ptr]
            name = getattr(sym, "name", str(sym))
            if name != origin_name:
                continue
            tag = getattr(sym, "tag", None)
            if tag not in {"TYPE", "NOTYPE"}:
                continue
            if getattr(sym, "is_parameter", False):
                continue
            if "expr" not in name:
                continue

            key = (name, tag, start, end)
            if key in seen:
                continue
            seen.add(key)
            children.append(
                {
                    "symbol": name,
                    "tag": tag,
                    "start": start,
                    "end": end,
                    "kind": "origin",
                }
            )
        return children

    def _record_reduce_event(self, rule, states, spans, parent_state):
        if not spans:
            return

        origin_children = self._origin_tagged_expr_children(
            rule,
            parent_state,
            spans,
        )
        expr_children = []
        anchor = 0
        for sym, (_, length) in zip(rule.expansion, states):
            child_spans = spans[anchor : anchor + length]
            anchor += length
            tag = getattr(sym, "tag", None)
            if tag not in {"TYPE", "NOTYPE"}:
                continue
            if getattr(sym, "is_parameter", False):
                continue
            start, end = self._join_spans(child_spans)
            if start is None or end is None:
                continue
            child = {
                "symbol": getattr(sym, "name", str(sym)),
                "tag": tag,
                "start": start,
                "end": end,
                "kind": "child",
            }
            expr_children.append(child)

        expr_children = origin_children + expr_children
        type_children = [
            child for child in expr_children if child.get("tag") == "TYPE"
        ]

        if not expr_children:
            return

        start, end = self._join_spans(spans)
        self.reduce_events.append(
            {
                "origin": rule.origin.name,
                "expansion": [getattr(sym, "name", str(sym)) for sym in rule.expansion],
                "start": start,
                "end": end,
                "origin_children": origin_children,
                "type_children": type_children,
                "expr_children": expr_children,
            }
        )

    def feed_token(self, token: Token, is_end=False) -> Any:
        state_stack = self.state_stack
        value_stack = self.value_stack
        span_stack = self.span_stack
        states = self.parse_conf.states
        end_state = self.parse_conf.end_state
        callbacks = self.parse_conf.callbacks
        

        while True:
            state, tokens = state_stack[-1]
            try:
                action, arg = states[state][token.type]
            except KeyError:
                expected = {s for s in states[state].keys() if s.isupper()}
                raise UnexpectedToken(token, expected, state=self, interactive_parser=None)

            assert arg != end_state

            if action is Shift:
                # shift once and return
                assert not is_end
                state_stack.append((arg, 1))
                if self.last_idx == 0:
                    states_ = arg if not isinstance(arg, int) else self.parse_conf.parse_table.idx_to_state[arg]
                    for s in states_:
                        # print(s)
                        for sym in s.rule.expansion:
                            if getattr(sym, 'rule_tag', None) is not None:
                                self.last_idx = len(self.state_stack)
                                break
                        if self.last_idx != 0:
                            break
                value_stack.append((-1, -1))
                span_stack.append(self._token_span(token))
                return
            else:
                # reduce+shift as many times as necessary
                rule = arg
                size = len(rule.expansion)
                if len(state_stack) - size < self.last_idx:
                    self.last_idx = 0
                    # print(self.last_idx)
                if size:
                    s = state_stack[-size:]
                    del state_stack[-size:]
                else:
                    s = []
                
                token_sum = 0
                for _, t in s:
                    token_sum += t
                if token_sum:
                    v = value_stack[-token_sum:]
                    del value_stack[-token_sum:]
                    span_values = span_stack[-token_sum:]
                    del span_stack[-token_sum:]
                else:
                    v = []
                    span_values = []
                    
                self._record_reduce_event(
                    rule,
                    s,
                    span_values,
                    state_stack[-1][0],
                )
                value = callbacks[rule](v, s) if callbacks else v

                _action, new_state = states[state_stack[-1][0]][rule.origin.name]
                assert _action is Shift
                state_stack.append((new_state, token_sum))
                if self.last_idx == 0:
                    states_ = new_state if not isinstance(new_state, int) else self.parse_conf.parse_table.idx_to_state[new_state]
                    for s in states_:
                        # print(s)
                        for sym in s.rule.expansion:
                            if getattr(sym, 'rule_tag', None) is not None:
                                self.last_idx = len(self.state_stack)
                                # print(self.last_idx)
                                break
                        if self.last_idx != 0:
                            break
                value_stack.extend(value)
                span_stack.extend(span_values)

                if is_end and state_stack[-1][0] == end_state:
                    return value_stack[-1] if len(value_stack) > 0 else None
    
    def fill_symbols(self):
        state_stack = self.state_stack
        states = self.parse_conf.states

        filled = []
        while True:
            state, _ = state_stack[-1]
            if isinstance(state, int):
                state = self.parse_conf.parse_table.idx_to_state[state]
            ptr = None
            for s in state:
                if s.index > 0:
                    if ptr is None:
                        ptr = s
                    else:
                        if s.rule.origin.name in [x.name for x in s.rule.expansion[:s.index]]:
                            continue
                        terms_to_fill = lambda p: len(p.rule.expansion) - p.index
                        if terms_to_fill(s) <= terms_to_fill(ptr):
                            ptr = s
            
            rule = ptr.rule
            if rule.origin.name == self.parse_conf.start:
                break

            if ptr is None:
                assert False, "No valid ptr to fill symbols"

            symbols = ptr.rule.expansion[ptr.index:]
            filled.extend(symbols)

            del state_stack[-ptr.index:]

            # print(filled)
            # print(ptr)
            # if len(symbols) > 0:
            #     print(symbols)
            
            _action, new_state = states[state_stack[-1][0]][rule.origin.name]
            assert _action is Shift
            state_stack.append((new_state, 0))

        filled = [x.name for x in filled]
        return filled


    def _get_nth_last_token(self, n: int) -> int:
        n = n + 1
        token_sum = 0
        for i, (_, tokens) in enumerate(reversed(self.state_stack)):
            token_sum += tokens
            if token_sum >= n:
                return i
        return -1

    def get_state_map_index_of(self, idx: int) -> StateMap:
        _idx = -(idx + 1)
        states, _ = self.state_stack[_idx]
        if self.map_cache.get(states) is None:
            if isinstance(states, int):
                real_states = self.parse_conf.parse_table.idx_to_state[states]
                self.map_cache[states] = StateMap(real_states)
            else:
                self.map_cache[states] = StateMap(states)
        return self.map_cache[states]

    def parent_check(self, tg_sym: str, idx: int, leaf: str) -> bool:
        state_map = self.get_state_map_index_of(idx)
        if leaf == tg_sym:
            return True
        path = state_map.get_roots(leaf)    
        if tg_sym in path:
            return True
        return False

    def can_reduce(self, tg_sym: str, idx: int) -> bool:
        if idx == -1:
            return True
        
        state_map = self.get_state_map_index_of(idx)

        if tg_sym.isupper(): # is terminal?
            for ruleptr in state_map.repr_ruleptr:
                ptr = ruleptr.index
                sym = ruleptr.rule.expansion[ptr-1].name
                # check if target_symbol is equals to represent_symbol of current state-map.
                return sym == tg_sym

        for ruleptr in state_map.repr_ruleptr:
            rule_name = str(ruleptr.rule.origin.name)
            ptr = ruleptr.index
            if idx > 0 and ptr >= len(ruleptr.rule.expansion): # almost-reduce check
                continue
            elif ptr > 1: # shift-from-past check
                sym = ruleptr.rule.expansion[ptr-1].name
                if tg_sym != sym:
                    continue
                return True
            new_tg_sym = ruleptr.rule.expansion[ptr].name if ptr < len(ruleptr.rule.expansion) else None
            if not self.can_reduce(new_tg_sym, idx - 1): # can future symbol be reduced? -> if so, current ruleptr can be reduced.
                continue
                
            if self.parent_check(tg_sym, idx + 1, rule_name): # does reducing current ruleptr affect to target symbol?
                return True

        return False

    def _get_possible_tag_from_state(self, idx: int, ignore_base: bool = False) -> Set[Optional[str]]:
        _idx = -(idx + 1)
        states, _ = self.state_stack[_idx]
        if isinstance(states, int):
            states = self.parse_conf.parse_table.idx_to_state[states]
        root = None
        for state in states:
            if state.index > 0:
                root = state.rule.expansion[state.index - 1].name
                break
        assert root is not None

        possible_tags = set()
        for state in states:
            ptr = state.index
            if ptr == 0:
                continue
            rule: Rule = state.rule
            prev_sym = rule.expansion[ptr - 1]
            if idx > 0 and (ptr >= len(rule.expansion) or not self.can_reduce(rule.expansion[ptr].name, idx - 1)):
                # can a ruleptr be reduced in not top-element of state stack? -> False
                continue
            if prev_sym.name != root:
                assert False, f"INVARIANT FAILED: Expected {root}, got {prev_sym.name}"
            if not ignore_base and not getattr(prev_sym, 'is_parameter', False): # SHORTCUT : clear tag
                possible_tags.add(getattr(prev_sym, 'tag', None))
            elif rule.options.is_tag_rule: # tag is not clear (by param. passing)
                par_rule = str(rule.origin.name)
                queue_depth = defaultdict(set)
                depth = ptr
                max_depth = depth
                queue_depth[ptr].add(par_rule) # don't need to call get_roots() - if ptr > 0, already root

                while depth <= max_depth:
                    state_map = self.get_state_map_index_of(idx + depth)
                    for leaf in queue_depth[depth]:
                        goals, tags =  state_map.get_roots(leaf, use_tag_edges=True)
                        for tag, sym in tags:
                            if self.can_reduce(sym, idx + depth - 1):
                                possible_tags.add(tag)
                        for goal, dep in goals:
                            nxt_depth = depth + dep
                            max_depth = max(nxt_depth, max_depth)
                            queue_depth[nxt_depth].add(goal)
                    depth += 1
            else:
                possible_tags.add(None)                    

        return possible_tags



    def get_nth_last_token_tag(self, n: int) -> Set[Optional[str]]:
        if (idx := self.value_stack[-(n+1)][0]) >= 0:
            tags = {self.parse_conf.tags[idx]}
        else:
            idx = self._get_nth_last_token(n)
            if idx == -1:
                return set()
            tags = self._get_possible_tag_from_state(idx)
        return tags

    def _get_possible_stag_from_state(self, idx: int, ignore_base: bool = False) -> Set[List[str]]:
        _idx = -(idx + 1)
        states, _ = self.state_stack[_idx]
        if isinstance(states, int):
            states = self.parse_conf.parse_table.idx_to_state[states]
        root = None
        for state in states:
            if state.index > 0:
                root = state.rule.expansion[state.index - 1].name
                break
        assert root is not None

        possible_tags = set()
        for state in states:
            ptr = state.index
            if ptr == 0:
                continue
            rule = state.rule
            prev_sym = rule.expansion[ptr - 1]
            if idx > 0 and (ptr >= len(rule.expansion) or not self.can_reduce(rule.expansion[ptr].name, idx - 1)):
                # can a ruleptr in not top-element of state stack be reduced? -> False
                continue
            if prev_sym.name != root:
                assert False, f"INVARIANT FAILED: Expected {root}, got {prev_sym.name}"
            # print(state, "PASS")
            base = None
            if isinstance(prev_sym, TagNonTerminal):
                base = prev_sym.rule_tag
            
            if ignore_base or base is None:
                base = tuple()
            else:
                base = tuple([base])
            # print(state, base)
            par_rule = str(rule.origin.name)
            queue_depth = defaultdict(lambda: defaultdict(set))
            depth = ptr
            max_depth = depth
            queue_depth[ptr][par_rule].add(base) # don't need to call get_roots() - if ptr > 0, already root

            while depth <= max_depth:
                if idx + depth + self.last_idx - 1 > len(self.state_stack):
                    for leaf, base_tag in queue_depth[depth].items():
                        for bt in base_tag:
                            possible_tags.add(bt) 
                    break
                state_map = self.get_state_map_index_of(idx + depth)
                for leaf, base_tag in queue_depth[depth].items():
                    findings = state_map.find_rule_tag(leaf)
                    for sym, stag, ptr in findings:
                        # print("-", sym, stag, ptr)
                        if self.can_reduce(sym, idx + depth - 1):
                            # print(ptr > 0, ptr)
                            if ptr > 0:
                                nxt_depth = depth + ptr
                                max_depth = max(nxt_depth, max_depth)
                                # print(nxt_depth)
                                for bt in base_tag:
                                    queue_depth[nxt_depth][sym].add(tuple(list(bt) + list(stag)))
                            else:
                                for bt in base_tag:
                                    possible_tags.add(tuple(list(bt) + list(stag)))                
                depth += 1
                                
        return possible_tags


    def get_nth_last_token_stag(self, n:int) -> Set[List[str]]:
        # base = []
        # if (rule_tag_idx := self.value_stack[-(n+1)][1]) >= 0:
        #     base.append(
        #         self.parse_conf.rule_tags[rule_tag_idx]
        #     )
        
        idx = self._get_nth_last_token(n)
        if idx == -1:
            return set()
        stags = self._get_possible_stag_from_state(idx)
        stags = set(tuple(reversed(t)) for t in stags)
        return stags
            
    def get_coming_term_stag(self, tag: str, ra: RuleAnalyzer) -> Set[List[str]]:
        possible_stags = set()
        for idx, (states, _) in enumerate(reversed(self.state_stack)):
            if idx + self.last_idx - 1 > len(self.state_stack):
                break
            par_stag = (
                self._get_possible_stag_from_state(idx, ignore_base=True)
                if idx + 1 < len(self.state_stack) else set([tuple()])
            )
            # print(idx,  par_stag)
            first_loop = idx == 0
            if isinstance(states, int):
                states = self.parse_conf.parse_table.idx_to_state[states]
            for state in states:
                ptr = state.index
                # if idx + 1 < len(self.state_stack) and ptr == 0:
                #     continue
                # print(idx, state)
                if not first_loop:
                    if ptr >= len(state.rule.expansion):
                        continue
                    sym = state.rule.expansion[ptr].name
                    if not self.can_reduce(sym, idx - 1):
                        continue
                # print("PASS")
                exp_len = len(state.rule.expansion)
                start_ptr = ptr if first_loop else ptr + 1 # if not first_loop, assume that pointed symbol is reduced.
                for i in range(start_ptr, exp_len):
                    cur_sym = state.rule.expansion[i]
                    is_reproducible = ra.is_tg_reproducible(cur_sym.name)
                    if not is_reproducible and getattr(cur_sym, 'is_parameter', False):
                        is_reproducible = (
                            ra.is_param_reproducible(cur_sym.name)
                            and tag in self._get_possible_tag_from_state(idx, ignore_base=True)
                        )
                    if is_reproducible:
                        cur_stag = getattr(cur_sym, 'rule_tag', None)
                        for bt in par_stag:
                            if cur_stag is None:
                                possible_stags.add(bt)
                            else:
                                possible_stags.add(tuple([cur_stag] + list(bt)))
        return possible_stags

    def can_come_term_with_tag(self, tag: str) -> bool:
        rule_analyzer = None
        if self.ra_cache.get(tag) is None:
            rule_analyzer = RuleAnalyzer(self.parse_conf.rules, tag)
            self.ra_cache[tag] = rule_analyzer
        else:
            rule_analyzer = self.ra_cache[tag]

        return self.get_coming_term_stag(tag, rule_analyzer)
###}
