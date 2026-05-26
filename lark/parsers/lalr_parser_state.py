from copy import deepcopy, copy
from collections import defaultdict
from typing import Dict, Any, Generic, List, Tuple, Optional, Set
from ..grammar import NonTerminal, TagNonTerminal, Rule, TagTerminal
from ..lexer import Token, TagToken, LexerThread
from ..common import ParserCallbacks
from ..rule_analyzer import RuleAnalyzer

from .grammar_analysis import StateMap, StateMapV2, rptr2vtx
from .lalr_analysis import Shift, ParseTableBase, StateT
from lark.exceptions import UnexpectedToken

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

    def __init__(self, parse_conf: TagParseConf[StateT], lexer: LexerThread, state_stack=None, value_stack=None):
        self.parse_conf : TagParseConf[StateT] = parse_conf
        self.lexer = lexer
        self.state_stack = state_stack or [(self.parse_conf.start_state, 0)]
        self.value_stack = value_stack or []
        
        self.last_idx = 0
        self.map_cache = dict()
        self.map_cache_v2 = dict()
        self.ra_cache = dict()

    @property
    def position(self) -> StateT:
        return self.state_stack[-1][0]

    def feed_token(self, token: Token, is_end=False) -> Any:
        state_stack = self.state_stack
        value_stack = self.value_stack
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
                else:
                    v = []
                    
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

                if is_end and state_stack[-1][0] == end_state:
                    return value_stack[-1] if len(value_stack) > 0 else None

    def fill_symbols(self, fill_type='shortest'):
        if fill_type == 'shortest':
            state_stack = self.state_stack

            top_state_map_v2 = self.get_state_map_v2_index_of(0)
            candidates = {(sym, rule, ptr) : list(rule)[ptr:] for (sym, rule, ptr) in top_state_map_v2.prepare()}

            for i in range(1, len(state_stack)):
                state_map_v2 = self.get_state_map_v2_index_of(i)
                new_candidates = {}
                for (src_sym, src_rule, src_ptr), fills in candidates.items():
                    results = state_map_v2.get_shortest_paths((src_sym, src_rule, src_ptr - 1))
                    src_dist = len(fills)
                    for (dst_sym, dst_rule, dst_ptr), syms in results.items():
                        dst_vtx = (dst_sym, dst_rule, dst_ptr)
                        if dst_vtx not in new_candidates or src_dist + len(syms) < len(new_candidates[dst_vtx]):
                            new_candidates[dst_vtx] = fills + syms
                candidates = new_candidates
            
            min_cost = None
            argmin = None
            for vtx in candidates:
                cost = len(candidates[vtx])
                if min_cost is None or cost < min_cost:
                    min_cost = cost
                    argmin = candidates[vtx]
            
            return argmin
                
        else:
            state_stack = self.state_stack

            top_state_map_v2 = self.get_state_map_v2_index_of(0)
            (sym, rule, ptr) = next(top_state_map_v2.prepare())

            fills = rule[ptr:]

            for i in range(1, len(state_stack)):
                state_map_v2 = self.get_state_map_v2_index_of(i)
                results = state_map_v2.get_paths((sym, rule, ptr - 1), check='remainder')
                for vtx in results:
                    fills = fills + results[vtx]
                    break

            return list(fills)
            

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

    def get_state_map_v2_index_of(self, idx: int) -> StateMap:
        _idx = -(idx + 1)
        states, _ = self.state_stack[_idx]
        if self.map_cache_v2.get(states) is None:
            if isinstance(states, int):
                real_states = self.parse_conf.parse_table.idx_to_state[states]
                self.map_cache_v2[states] = StateMapV2(real_states)
            else:
                self.map_cache_v2[states] = StateMapV2(states)
        return self.map_cache_v2[states]


    def can_reduce(self, vtx: Tuple[str, Tuple[str], int], idx: int) -> bool:
        top_state_map_v2 = self.get_state_map_v2_index_of(0)
        candidates = top_state_map_v2.prepare()
        for i in range(1, idx+1):
            state_map_v2 = self.get_state_map_v2_index_of(i)
            new_candidates = set()
            for (src_sym, src_rule, src_ptr) in candidates:
                results = state_map_v2.get_paths((src_sym, src_rule, src_ptr - 1))
                new_candidates = new_candidates.union(results)
            candidates = new_candidates
        return vtx in candidates

    def resolve_tagrule(self, vtx: Tuple[str, Tuple[str], int], idx: int) -> Set[str]:
        candidates = {vtx} 
        tags = set()
        for i in range(idx+1, len(self.state_stack)):
            state_map_v2 = self.get_state_map_v2_index_of(i)
            new_candidates = set()
            for (src_sym, src_rule, src_ptr) in candidates:
                results = state_map_v2.get_paths(
                    (src_sym, src_rule, src_ptr - 1), 
                    check='tagrule',
                    nondeterministic=True,
                )
                for vtx, vtx_results in results.items():
                    for found, tag in vtx_results:
                        if found:
                            tags.add(tag)
                        else:
                            new_candidates.add(vtx)
            candidates = new_candidates
        assert len(tags) > 0
        return tags

    def resolve_stag(self, vtx: Tuple[str, Tuple[str], int], idx: int, base = tuple()) -> Set[List[str]]:
        candidates = {vtx : set([base])} 
        if self.last_idx == 0:
            return set([base])

        for i in range(idx+1, len(self.state_stack) - self.last_idx + 1):
            state_map_v2 = self.get_state_map_v2_index_of(i)
            new_candidates = defaultdict(set)
            for (src_sym, src_rule, src_ptr), src_stags in candidates.items():
                results = state_map_v2.get_paths(
                    (src_sym, src_rule, src_ptr - 1), 
                    check='stag',
                    nondeterministic=True,
                )
                for vtx, vtx_stags in results.items():
                    for src_stag in src_stags:
                        for vtx_stag in vtx_stags:
                            new_candidates[vtx].add(tuple(src_stag + vtx_stag))
            candidates = new_candidates
            
        return set(stag for stags in candidates.values() for stag in stags)

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
            if idx > 0 and (ptr >= len(rule.expansion) or not self.can_reduce(rptr2vtx(state), idx)):
                # can a ruleptr be reduced in not top-element of state stack? -> False
                continue
            if prev_sym.name != root:
                assert False, f"INVARIANT FAILED: Expected {root}, got {prev_sym.name}"
            if not ignore_base and not getattr(prev_sym, 'is_parameter', False): # SHORTCUT : clear tag
                possible_tags.add(getattr(prev_sym, 'tag', None))
            elif rule.options.is_tag_rule: # tag is not clear (by param. passing)
                possible_tags = possible_tags.union(self.resolve_tagrule(rptr2vtx(state), idx))
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
            if idx > 0 and (ptr >= len(rule.expansion) or not self.can_reduce(rptr2vtx(state), idx)):
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
            possible_tags = possible_tags.union(self.resolve_stag(rptr2vtx(state), idx, base))
                                
        return possible_tags


    def get_nth_last_token_stag(self, n:int) -> Set[List[str]]:
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
                    if not self.can_reduce(rptr2vtx(state), idx):
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
