from copy import deepcopy, copy
from collections import defaultdict
from typing import Dict, Any, Generic, List, Tuple, Optional, Set
from ..grammar import NonTerminal
from ..lexer import Token, TagToken, LexerThread
from ..common import ParserCallbacks

from .grammar_analysis import StateMap
from .lalr_analysis import Shift, ParseTableBase, StateT
from lark.exceptions import UnexpectedToken

###{standalone

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
    __slots__ = 'parse_table', 'callbacks', 'start', 'start_state', 'end_state', 'states', 'tags'

    parse_table: ParseTableBase[StateT]
    callbacks: ParserCallbacks
    start: str

    start_state: StateT
    end_state: StateT
    states: Dict[StateT, Dict[str, tuple]]

    tags: List[Optional[str]]

    def __init__(self, parse_table: ParseTableBase[StateT], callbacks: ParserCallbacks, start: str, tags: List[Optional[str]]):
        self.parse_table = parse_table

        self.start_state = self.parse_table.start_states[start]
        self.end_state = self.parse_table.end_states[start]
        self.states = self.parse_table.states

        self.callbacks = callbacks
        self.start = start
        self.tags = tags


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
        self.parse_conf = parse_conf
        self.lexer = lexer
        self.state_stack = state_stack or [(self.parse_conf.start_state, 0)]
        self.value_stack = value_stack or []
        
        self.map_cache = dict()

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
                value_stack.append(0)
                return
            else:
                # reduce+shift as many times as necessary
                rule = arg
                size = len(rule.expansion)
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
                value_stack.extend(value)

                if is_end and state_stack[-1][0] == end_state:
                    return value_stack[-1]
    
    def _get_nth_last_token(self, n: int) -> int:
        n = n + 1
        token_sum = 0
        for i, (_, tokens) in enumerate(reversed(self.state_stack)):
            token_sum += tokens
            if token_sum >= n:
                return i
        return -1

    # def _stack_traverse(self, idx: int, target: str, visited: Set[str]) -> Tuple[Set[Optional[str]], Set[str]]:
    #     # TODO: make this more efficient - graph version?
    #     print("Stack traverse:", idx, target, visited)
    #     if idx == len(self.state_stack):
    #         return set(), visited
    #     _idx = -(idx + 1)
    #     states, _ = self.state_stack[_idx]
    #     if isinstance(states, int):
    #         states = self.parse_conf.parse_table.idx_to_state[states]
        
    #     possible_tags = set()
    #     found_same_depth = False

    #     for state in states:
    #         try:
    #             nxt = state.next
    #         except IndexError:
    #             continue
    #         rule_name = state.rule.origin.name
    #         rule_name = str(rule_name)

    #         if not isinstance(nxt, NonTerminal) or nxt.name != target or rule_name in visited:
    #             continue
    #         found_same_depth = True
    #         if not getattr(nxt, 'is_parameter', False):
    #             possible_tags.add(getattr(nxt, 'tag', None))
    #             print(state)
    #             print(possible_tags)
    #         else:
    #             visited.add(rule_name)
    #             new_tags, visited = self._stack_traverse(idx, rule_name, visited)
    #             if "OPEN" in new_tags:
    #                 print("Found OPEN tag in", rule_name)
    #             possible_tags.update(new_tags)
            
    #     if found_same_depth:
    #         return possible_tags, visited
    #     else:
    #         return self._stack_traverse(idx + 1, target, visited)

    def get_state_map_index_of(self, idx: int) -> int:
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


    def _get_possible_tag_from_state(self, idx: int) -> Set[Optional[str]]:
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
                # can a ruleptr be reduced in not top-element of state stack? -> False
                continue
            if prev_sym.name != root:
                assert False, f"INVARIANT FAILED: Expected {root}, got {prev_sym.name}"
            if not getattr(prev_sym, 'is_parameter', False): # SHORTCUT : clear tag
                possible_tags.add(getattr(prev_sym, 'tag', None))
            else: # tag is not clear (by param. passing)
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
                                

        return possible_tags



    def get_nth_last_token_tag(self, n: int) -> Set[Optional[str]]:
        if idx := self.value_stack[-(n+1)]:
            tags = {self.parse_conf.tags[idx]}
        else:
            idx = self._get_nth_last_token(n)
            if idx == -1:
                return set()
            tags = self._get_possible_tag_from_state(idx)
        return tags


###}
