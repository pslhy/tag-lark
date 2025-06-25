from copy import deepcopy, copy
from typing import Dict, Any, Generic, List, Tuple, Optional, Set
from ..grammar import NonTerminal
from ..lexer import Token, TagToken, LexerThread
from ..common import ParserCallbacks
from utils import utils

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
    
    def _get_nth_last_token(self, n: int) -> Tuple[int, Optional[Token]]:
        for i, value in enumerate(reversed(self.value_stack)):
            if isinstance(value, Token):
                if n == 0:
                    return i, value
                n -= 1
            else:
                n, token = value._get_nth_last_leaf(n)
                if token is not None:
                    return i, token
        return -1, None

    def _stack_traverse(self, idx: int, target: str, visited: Set[str]) -> Tuple[Set[Optional[str]], Set[str]]:
        # TODO: make this more efficient - graph version?
        if idx == len(self.state_stack):
            return set(), visited
        _idx = -(idx + 1)
        states = self.state_stack[_idx]
        possible_tags = set()
        found_same_depth = False

        for state in states:
            try:
                nxt = state.next
            except IndexError:
                continue
            rule_name = state.rule.alias or state.rule.options.template_source or state.rule.origin.name
            rule_name = str(rule_name)

            if not isinstance(nxt, NonTerminal) or nxt.name != target or rule_name in visited:
                continue
            found_same_depth = True
            if not getattr(nxt, 'is_parameter', False):
                possible_tags.add(getattr(nxt, 'tag', None))
            else:
                visited.add(rule_name)
                new_tags, visited = self._stack_traverse(idx, rule_name, visited)
                possible_tags.update(new_tags)
            
        if found_same_depth:
            return possible_tags, visited
        else:
            return self._stack_traverse(idx + 1, target, visited)


    def _get_possible_tag_from_state(self, idx: int) -> Set[Optional[str]]:
        _idx = -(idx + 1)
        value = self.value_stack[_idx]
        root = value.type if isinstance(value, Token) else value.data
        visited = set()
        states = self.state_stack[_idx]
        possible_tags = set()
        for state in states:
            ptr = state.index
            if ptr == 0:
                continue
            rule = state.rule
            prev_sym = rule.expansion[ptr - 1]
            if prev_sym.name != root:
                continue
            if not getattr(prev_sym, 'is_parameter', False):
                possible_tags.add(getattr(prev_sym, 'tag', None))
            else:
                parent_rule = rule.alias or rule.options.template_source or rule.origin.name
                parent_rule = str(parent_rule)
                if parent_rule not in visited:
                    visited.add(parent_rule)
                    new_tags, visited = self._stack_traverse(idx, parent_rule, visited)
                    possible_tags.update(new_tags)

        return possible_tags


    def get_nth_last_token_tag(self, n: int) -> Set[Optional[str]]:
        utils.info(f"Getting tag for the {n}th last token")
        value_idx, token = self._get_nth_last_token(n)
        utils.info(f"Value index: {value_idx}, Token: {token.type}")
        if isinstance(token, Token) or token.is_undecided:
            return self._get_possible_tag_from_state(value_idx)
        elif isinstance(token, TagToken): # value is TagTree
            return {token.tag}

        return set()

###}
