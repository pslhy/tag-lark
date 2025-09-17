# from .common import Rule
from .grammar import Terminal, TagTerminal, TagNonTerminal
from .lexer import Token
from collections import defaultdict
from typing import List, Set, Dict, Optional, Tuple
import sys

class RuleAnalyzer:
    def __init__(self, rules, tg_tag: str, same_structure: bool = True):
        names = {rule.origin.name for rule in rules}
        names = {name if not isinstance(name, Token) else name.value for name in names}
        self.rule_to_id = {
            ele : i for i, ele in enumerate(names)
        }
        rule_count = len(self.rule_to_id)
        self.id_to_rule = [None] * (rule_count)
        for rule_name, i in self.rule_to_id.items():
            self.id_to_rule[i] = rule_name
            # self.id_to_rule[i + self.rule_count] = rule_name + "(param)"

        
        self.NONE_TERM = rule_count
        self.PARAM_TERM = rule_count + 1
        self.TG_TERM = rule_count + 2

        # print(rule_count, self.NONE_TERM, self.PARAM_TERM, self.TG_TERM)

        edges = [set() for _ in range(rule_count)] # (to, is_tg_tag)
        param_edges = [set() for _ in range(rule_count)]
        back_edges = [set() for _ in range(rule_count  + 3)]
        param_back_edges = [set() for _ in range(rule_count + 3)]
        param_vtx = set()
        int_rules = defaultdict(list)
        for rule in rules:
            int_exp = []
            par_idx = self.rule_to_id[rule.origin.name]
            if rule.options.is_tag_rule:
                param_vtx.add(par_idx)
            for sym in rule.expansion:
                if isinstance(sym, Terminal):
                    term_idx = (
                        (
                            self.PARAM_TERM if sym.is_parameter 
                            else self.TG_TERM if sym.tag == tg_tag 
                            else self.NONE_TERM
                        ) if isinstance(sym, TagTerminal)
                        else self.NONE_TERM
                    )
                    edges[par_idx].add(term_idx)
                    back_edges[term_idx].add((par_idx, False))
                    if term_idx == self.PARAM_TERM:
                        param_back_edges[term_idx].add((par_idx, None))
                        param_edges[par_idx].add(term_idx)
                    int_exp.append(term_idx)
                else: # Non-terminal
                    child_idx = self.rule_to_id[sym.name]
                    is_tg_tag = isinstance(sym, TagNonTerminal) and sym.tag == tg_tag
                    if same_structure and isinstance(sym, TagNonTerminal) and sym.rule_tag is not None:
                        # print(sym, rule)
                        continue
                    if isinstance(sym, TagNonTerminal) and sym.is_parameter:
                        param_back_edges[child_idx].add((par_idx, is_tg_tag))
                        param_edges[par_idx].add(child_idx)

                    edges[par_idx].add(child_idx)
                    back_edges[child_idx].add((par_idx, is_tg_tag))
                    int_exp.append(child_idx)
            int_rules[par_idx].append(int_exp)
        param_reachable = self.analyze_param_terminal_reachability(param_back_edges)
        tg_reachable = self.analyze_tg_reachability(param_reachable, back_edges)
        param_replicable = self.analyze_replicability(int_rules, param_reachable, param_edges)
        tg_replicable = self.analyze_replicability(int_rules, tg_reachable, edges)

        self.param_reproducible = self.analyze_reachable(param_replicable, back_edges)
        self.tg_reproducible = self.analyze_reachable(tg_replicable, back_edges)
        self.param_reproducible = self.param_reproducible - self.tg_reproducible

        # print("PARAM REACHABLE", len(param_reachable))
        # print([self.id_to_rule[i] for i in sorted(param_reachable)])
        # print("TG REACHABLE", len(tg_reachable))
        # print([self.id_to_rule[i] for i in sorted(tg_reachable)])
        # x = param_reachable - tg_reachable
        # print("TEST", len(x))
        # print([self.id_to_rule[i] for i in sorted(x)])


        # print("PARAM REPLICABLE", len(self.param_reproducible))
        # print([self.id_to_rule[i] for i in sorted(self.param_reproducible)])
        # print("TG REPLICABLE", len(self.tg_reproducible))
        # print([self.id_to_rule[i] for i in sorted(self.tg_reproducible)])

    def analyze_param_terminal_reachability(self, edges : List[Set[int]]):
        queue_idx = 0
        queue = [self.PARAM_TERM]
        visited = set()

        while queue_idx < len(queue):
            v = queue[queue_idx]
            queue_idx += 1

            for nxt, _ in edges[v]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        return visited

    def analyze_tg_reachability(self, vtx : Set[int], edges : List[Set[Tuple[int, int]]]):
        queue_idx = 0
        queue = [self.TG_TERM]
        visited = set()

        # case for `A ::= B@tg` where B is param-reachable
        for v in vtx:
            for nxt, is_tg in edges[v]:
                if is_tg and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        while queue_idx < len(queue):
            v = queue[queue_idx]
            queue_idx += 1

            for nxt, _ in edges[v]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        
        return visited

    def analyze_scc(self, vtx: Set[int], edges: List[Set[int]]):
        visited : List[Optional[int]] = [None] * len(edges)
        scc : List[Optional[int]] = [None] * len(edges)
        call_stack : List[int] = []

        # SCC algorithm : http://boj.kr/0574f40760ef423fbb9c4b855110c20a
        cnt = 0
        scc_cnt = 0
        def dfs(v: int) -> int: 
            nonlocal cnt, scc_cnt
            cnt += 1
            nin = cnt
            visited[v] = nin
            
            call_stack.append(v)
            
            for nxt in edges[v]:
                if nxt in {self.NONE_TERM, self.PARAM_TERM, self.TG_TERM}:
                    continue
                elif visited[nxt] is None:
                    nin = min(nin, dfs(nxt))
                elif scc[nxt] is None:
                    nin = min(nin, visited[nxt])
            
            if nin == visited[v]:
                scc_cnt += 1
                while True:
                    top = call_stack.pop()
                    scc[top] = scc_cnt
                    if top == v or len(call_stack) == 0:
                        break

            return nin

        for ver in vtx:
            if visited[ver] is None:
                dfs(ver)

        return scc

    def analyze_replicability(self, rules:Dict[int, List[int]], vtx: Set[int], edges: List[Set[Tuple[int, int]]]):
        scc = self.analyze_scc(vtx, edges)
        replicable = set()
        for v in vtx:
            if scc[v] is None:
                continue
            for exp in rules[v]:
                for i, tg_gen in enumerate(exp):
                    if tg_gen not in vtx:
                        continue
                    for j, par_gen in enumerate(exp):
                        if par_gen in {self.NONE_TERM, self.PARAM_TERM, self.TG_TERM}:
                            continue
                        if i == j:
                            continue
                        if scc[par_gen] == scc[v]:
                            replicable.add(v)
                            break
                    if v in replicable:
                        break
                if v in replicable:
                    break
        return replicable


    def analyze_reachable(self, vtx: Set[int], edges: List[Set[Tuple[int, int]]], param_mode: bool = False):
        queue_idx = 0
        queue = list(vtx)
        visited = vtx.copy()

        while queue_idx < len(queue):
            v = queue[queue_idx]
            queue_idx += 1

            for nxt, _ in edges[v]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        return visited

    def is_param_reproducible(self, rule_name: str) -> bool:
        if rule_name not in self.rule_to_id:
            return False
        rule_id = self.rule_to_id[rule_name]
        return rule_id in self.param_reproducible
    
    def is_tg_reproducible(self, rule_name: str) -> bool:
        if rule_name not in self.rule_to_id:
            return False
        rule_id = self.rule_to_id[rule_name]
        return rule_id in self.tg_reproducible