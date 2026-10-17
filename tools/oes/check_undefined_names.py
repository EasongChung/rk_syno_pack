#!/usr/bin/env python3
"""静态检查：函数内引用了未定义的自由名（漏 import / 提取函数时漏搬）。

## 为什么需要
连续两轮犯同一类错：
  1. regress.py 提取 preprocess_input 后，run_once 里遗留 `shape`
     -> NameError（CI run 39004180722，掩盖了一整轮 regress）
  2. 同一次改动把 `import numpy as np` 也搬走，run_once 的 DBG 块仍用 np
     -> NameError（CI run 39005770479）

两次都是「本地无法跑 TF -> 靠 CI 才发现」。本检查不依赖 TF，
纯 AST，能在 push 前拦住。

用法：python check_undefined_names.py tools/oes/*.py
退出码 5 = 发现可疑未定义名。
"""
import ast
import builtins
import sys


def module_level_names(tree):
    names = set(dir(builtins))
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                names.add((a.asname or a.name).split('.')[0])
        elif isinstance(n, ast.Assign):
            for tg in n.targets:
                if isinstance(tg, ast.Name):
                    names.add(tg.id)
                elif isinstance(tg, ast.Tuple):
                    for e in tg.elts:
                        if isinstance(e, ast.Name):
                            names.add(e.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, ast.Try):
            for sub in n.body + [x for h in n.handlers for x in h.body]:
                if isinstance(sub, ast.Assign):
                    for tg in sub.targets:
                        if isinstance(tg, ast.Name):
                            names.add(tg.id)
    return names


def collect_bound(node):
    """收集某作用域内所有被绑定的名字（含嵌套函数的形参/局部）。"""
    bound = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                bound.add((a.asname or a.name).split('.')[0])
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.keyword) and n.arg:
            bound.add(n.arg)
        elif isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp,
                            ast.GeneratorExp)):
            for gen in n.generators:
                for t in ast.walk(gen.target):
                    if isinstance(t, ast.Name):
                        bound.add(t.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.ClassDef)):
            bound.add(n.name)
    return bound


def check_func(fn, visible, filename):
    """visible = 模块级名 + 所有外层作用域已绑定名（闭包可见）。"""
    bound = collect_bound(fn)
    used = {n.id for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = sorted(x for x in used - bound - visible
                     if not (x.startswith('__') and x.endswith('__')))
    if missing:
        print('%s:%d %s -> possibly undefined: %s'
              % (filename, fn.lineno, fn.name, missing))
    # 递归：嵌套函数可见外层（含当前层）所有绑定
    inner_visible = visible | bound
    for n in ast.walk(fn):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn:
            check_func(n, inner_visible, filename)
    return len(missing)


def main():
    bad = 0
    for path in sys.argv[1:]:
        try:
            tree = ast.parse(open(path, 'rb').read().decode('utf-8'))
        except SyntaxError as e:
            print('%s: SyntaxError %s' % (path, e))
            bad += 1
            continue
        mn = module_level_names(tree)
        # 只从**顶层**函数进入，嵌套函数由 check_func 递归处理
        # （否则嵌套函数会被顶层循环按其自身检查一遍，丢掉闭包可见名，
        #  产生大量误报）。
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bad += check_func(n, mn, path)
    print('check_undefined_names: %d suspicious name(s)' % bad)
    return 5 if bad else 0


if __name__ == '__main__':
    sys.exit(main())