"""
Compute the Boolean gradient (Jacobian) for each of the 88 non-equivalent
elementary cellular automata (ECAs).

For a rule φ(p, q, r) with p = s_{i-1}, q = s_i, r = s_{i+1}:

    ∇φ = (J_{i,i-1}, J_{i,i}, J_{i,i+1})

where each component is φ(p,q,r) ⊕ φ with that variable flipped.

Equivalence is under left-right reflection and 0/1 complementation;
we take the minimal Wolfram rule number as representative.

Output: LaTeX table in Vichniac style + human-readable summary.
"""

from sympy import symbols, true as STrue, false as SFalse
from sympy.logic.boolalg import SOPform, simplify_logic, And, Or, Not, Xor

# ---------------------------------------------------------------------------
# Boolean variables: p = left (s_{i-1}), q = center (s_i), r = right (s_{i+1})
# ---------------------------------------------------------------------------
p, q, r = symbols('p q r')


# ---------------------------------------------------------------------------
# Core ECA helpers
# ---------------------------------------------------------------------------
def rule_output(rule_num, p_val, q_val, r_val):
    """Output bit of an ECA rule for a given input triple."""
    index = 4 * p_val + 2 * q_val + r_val
    return (rule_num >> index) & 1


def reflect_rule(rule_num):
    """Reflected rule: swap left ↔ right neighbour."""
    new_rule = 0
    for pv in range(2):
        for qv in range(2):
            for rv in range(2):
                bit = rule_output(rule_num, rv, qv, pv)  # swap p↔r
                new_rule |= bit << (4 * pv + 2 * qv + rv)
    return new_rule


def complement_rule(rule_num):
    """Conjugate / complement rule: flip all inputs and output."""
    new_rule = 0
    for pv in range(2):
        for qv in range(2):
            for rv in range(2):
                bit = 1 - rule_output(rule_num, 1 - pv, 1 - qv, 1 - rv)
                new_rule |= bit << (4 * pv + 2 * qv + rv)
    return new_rule


def get_88_representatives():
    """Return sorted list of the 88 minimal-rule-number representatives."""
    visited = set()
    reps = []
    for rule in range(256):
        if rule in visited:
            continue
        ref = reflect_rule(rule)
        comp = complement_rule(rule)
        ref_comp = reflect_rule(comp)
        equiv = {rule, ref, comp, ref_comp}
        visited.update(equiv)
        reps.append(min(equiv))
    return sorted(reps)


def equivalence_class(rule):
    """Return the full equivalence class for a rule."""
    ref = reflect_rule(rule)
    comp = complement_rule(rule)
    ref_comp = reflect_rule(comp)
    return sorted({rule, ref, comp, ref_comp})


# ---------------------------------------------------------------------------
# Boolean expression construction
# ---------------------------------------------------------------------------
def truth_table_to_expr(func_values):
    """
    Convert an 8-entry truth table (indexed by 4p+2q+r) to a simplified
    sympy Boolean expression in variables (p, q, r).
    """
    minterms = []
    for pv in range(2):
        for qv in range(2):
            for rv in range(2):
                idx = 4 * pv + 2 * qv + rv
                if func_values[idx]:
                    minterms.append([pv, qv, rv])
    if len(minterms) == 0:
        return SFalse
    if len(minterms) == 8:
        return STrue
    expr = SOPform([p, q, r], minterms)
    return simplify_logic(expr, form='dnf')


def get_rule_expr(rule_num):
    """Boolean expression for φ."""
    tt = [rule_output(rule_num, pv, qv, rv)
          for pv in range(2) for qv in range(2) for rv in range(2)]
    return truth_table_to_expr(tt)


def get_gradient(rule_num):
    """
    Compute ∇φ = (J_{i,i-1}, J_{i,i}, J_{i,i+1}).
    Returns list of three sympy Boolean expressions.
    """
    jacobians = []
    for flip_idx in range(3):  # 0 = p (left), 1 = q (center), 2 = r (right)
        tt = []
        for pv in range(2):
            for qv in range(2):
                for rv in range(2):
                    inputs = [pv, qv, rv]
                    flipped = list(inputs)
                    flipped[flip_idx] = 1 - flipped[flip_idx]
                    orig = rule_output(rule_num, *inputs)
                    flip_out = rule_output(rule_num, *flipped)
                    tt.append(orig ^ flip_out)
        jacobians.append(truth_table_to_expr(tt))
    return jacobians


# ---------------------------------------------------------------------------
# LaTeX formatting — algebraic convention:
#   AND → juxtaposition,  OR → +,  NOT → \overline{}
# Variables rendered as s_{i-1}, s_i, s_{i+1}
# ---------------------------------------------------------------------------
def _needs_parens_in_product(expr):
    """Check if an expression needs parentheses when inside a product."""
    return isinstance(expr, Or)


def expr_to_latex(expr):
    """Convert a sympy Boolean expression to LaTeX algebraic notation."""
    if expr is STrue:
        return '1'
    if expr is SFalse:
        return '0'

    # Variable names → LaTeX mapping
    var_map = {
        p: r's_{i\text{-}1}',
        q: r's_i',
        r: r's_{i\text{+}1}',
    }

    # Negated variable → LaTeX (bar over letter only, not subscript)
    neg_var_map = {
        p: r'\bar{s}_{i\text{-}1}',
        q: r'\bar{s}_i',
        r: r'\bar{s}_{i\text{+}1}',
    }

    if expr in var_map:
        return var_map[expr]

    if isinstance(expr, Not):
        inner = expr.args[0]
        if inner in neg_var_map:
            return neg_var_map[inner]
        else:
            return r'\overline{' + expr_to_latex(inner) + '}'

    if isinstance(expr, And):
        parts = []
        for arg in expr.args:
            s = expr_to_latex(arg)
            if _needs_parens_in_product(arg):
                s = '(' + s + ')'
            parts.append(s)
        return ' '.join(parts)

    if isinstance(expr, Or):
        parts = [expr_to_latex(arg) for arg in expr.args]
        return ' + '.join(parts)

    if isinstance(expr, Xor):
        parts = [expr_to_latex(arg) for arg in expr.args]
        return r' \oplus '.join(parts)

    # Fallback: single symbol
    if expr in var_map:
        return var_map[expr]

    return str(expr)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
def main(include_equiv=True):
    reps = get_88_representatives()

    print(f"Found {len(reps)} equivalence class representatives.\n")

    # Collect all data
    rows = []
    for rule in reps:
        phi_expr = get_rule_expr(rule)
        grad = get_gradient(rule)
        equiv = equivalence_class(rule)
        rows.append({
            'rule': rule,
            'equiv': equiv,
            'phi': phi_expr,
            'J_left': grad[0],
            'J_center': grad[1],
            'J_right': grad[2],
        })

    # ---- Human-readable output ----
    print("=" * 100)
    print(f"{'Rule':<6} {'Equiv. class':<24} {'J_left':<22} {'J_center':<22} {'J_right':<22}")
    print("=" * 100)
    for row in rows:
        equiv_str = ','.join(str(x) for x in row['equiv'])
        jl = str(row['J_left'])
        jc = str(row['J_center'])
        jr = str(row['J_right'])
        print(f"{row['rule']:<6} {equiv_str:<24} {jl:<22} {jc:<22} {jr:<22}")
    print()

    # ---- LaTeX table output ----
    latex_lines = []

    if include_equiv:
        n_cols = 6
        col_spec = r'r l l l l l'
        header_cols = (r'Rule & Equiv.\ & $\phi(s_{i\text{-}1}, s_i, s_{i\text{+}1})$ '
                       r'& $J_{i,i-1}$ & $J_{i,i}$ & $J_{i,i+1}$ \\')
    else:
        n_cols = 5
        col_spec = r'r l l l l'
        header_cols = (r'Rule & $\phi(s_{i\text{-}1}, s_i, s_{i\text{+}1})$ '
                       r'& $J_{i,i-1}$ & $J_{i,i}$ & $J_{i,i+1}$ \\')

    latex_lines.append(r'\begin{longtable}{' + col_spec + '}')
    latex_lines.append(r'\caption{Boolean gradient $\nabla\phi = (J_{i,i-1},\, J_{i,i},\, J_{i,i+1})$ '
                       r'for each of the 88 non-equivalent elementary cellular automata. '
                       r'Algebraic notation: juxtaposition $=$ AND, $+$ $=$ OR, '
                       r'$\bar{\cdot}$ $=$ NOT.}')
    latex_lines.append(r'\label{tab:eca-gradient} \\')
    latex_lines.append(r'\toprule')
    latex_lines.append(header_cols)
    latex_lines.append(r'\midrule')
    latex_lines.append(r'\endfirsthead')
    latex_lines.append(r'\toprule')
    latex_lines.append(header_cols)
    latex_lines.append(r'\midrule')
    latex_lines.append(r'\endhead')
    latex_lines.append(r'\midrule')
    latex_lines.append(rf'\multicolumn{{{n_cols}}}{{r}}{{\textit{{Continued on next page}}}} \\')
    latex_lines.append(r'\endfoot')
    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\endlastfoot')

    for row in rows:
        r_num = row['rule']
        phi_tex = expr_to_latex(row['phi'])
        jl_tex = expr_to_latex(row['J_left'])
        jc_tex = expr_to_latex(row['J_center'])
        jr_tex = expr_to_latex(row['J_right'])

        if include_equiv:
            others = [x for x in row['equiv'] if x != r_num]
            equiv_tex = ', '.join(str(x) for x in others) if others else '---'
            latex_lines.append(
                f"  {r_num} & {equiv_tex} & ${phi_tex}$ & ${jl_tex}$ & ${jc_tex}$ & ${jr_tex}$ \\\\"
            )
        else:
            latex_lines.append(
                f"  {r_num} & ${phi_tex}$ & ${jl_tex}$ & ${jc_tex}$ & ${jr_tex}$ \\\\"
            )

    latex_lines.append(r'\end{longtable}')

    latex_str = '\n'.join(latex_lines)

    # Write LaTeX to file
    with open('eca_gradient_table.tex', 'w') as f:
        f.write(latex_str)

    print("LaTeX table written to: eca_gradient_table.tex\n")

    # Also print it
    print("--- LaTeX table ---\n")
    print(latex_str)


if __name__ == '__main__':
    import sys
    # Toggle: pass 'no-equiv' as argument to exclude equivalence column
    include_equiv = '--no-equiv' not in sys.argv
    main(include_equiv=include_equiv)