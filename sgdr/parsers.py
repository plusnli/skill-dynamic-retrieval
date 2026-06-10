import pyparsing as pp

from dataclasses import dataclass
from typing import Any

@dataclass
class NamedArgument:
    name: str
    value: Any

    def __repr__(self):
        return f"{self.name}={repr(self.value)}"


def _build_python_subset_parser() -> pp.ParserElement:
    """Build the action parser."""
    # Literals.
    TRUE = pp.Keyword("True").setParseAction(pp.replaceWith(True))
    FALSE = pp.Keyword("False").setParseAction(pp.replaceWith(False))
    NONE = pp.Keyword("None").setParseAction(pp.replaceWith(None))
    
    LBRACK, RBRACK, LBRACE, RBRACE, LPAREN, RPAREN, COLON = map(pp.Suppress, "[]{}():")
    EQUALS = pp.Literal("=")
    
    # Operators.
    COMP_OP = pp.oneOf("< <= > >= == != in not in is is not")
    AND = pp.Keyword("and")
    OR = pp.Keyword("or")
    NOT = pp.Keyword("not")

    # Identifiers.
    identifier = pp.Word(pp.alphas + "_", pp.alphanums + "_")
    
    # Strings.
    single_string = pp.QuotedString('"') | pp.QuotedString("'")
    triple_string = (
        pp.QuotedString('"""', multiline=True) | 
        pp.QuotedString("'''", multiline=True)
    )
    fstring = (
        pp.Combine("f" + (single_string | triple_string)) |
        pp.Combine("F" + (single_string | triple_string))
    )
    string = fstring | triple_string | single_string
    
    number = pp.pyparsing_common.number()
    expression = pp.Forward()
    statement = pp.Forward()
    
    # Collections.
    list_items = pp.DelimitedList(expression, allow_trailing_delim=True)
    list_expr = pp.Group(LBRACK + pp.Optional(list_items) + RBRACK)

    # Comprehensions.
    list_comp = pp.Group(
        LBRACK + expression + 
        "for" + identifier + "in" + expression +
        pp.Optional("if" + expression) + 
        RBRACK
    )
    
    # Calls.
    arg = expression
    named_arg = (identifier + EQUALS + expression).setParseAction(
        lambda t: NamedArgument(name=t[0], value=t[2])
    )
    args = pp.DelimitedList(arg | named_arg)
    func_call = pp.Group(identifier + LPAREN + pp.Optional(args) + RPAREN).set_name("function_call").add_condition(
        lambda tokens: tokens[0] not in ['eval', 'exec', '__import__']
    )
    
    # Expressions.
    atom = (string | number | list_expr | list_comp | func_call | identifier | TRUE | FALSE | NONE)
    comparison = pp.Group(atom + pp.ZeroOrMore(COMP_OP + atom))
    not_expr = pp.Group(NOT + comparison) | comparison
    and_expr = pp.Group(not_expr + pp.ZeroOrMore(AND + not_expr))
    or_expr = pp.Group(and_expr + pp.ZeroOrMore(OR + and_expr))
    
    # Assignment.
    assignment = pp.Group(identifier + EQUALS + or_expr)
    
    # Control flow.
    for_loop = pp.Group(
        "for" + identifier + "in" + or_expr + COLON + 
        pp.IndentedBlock(statement)
    )
    
    while_loop = pp.Group(
        "while" + or_expr + COLON +
        pp.IndentedBlock(statement)
    )
    
    if_stmt = pp.Group(
        "if" + or_expr + COLON +
        pp.IndentedBlock(statement) +
        pp.ZeroOrMore(
            "elif" + or_expr + COLON +
            pp.IndentedBlock(statement)
        ) +
        pp.Optional(
            "else" + COLON +
            pp.IndentedBlock(statement)
        )
    )
    
    try_except = pp.Group(
        "try" + COLON +
        pp.IndentedBlock(statement) +
        "except" + 
        pp.Optional(identifier + pp.Optional("as" + identifier)) + 
        COLON +
        pp.IndentedBlock(statement) +
        pp.Optional(
            "else" + COLON +
            pp.IndentedBlock(statement)
        ) +
        pp.Optional(
            "finally" + COLON +
            pp.IndentedBlock(statement)
        )
    )
    
    # Finalize grammar.
    expression << or_expr
    
    statement << (if_stmt | while_loop | for_loop | try_except | assignment | expression)
    
    parser = pp.ZeroOrMore(statement)
    
    # Comments.
    single_line_comment = pp.python_style_comment
    parser.ignore(single_line_comment)
    
    return parser
