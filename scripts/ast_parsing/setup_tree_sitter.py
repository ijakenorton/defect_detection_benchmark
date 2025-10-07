# from tree_sitter import Language, Parser
# import tree_sitter_c as tsc
from tree_sitter_language_pack import get_language, get_parser


def setup_treesitter():

    language = get_language("c")
    parser = get_parser("c")

    return language, parser


