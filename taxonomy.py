"""Google Product Taxonomy as an augmented n-ary tree with fast search.

Loads the full taxonomy (5,595 categories) into a tree structure where each node
carries aggregated subtree tokens for O(1) branch elimination. Supports:
  - resolve(): fast exact/fuzzy resolution of raw category strings or numeric IDs
  - classify(): beam search classification from product signals
  - subtree_categories(): all leaf categories under a path
  - broaden(): go one level up for retry
"""

import math
import re
from collections import defaultdict
from pathlib import Path

from models import VALID_CATEGORIES

# =====================================================================
# Stemmer (no external dependencies)
# =====================================================================

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "of",
        "in",
        "on",
        "to",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "its",
        "it",
        "this",
        "that",
    }
)


def _stem(word: str) -> str:
    """Simple English suffix stripping for taxonomy matching."""
    w = word.lower()
    if len(w) < 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"  # "batteries" → "battery", "accessories" → "accessory"
    if w.endswith("es") and len(w) > 4:
        pre = w[:-2]
        # -es is a proper suffix only after sibilants (ch, sh, x, ss, zz)
        if pre.endswith(("ch", "sh", "x", "ss", "zz")):
            return pre  # "watches" → "watch", "presses" → "press", "boxes" → "box"
        return w[:-1]  # "shoes" → "shoe", "houses" → "house", "laces" → "lace"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]  # "drills" → "drill", "lamps" → "lamp"
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]  # "lighting" → "light"
    return w


def _tokenize(text: str) -> list[str]:
    """Split text into stemmed tokens, removing stop words."""
    raw = re.findall(r"[a-z]+", text.lower())
    return [_stem(t) for t in raw if len(t) > 1 and t not in _STOP_WORDS]


# =====================================================================
# Tree Nodes
# =====================================================================


class TaxonomyNode:
    """A node in the taxonomy tree."""

    __slots__ = (
        "children",
        "depth",
        "full_path",
        "is_leaf",
        "leaf_tokens",
        "name",
        "parent",
        "subtree_tokens",
        "taxonomy_id",
    )

    def __init__(
        self,
        name: str,
        full_path: str,
        taxonomy_id: int | None = None,
        parent: "TaxonomyNode | None" = None,
        depth: int = 0,
    ):
        self.name = name
        self.full_path = full_path
        self.taxonomy_id = taxonomy_id
        self.parent = parent
        self.children: dict[str, TaxonomyNode] = {}
        self.is_leaf = True
        self.depth = depth
        self.leaf_tokens: set[str] = set(_tokenize(name))
        self.subtree_tokens: set[str] = set()

    def __repr__(self) -> str:
        return f"TaxonomyNode({self.full_path!r}, leaf={self.is_leaf}, children={len(self.children)})"


# =====================================================================
# Taxonomy Tree
# =====================================================================


class TaxonomyTree:
    """Augmented n-ary tree over Google's Product Taxonomy.

    Built once from categories.txt. Each node stores the union of all stemmed
    tokens in its subtree, enabling O(1) branch elimination during search.
    """

    def __init__(self, categories_file: str | Path):
        self.root = TaxonomyNode("ROOT", "", depth=-1)

        # Fast lookup indexes
        self._path_to_node: dict[str, TaxonomyNode] = {}
        self._id_to_node: dict[int, TaxonomyNode] = {}
        self._lower_to_path: dict[str, str] = {}
        self._leaf_name_to_path: dict[str, str] = {}
        self._token_idf: dict[str, float] = {}

        self._build(Path(categories_file))
        self._propagate_subtree_tokens()
        self._compute_idf()

    def _build(self, path: Path) -> None:
        """Parse categories.txt and build tree + lookup indexes."""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse "ID - Category > Path" format
                taxonomy_id = None
                if " - " in line:
                    id_str, cat_path = line.split(" - ", 1)
                    try:
                        taxonomy_id = int(id_str.strip())
                    except ValueError:
                        cat_path = line
                else:
                    cat_path = line

                segments = [s.strip() for s in cat_path.split(" > ")]

                # Walk/create tree path
                node = self.root
                for i, seg in enumerate(segments):
                    if seg not in node.children:
                        child_path = " > ".join(segments[: i + 1])
                        child = TaxonomyNode(
                            name=seg,
                            full_path=child_path,
                            parent=node,
                            depth=i,
                        )
                        node.children[seg] = child
                        node.is_leaf = False
                    node = node.children[seg]

                # Set taxonomy ID on the deepest node for this line
                if taxonomy_id is not None:
                    node.taxonomy_id = taxonomy_id
                    self._id_to_node[taxonomy_id] = node

                # Register path
                self._path_to_node[cat_path] = node
                self._lower_to_path[cat_path.lower()] = cat_path

        # Build name-to-path index.
        # Index leaf nodes first (highest priority), then non-leaf valid
        # categories when their name is unambiguous (maps to exactly one path).
        for cat_path, node in self._path_to_node.items():
            if not node.children:  # true leaf
                node.is_leaf = True
                leaf_name = node.name.lower()
                self._leaf_name_to_path[leaf_name] = cat_path
                # Also index the stemmed version
                stemmed_leaf = " ".join(_tokenize(node.name))
                if stemmed_leaf and stemmed_leaf != leaf_name:
                    self._leaf_name_to_path[stemmed_leaf] = cat_path

        # Index non-leaf valid categories by their last segment name.
        # Only add if the name doesn't conflict with an existing leaf entry.
        # E.g., "Lighting" → "Home & Garden > Lighting" (non-leaf but valid).
        ambiguous = object()  # sentinel for names mapping to multiple paths
        nonleaf_names: dict[str, str | object] = {}
        for cat_path, node in self._path_to_node.items():
            if node.children and cat_path in VALID_CATEGORIES:
                name_lower = node.name.lower()
                if name_lower in nonleaf_names:
                    nonleaf_names[name_lower] = ambiguous
                else:
                    nonleaf_names[name_lower] = cat_path
        for name_lower, cat_path in nonleaf_names.items():
            if cat_path is not ambiguous and name_lower not in self._leaf_name_to_path:
                self._leaf_name_to_path[name_lower] = cat_path

    def _propagate_subtree_tokens(self) -> None:
        """Bottom-up propagation: each node's subtree_tokens = union of all descendant tokens."""

        def _propagate(node: TaxonomyNode) -> set[str]:
            tokens = set(node.leaf_tokens)
            for child in node.children.values():
                tokens |= _propagate(child)
            node.subtree_tokens = tokens
            return tokens

        _propagate(self.root)

    def _compute_idf(self) -> None:
        """Compute IDF weights for all tokens across all categories.

        Caps IDF at 6.0 to prevent tokens appearing in very few categories
        (e.g., 'cordless' in only 'Cordless Phone Batteries') from
        overwhelming scores when they're actually modifiers, not product types.
        """
        doc_freq: dict[str, int] = defaultdict(int)
        n_docs = 0
        for node in self._path_to_node.values():
            if node.is_leaf:
                n_docs += 1
                path_tokens = set(_tokenize(node.full_path))
                for t in path_tokens:
                    doc_freq[t] += 1

        if n_docs == 0:
            return
        for token, df in doc_freq.items():
            self._token_idf[token] = min(math.log(n_docs / df), 6.0)

    # =================================================================
    # Public API
    # =================================================================

    def resolve(self, raw: str) -> str | None:
        """Try to resolve a raw category string to an exact taxonomy path.

        Strategies (fast to slow):
        1. Numeric ID lookup
        2. Exact path match
        3. Case-insensitive path match
        4. Exact leaf-term match
        5. Stemmed leaf-term match
        Returns None if no confident match found.
        """
        if not raw:
            return None

        raw = raw.strip()

        # 1. Numeric ID lookup
        if raw.isdigit():
            node = self._id_to_node.get(int(raw))
            if node:
                return node.full_path

        # 2. Exact path match
        if raw in self._path_to_node:
            return raw

        # 3. Case-insensitive
        lower = raw.lower()
        if lower in self._lower_to_path:
            return self._lower_to_path[lower]

        # 4. Exact leaf-term match
        # Try the most specific segment (after last " > ")
        leaf = raw.rsplit(" > ", 1)[-1].strip().lower()
        if leaf in self._leaf_name_to_path:
            return self._leaf_name_to_path[leaf]

        # 5. Stemmed leaf-term match
        stemmed = " ".join(_tokenize(leaf))
        if stemmed in self._leaf_name_to_path:
            return self._leaf_name_to_path[stemmed]

        return None

    def classify(
        self,
        signals: list[str],
        top_n: int = 10,
    ) -> tuple[str | None, list[str]]:
        """Classify product signals into the taxonomy.

        Uses the tree's subtree_tokens for efficient pruning — only enters
        branches that share tokens with the query. Scores all reachable leaves,
        so no branch is permanently pruned like in strict beam search.

        Tokens are weighted by signal frequency: a token mentioned by 3 independent
        signals (name, hints, breadcrumbs) scores 3x higher than one mentioned
        by only 1 signal. This naturally amplifies core product terms over accessories.

        Args:
            signals: text fragments — product name, brand, description, hints, breadcrumbs
            top_n: number of top candidates to return

        Returns:
            (confident_match_or_None, candidate_list)
            confident_match is set when the top candidate scores >= 2x the runner-up.
        """
        # Deduplicate exact signal strings (e.g. product name appearing as last breadcrumb)
        seen: set[str] = set()
        unique_signals: list[str] = []
        for s in signals:
            if s not in seen:
                seen.add(s)
                unique_signals.append(s)

        # Compute token weights: how many distinct signals mention each token
        token_weights: dict[str, float] = {}
        all_tokens: set[str] = set()
        for signal in unique_signals:
            signal_tokens = set(_tokenize(signal))
            all_tokens |= signal_tokens
            for token in signal_tokens:
                token_weights[token] = token_weights.get(token, 0) + 1.0

        if not all_tokens:
            return None, sorted(VALID_CATEGORIES)[:top_n]

        # Score all reachable leaves using tree-guided pruning
        leaf_scores: dict[str, float] = {}
        self._score_subtree(self.root, all_tokens, leaf_scores, token_weights)

        # Sort by score descending, then depth ascending (prefer shallower when tied)
        ranked = sorted(
            leaf_scores.items(),
            key=lambda x: (-x[1], x[0].count(" > ")),
        )

        if not ranked:
            return None, sorted(VALID_CATEGORIES)[:top_n]

        candidates = [path for path, _ in ranked[:top_n]]

        # Confidence check: need multiple candidates AND clear gap
        # Single-candidate results are low-confidence (might mean vocabulary gap)
        if len(ranked) >= 3 and ranked[0][1] > 0:
            ratio = ranked[0][1] / ranked[1][1] if ranked[1][1] > 0 else float("inf")
            if ratio >= 2.0:
                return ranked[0][0], candidates

        return None, candidates

    def _score_subtree(
        self,
        node: TaxonomyNode,
        query_tokens: set[str],
        leaf_scores: dict[str, float],
        token_weights: dict[str, float] | None = None,
    ) -> None:
        """Recursively score leaves, pruning branches with no token overlap."""
        if node.is_leaf:
            score = self._score_leaf(node, query_tokens, token_weights)
            if score > 0:
                leaf_scores[node.full_path] = score
            return

        for child in node.children.values():
            # O(1) branch elimination: skip subtrees with no matching tokens
            if query_tokens & child.subtree_tokens:
                self._score_subtree(child, query_tokens, leaf_scores, token_weights)

    def _score_leaf(
        self,
        node: TaxonomyNode,
        query_tokens: set[str],
        token_weights: dict[str, float] | None = None,
    ) -> float:
        """Score a leaf node against query tokens.

        Leaf-name tokens get 3x weight (they describe WHAT the category IS).
        Ancestor tokens get 1x weight (they describe WHERE it belongs).
        All weighted by IDF and signal frequency (how many signals mention the token).
        """
        path_tokens = set(_tokenize(node.full_path))
        overlap = query_tokens & path_tokens
        if not overlap:
            return 0.0

        score = 0.0
        for token in overlap:
            idf = self._token_idf.get(token, 1.0)
            tw = token_weights.get(token, 1.0) if token_weights else 1.0
            if token in node.leaf_tokens:
                score += idf * 3.0 * tw  # Leaf-position boost
            else:
                score += idf * 1.0 * tw

        return score

    def subtree_categories(self, path: str) -> list[str]:
        """Return all leaf category paths under the given path."""
        node = self._path_to_node.get(path)
        if not node:
            return []
        return self._collect_leaves(node)

    def broaden(self, path: str) -> list[str]:
        """Go one level up and return all leaf categories under the parent.

        Useful for retry: if "Hardware > Tools > Drills" didn't work,
        broaden to "Hardware > Tools" to include Saws, Sanders, etc.
        """
        node = self._path_to_node.get(path)
        if not node or not node.parent or node.parent is self.root:
            return sorted(VALID_CATEGORIES)[:200]
        return self._collect_leaves(node.parent)

    def _collect_leaves(self, node: TaxonomyNode) -> list[str]:
        """Recursively collect all leaf paths under a node."""
        if node.is_leaf and node.full_path in VALID_CATEGORIES:
            return [node.full_path]
        leaves: list[str] = []
        # Include this node itself if it's a valid category
        if node.full_path in VALID_CATEGORIES:
            leaves.append(node.full_path)
        for child in node.children.values():
            leaves.extend(self._collect_leaves(child))
        return sorted(leaves)


# =====================================================================
# Module-level singleton
# =====================================================================

_TAXONOMY_FILE = Path(__file__).parent / "categories.txt"
tree = TaxonomyTree(_TAXONOMY_FILE)


# =====================================================================
# Convenience functions (delegates to singleton)
# =====================================================================


def resolve(raw: str) -> str | None:
    """Try to resolve a raw category string to an exact taxonomy path."""
    return tree.resolve(raw)


def classify(
    signals: list[str],
    top_n: int = 10,
) -> tuple[str | None, list[str]]:
    """Classify product signals into the taxonomy."""
    return tree.classify(signals, top_n=top_n)


def subtree_categories(path: str) -> list[str]:
    """All leaf categories under a path."""
    return tree.subtree_categories(path)


def broaden(path: str) -> list[str]:
    """Go one level up, return all sibling subtree leaves."""
    return tree.broaden(path)
