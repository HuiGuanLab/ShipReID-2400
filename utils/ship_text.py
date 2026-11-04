# utils/ship_text.py
import json
import re


def load_jsonl_mapping(jsonl_path: str):
    """
    return dict: image_path -> json obj
    """
    mp = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = obj.get("image_path", None)
            if p:
                mp[p] = obj
    return mp


def _clean(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _is_valid_field(s: str) -> bool:
    s = _clean(s)
    if not s:
        return False
    low = s.lower()
    if low in ["none", "null", "not visible", "unknown", "n/a"]:
        return False
    return True


def _uniq_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        x = _clean(x)
        if not x:
            continue
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def _drop_redundant_specificity(phrases):
    """
    Remove redundant generic terms when more specific terms exist.
    e.g., keep 'hanging tire fenders' -> drop 'tire fenders'
    Also helps drop simple object names if a full description exists.
    """
    low_phrases = [(p, p.lower()) for p in phrases]
    drop_indices = set()

    for i, (_, small) in enumerate(low_phrases):
        for j, (_, big) in enumerate(low_phrases):
            if i == j:
                continue
            # Logic: If 'small' is a pure substring of 'big', and 'big' is not overwhelmingly long
            # we prefer 'big' (description) and drop 'small' (generic tag).
            # e.g. small="flag", big="red flag with yellow star" -> drop "flag"
            
            # Use word boundary check to avoid false positives (e.g. "car" inside "scarf")
            # Simple check: if unique words in small are a subset of big
            small_words = set(small.split())
            big_words = set(big.split())
            
            if small_words.issubset(big_words) and len(big_words) > len(small_words):
                drop_indices.add(i)
                break

    out = []
    for i, p in enumerate(phrases):
        if i not in drop_indices:
            out.append(p)
    return out


def annotation_to_texts(obj: dict, max_texts: int = 5):
    """
    Convert annotation into natural language prompts.
    Focus:
    1. Rich semantic descriptions for objects (no fragmentation).
    2. No hardcoded specific object logic.
    3. Minimal repetition.
    """
    if not obj:
        return ["A photo of a ship."][:max_texts]

    ann = (obj.get("annotation", {}) or {})

    hull_color = _clean(ann.get("hull_color", ""))
    
    # --- 1. Extract Attributes (Structured Data) ---
    attrs = []
    
    if _is_valid_field(hull_color):
        attrs.append(hull_color)

    # Fenders
    fenders = ann.get("fenders", {}) or {}
    f_den = _clean(fenders.get("density", ""))
    f_arr = _clean(fenders.get("arrangement", ""))
    if _is_valid_field(f_den):
        attrs.append(f"{f_den.lower()} fenders")
    if _is_valid_field(f_arr):
        attrs.append(f"{f_arr.lower()} fenders")

    # Cabin
    cabin = ann.get("cabin_details", {}) or {}
    cabin_struct = _clean(cabin.get("structure_description", ""))
    cabin_wd = _clean(cabin.get("windows_doors", ""))
    
    # We add specific attributes extracted from cabin strings if needed, 
    # but the full structure description is usually best kept whole in the visual cues.
    if _is_valid_field(cabin_struct):
        # Extract simple tags from complex cabin description for the "Tags" list
        lc = cabin_struct.lower()
        if "flat" in lc: attrs.append("flat roof cabin")
        if "single-story" in lc or "single story" in lc: attrs.append("single-story cabin")
        
    if _is_valid_field(cabin_wd):
        lw = cabin_wd.lower()
        if "window" in lw: attrs.append("multiple windows")
        if "porthole" in lw: attrs.append("circular porthole")
        if "door" in lw: attrs.append("cabin door")

    # Deck
    deck = ann.get("deck_info", {}) or {}
    deck_surface = _clean(deck.get("deck_surface", ""))
    if _is_valid_field(deck_surface):
        attrs.append(deck_surface)

    # --- 2. Extract Objects (Generic & Descriptive) ---
    objs = deck.get("discriminative_objects", []) or []
    obj_phrases = []

    if isinstance(objs, list):
        for it in objs:
            it = it or {}
            o = _clean(it.get("object", ""))
            d = _clean(it.get("description", ""))
            loc = _clean(it.get("location", ""))

            if not _is_valid_field(o) and not _is_valid_field(d):
                continue

            # Construct Phrase: "Description Object (Location)"
            # Strategy: Use Description. If Description doesn't contain Object name, prepend Object.
            parts = []
            
            main_desc = d if d else o
            
            # If we have a description but it doesn't mention the object name clearly, 
            # sometimes it's safer to format as "Object: Description" or just "Description object"
            # Here we try to make it natural.
            if d and o and (o.lower() not in d.lower()):
                 parts.append(f"{o}: {d}")
            else:
                 parts.append(main_desc)

            # if _is_valid_field(loc):
            #      parts.append(f"({loc})")

            full_phrase = " ".join(parts)
            obj_phrases.append(full_phrase)

    # --- 3. Construct Output Lists ---
    
    # Combine lists and clean
    # We process attrs and obj_phrases together to remove overlaps
    # (e.g. if obj is "Red flag", remove generic "Flag" attribute if it existed)
    all_tags = _uniq_keep_order(attrs + obj_phrases)
    all_tags = _drop_redundant_specificity(all_tags)

    texts = []

    # Text 1: Global Caption
    base_cap = "A photo of a ship"
    if _is_valid_field(hull_color):
        base_cap += f" with {hull_color}"
    texts.append(base_cap + ".")

    # Text 2: Visual Cues (Summary of most visual elements)
    # Priority: Cabin structure + Top object descriptions
    cues = []
    if _is_valid_field(cabin_struct):
        cues.append(f"Cabin: {cabin_struct}")
    
    # Add top 3 distinctive objects to visual cues
    cues.extend(obj_phrases[:3])
    
    if cues:
        # Clean cues to remove punctuation at end just in case
        clean_cues = [c.rstrip(".,;") for c in cues]
        texts.append("Visual cues: " + "; ".join(clean_cues) + ".")

    # Text 3: Tags (Comprehensive list)
    if all_tags:
        # Filter very long sentences from tags to keep them tag-like, 
        # but since we want description, we allow slightly longer length (e.g. 64 chars)
        short_tags = [t for t in all_tags if len(t) < 80] 
        texts.append("Tags: " + ", ".join(short_tags) + ".")

    return texts[:max_texts]