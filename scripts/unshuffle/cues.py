"""Remove order cue words from captions (the leak sets in_context_nocue and original_nocue; pilot
pre-registration §1). A fixed, rule-based rewrite, applied identically to every caption set it is used on.

What counts as a cue: words that refer to other pieces or to the piece's place in the clip.
    continuation   continue(s/d/ing), keep(s) + -ing or an object, remain(s/ed/ing), resume(s), still (adverb),
                   continuously / continually
    repetition     again, once more, another (-> a / an)
    onset/offset   begin(s) / start(s) to X or + -ing, finish(es) + -ing; "begins / starts / ends with"
    position       now, next (not "next to"), final(ly), eventually, later, earlier, previously, meanwhile
Kept on purpose: before, after, then (they mostly order events inside one piece, and the isolated captions
use them as often as the in-context ones); transitive start / finish ("starts the engine"); completely.

The rewrite keeps the sentence grammatical where it can:
    continues to cut   -> cuts             continue to cut    -> cut
    continues cutting  -> is cutting       continue cutting   -> are cutting
    continues its path -> makes its path   continues down X   -> moves down X
    continues the task -> carries out the task;  finishes the task -> does the task
    the storm continues as ... -> the storm is visible as ...
    remains seated     -> is seated        the remaining logs -> the logs
    stands still (adjective) is kept; "is still dark" -> "is dark"
Telegraphic captions ("Boat continues steady drift" -> "Boat makes steady drift") come out less natural.
"""
from __future__ import annotations

import re

_IRREG_3SG = {"be": "is", "have": "has", "do": "does", "go": "goes"}
_DOUBLE = {"cut", "run", "put", "sit", "get", "stop", "drop", "step", "dig", "hit", "shut", "set", "grab", "drag",
           "plan", "skip", "slip", "spin", "trim", "chop", "wrap", "tap", "pat", "rub", "scrub", "stir", "swim",
           "begin", "nod", "jog", "hop", "flap", "clap", "drip", "grip", "strip", "snap", "trap", "pop"}
_MOTION = (r"along|down|up|through|forward|forwards|across|around|over|into|past|toward|towards|ahead|onward|"
           r"onwards|away|back|straight|off|out|uphill|downhill")
_STATIVE_NEXT = (r"as|while|at|in|on|outside|inside|for|throughout|and|under|beneath|behind|near|amid|despite|"
                 r"unabated|to|from|until|within")
_OBJ = r"its|his|her|their|the|a|an|this|that|our|my|one's|some|several"
_MODAL = r"to|will|would|can|could|may|might|must|should|shall"
_CAP = "\x01"  # marks a deleted sentence-initial word: the next letter is capitalised


def third(v: str) -> str:
    """Third person singular of a base-form verb."""
    lv = v.lower()
    if lv in _IRREG_3SG:
        return _IRREG_3SG[lv]
    if re.search(r"(s|sh|ch|x|z|o)$", lv):
        return v + "es"
    if re.search(r"[^aeiou]y$", lv):
        return v[:-1] + "ies"
    return v + "s"


def gerund(v: str) -> str:
    lv = v.lower()
    if lv == "be":
        return "being"
    if lv.endswith("ie"):
        return v[:-2] + "ying"
    if lv.endswith("e") and not re.search(r"(ee|ye|oe)$", lv):
        return v[:-1] + "ing"
    if lv in _DOUBLE:
        return v + v[-1] + "ing"
    return v + "ing"


def _sub(pattern: str, repl, text: str) -> str:
    """re.sub (case-insensitive) that keeps a capital letter: a replacement of a capitalised match is
    capitalised, and a deleted capitalised match capitalises what follows."""
    def f(m):
        out = repl(m)
        if m.group(0)[:1].isupper():
            out = out[:1].upper() + out[1:] if out.strip() else _CAP
        return out
    return re.sub(pattern, f, text, flags=re.I)


def _form(word: str, forms: dict[str, str]) -> str:
    return forms[word.lower()]


_TO = r"\s+to\s+(?P<adv>\w+ly\s+)?(?P<verb>[a-z]+)\b"
_ING = r"\s+(?P<ing>(?:\w+ly\s+)?\w+ing)\b"


def _to(form):
    return lambda m: (m.group("adv") or "") + form(m.group("verb"))


def remove_cues(text: str) -> str:
    t = text
    # video-level position: "The video begins with a view of" -> "The video shows a view of"
    t = _sub(r"\b(?:begins|starts|ends|opens|closes|concludes) with\b", lambda m: "shows", t)
    # still: an adverb of continuation, except as an adjective ("stands still", "still water")
    t = _sub(r"(?<!\bstand )(?<!\bstands )(?<!\bstanding )(?<!\bstood )(?<!\bhold )(?<!\bholds )(?<!\bholding )"
             r"(?<!\bsit )(?<!\bsits )(?<!\bsitting )(?<!\bstay )(?<!\bstays )(?<!\bstaying )(?<!\blie )"
             r"(?<!\blies )(?<!\blying )(?<!\bkeep )(?<!\bkeeps )(?<!\bremain )(?<!\bremains )(?<!\ba )(?<!\bthe )"
             r"\bstill\b(?!\s+(?:water|image|images|shot|shots|frame|photo|photos|life|lake|pond|surface|air|"
             r"picture|pictures))\s*", lambda m: "", t)
    # X to V: continue / begin / start
    for s3, ing, other in [("continues", "continuing", "continue|continued"),
                           ("begins", "beginning", "begin|began|begun"),
                           ("starts", "starting", "start|started")]:
        t = _sub(rf"\b{s3}{_TO}", _to(third), t)
        t = _sub(rf"\b{ing}{_TO}", _to(gerund), t)
        t = _sub(rf"\b(?:{other}){_TO}", _to(lambda v: v), t)
    # X + -ing: continue / keep / begin / start / finish / resume
    t = _sub(rf"\b(?P<modal>{_MODAL})\s+(?:continue|keep|begin|start|finish|resume){_ING}",
             lambda m: f"{m.group('modal')} be {m.group('ing')}", t)
    t = _sub(rf"\b(?:continues|keeps|begins|starts|finishes|resumes){_ING}", lambda m: "is " + m.group("ing"), t)
    t = _sub(rf"\b(?:continued|kept|began|started|finished|resumed){_ING}", lambda m: "was " + m.group("ing"), t)
    t = _sub(rf"\b(?:continuing|keeping|beginning|starting|finishing|resuming){_ING}", lambda m: m.group("ing"), t)
    t = _sub(rf"\b(?:continue|keep|begin|start|finish|resume){_ING}", lambda m: "are " + m.group("ing"), t)
    # continue / resume + possessive object -> make ("makes its way"); + other object -> carry out;
    # finish / complete + object -> do; keep + object -> hold
    make = {"continues": "makes", "continue": "make", "continued": "made", "continuing": "making",
            "resumes": "makes", "resume": "make", "resumed": "made", "finishes": "does", "finish": "do",
            "finished": "did", "completes": "does", "complete": "do",
            "keeps": "holds", "keep": "hold", "keeping": "holding", "kept": "held"}
    carry = {"continues": "carries out", "continue": "carry out", "continued": "carried out",
             "continuing": "carrying out", "resumes": "carries out", "resume": "carry out", "resumed": "carried out"}
    t = _sub(rf"\b({'|'.join(carry)})(?=\s+(?:the|a|an|this|that|some|several)\b)",
             lambda m: _form(m.group(1), carry), t)
    t = _sub(rf"\b({'|'.join(make)})(?=\s+(?:{_OBJ})\b)", lambda m: _form(m.group(1), make), t)
    # continue + direction -> move
    move = {"continues": "moves", "continue": "move", "continued": "moved", "continuing": "moving"}
    t = _sub(rf"\b(continues|continue|continued|continuing)(?=\s+(?:{_MOTION})\b)",
             lambda m: _form(m.group(1), move), t)
    # continue with -> show; continue as / at / ... or at the end of a clause -> is visible
    show = {"continues": "shows", "continue": "show", "continued": "showed", "continuing": "showing"}
    t = _sub(r"\b(continues|continue|continued|continuing)\s+with\b", lambda m: _form(m.group(1), show), t)
    vis = {"continues": "is visible", "continue": "are visible", "continued": "was visible",
           "continuing": "visible"}
    t = _sub(rf"\b(continues|continue|continued|continuing)(?=\s*(?:[.,;:!?)]|$)|\s+(?:{_STATIVE_NEXT})\b)",
             lambda m: _form(m.group(1), vis), t)
    # any continue left has a bare object ("continues steady drift") -> make
    make_c = {k: make[k] for k in ("continues", "continue", "continued", "continuing")}
    t = _sub(r"\b(continues|continue|continued|continuing)\b", lambda m: _form(m.group(1), make_c), t)
    t = _sub(r"\b(resumes|resume|resumed|resuming)\b",
             lambda m: {"resumes": "starts", "resume": "start", "resumed": "started",
                        "resuming": "starting"}[m.group(1).lower()], t)
    # remain
    t = _sub(r"\b(the|a|any|few|all|its|his|her|their)\s+remaining\s+", lambda m: m.group(1) + " ", t)
    t = _sub(r"\bremaining\b", lambda m: "being", t)
    t = _sub(r"\b(remains|remain|remained)\b",
             lambda m: {"remains": "is", "remain": "are", "remained": "was"}[m.group(1).lower()], t)
    # repetition
    t = _sub(r"\bone another\b", lambda m: "each other", t)
    t = _sub(r"\banother(?=\s+(\w))", lambda m: "an" if m.group(1)[0].lower() in "aeiou" else "a", t)
    t = _sub(r",?\s*\b(?:once more|once again|yet again|all over again|again)\b", lambda m: "", t)
    # adverbs of continuation and position
    t = _sub(r"\b(?:continuously|continually)\b\s*", lambda m: "", t)
    t = _sub(r"\bnow-", lambda m: "", t)
    t = _sub(r"\bnow\b\s*,?\s*", lambda m: "", t)
    t = _sub(r"\bnext\b(?!\s+to\b)\s*", lambda m: "", t)
    t = _sub(r"\b(?:finally|eventually|later|earlier|previously|meanwhile|subsequently)\b\s*,?\s*", lambda m: "", t)
    t = _sub(r"\bfinal\s+", lambda m: "", t)
    return _tidy(t)


def _tidy(t: str) -> str:
    t = re.sub(_CAP + r"\s*(\W*)(\w)", lambda m: m.group(1) + m.group(2).upper(), t).replace(_CAP, "")
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r",\s*([.;:!?])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return re.sub(r"^[\s,;]+", "", t)
