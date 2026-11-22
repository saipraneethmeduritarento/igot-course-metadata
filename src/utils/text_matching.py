"""Text helpers that decide whether two entries written by different resources are the same thing.

The course roll-up uses them to merge presenters, reference resources and keyword short forms.
Every function is pure: it only compares or normalises text and never changes what is shown.
"""

import re

# Words that are not part of a person's name. They are ignored when comparing names, but kept in
# the name that is shown (for example "Dr. Anil Kumar").
NAME_PREFIXES = {
    "dr",
    "prof",
    "professor",
    "shri",
    "sri",
    "shree",
    "smt",
    "shrimati",
    "srimati",
    "sushri",
    "kumari",
    "km",
    "mr",
    "mrs",
    "ms",
    "miss",
    "er",
    "adv",
    "advocate",
    "justice",
    "col",
    "lt",
    "capt",
    "maj",
    "gen",
    "brig",
}
NAME_SUFFIXES = {"ias", "ips", "ifs", "irs", "ji"}

# Words skipped when building the initials of an organisation
# ("ministry of home affairs" -> "mha").
ORGANISATION_FILLER_WORDS = {"of", "and", "for", "the", "in", "on", "&"}

# The scheme and "www." at the start of a web address.
URL_PREFIX = re.compile(r"^(?:https?://)?(?:www\.)?")


def normalise_text(text: str | None) -> str | None:
    """Lower case, single spaces, no punctuation. None for a missing or blank value."""
    if not text:
        return None
    return " ".join(re.findall(r"[\w&]+", text.casefold())) or None


# --- Presenters -------------------------------------------------------------------------------


def name_key(name: str) -> tuple[str, ...]:
    """The words of a name without titles, case, punctuation or word order.

    "Dr. Anil Kumar", "anil kumar" and "Kumar, Anil" all give ("anil", "kumar").
    At least one word is always kept, even if it is a title.
    """
    words = re.findall(r"\w+", name.casefold())
    start, end = 0, len(words)
    while start < end - 1 and words[start] in NAME_PREFIXES:
        start += 1
    while end > start + 1 and words[end - 1] in NAME_SUFFIXES:
        end -= 1
    return tuple(sorted(words[start:end]))


def has_name_prefix(name: str) -> bool:
    """True if the name starts with a title such as "Dr" or "Smt"."""
    words = re.findall(r"\w+", name.casefold())
    return bool(words) and words[0] in NAME_PREFIXES


def is_acronym_of(short: str, long: str) -> bool:
    """True if `short` is the initials of the words of `long`.

    For example "nic" and "national informatics centre". Both values must already be normalised
    with `normalise_text`.
    """
    initials = "".join(word[0] for word in long.split() if word not in ORGANISATION_FILLER_WORDS)
    return " " not in short and len(short) >= 2 and short == initials


def organisations_compatible(first: str | None, second: str | None) -> bool:
    """Two normalised organisations can belong to one person unless both are given and differ.

    An acronym and its full form ("nic" and "national informatics centre") do not differ.
    """
    if first is None or second is None or first == second:
        return True
    return is_acronym_of(first, second) or is_acronym_of(second, first)


# --- Reference resources ----------------------------------------------------------------------


def reference_key(value: str, reference_type: str) -> str:
    """What makes two references the same, ignoring how they are written. Never shown.

    - A web address ignores case, `http(s)://`, a leading `www.` and trailing slashes or
      punctuation. Its path still counts.
    - A helpline keeps only its digits.
    - Anything else ignores case, punctuation and spacing.
    The type is not part of the key, so one value given two types is one reference.
    """
    value = value.strip()
    if " " not in value and "." in value:
        return "url:" + URL_PREFIX.sub("", value.casefold()).rstrip("/.,;:")
    if reference_type == "Helpline" and (digits := re.sub(r"\D", "", value)):
        return "phone:" + digits
    return "text:" + (normalise_text(value) or value.casefold())


# --- Keywords ---------------------------------------------------------------------------------


def is_short_form(text: str) -> bool:
    """True for an acronym such as "QoS" or "PDU": one word, at most 8 characters, 2+ capitals."""
    return " " not in text and len(text) <= 8 and sum(char.isupper() for char in text) >= 2
