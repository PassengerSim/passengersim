import random

adjective = [
    "anonymous",
    "arbitrary",
    "clandestine",
    "cryptic",
    "enigmatic",
    "faceless",
    "impersonal",
    "indefinite",
    "indeterminate",
    "mysterious",
    "nameless",
    "obscure",
    "pseudonymous",
    "shadowy",
    "strange",
    "unbeknown",
    "undesignated",
    "undetermined",
    "undisclosed",
    "undistinguished",
    "undubbed",
    "unexplained",
    "unfamiliar",
    "unidentified",
    "unknown",
    "unlabelled",
    "unmarked",
    "unnamed",
    "unnoted",
    "unnumbered",
    "unrecognized",
    "unrevealed",
    "unsigned",
    "unspecified",
    "unstipulated",
    "untagged",
    "vague",
]

noun = [
    "backdrop",
    "canvas",
    "case",
    "circumstances",
    "circumstances",
    "conditions",
    "context",
    "details",
    "experiment",
    "particulars",
    "pattern",
    "picture",
    "plot",
    "policy",
    "problem",
    "scenario",
    "scene",
    "scoop",
    "scope",
    "setup",
    "situation",
    "sketch",
    "specifics",
    "state",
    "story",
    "structure",
    "subject",
    "synopsis",
    "thread",
    "treatment",
    "trial",
]


def random_label() -> str:
    """
    Generate a random label for a model run.

    Returns
    -------
    str
        A random label.
    """
    return f"{random.choice(adjective)} {random.choice(noun)}"


def is_random_label(label: str) -> bool:
    """
    Check if a label is a random label.

    Parameters
    ----------
    label : str
        The label to check.

    Returns
    -------
    bool
        True if the label is a random label, False otherwise.
    """
    return len(label.split()) == 2 and all(word in adjective + noun for word in label.split())
