import altair as alt


def _concat(*figs, glue_func):
    if len(figs) == 1:
        return figs[0]

    configs = [fig._kwds.pop("config", alt.Config()).to_dict() for fig in figs]

    for config2 in configs[1:]:
        for k, v in config2.items():
            if (
                k in configs[0]
                and isinstance(configs[0][k], dict)
                and isinstance(v, dict)
            ):
                configs[0][k].update(v)
            else:
                configs[0][k] = v

    return glue_func(*figs).configure(**configs[0])


def hconcat(*args):
    return _concat(*args, glue_func=alt.hconcat)


def vconcat(*args):
    return _concat(*args, glue_func=alt.vconcat)
