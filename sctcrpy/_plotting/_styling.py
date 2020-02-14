def reset_plotting_profile(adata: AnnData) -> None:
    """
    Reverts plotting profile to matplotlib defaults (rcParams).  
    """
    try:
        p = _get_from_uns(adata, "plotting_profile")
    except KeyError:
        p = dict()
    p["title_loc"] = plt.rcParams["axes.titleloc"]
    p["title_pad"] = plt.rcParams["axes.titlepad"]
    p["title_fontsize"] = plt.rcParams["axes.titlesize"]
    p["label_fontsize"] = plt.rcParams["axes.labelsize"]
    p["tick_fontsize"] = plt.rcParams["xtick.labelsize"]
    _add_to_uns(adata, "plotting_profile", p)
    return


def check_for_plotting_profile(profile: Union[AnnData, str, None] = None) -> dict:
    """
    Passes a predefined set of plotting atributes to basic plotting fnctions.
    """
    profiles = {
        "vanilla": {},
        "small": {
            "figsize": (3.44, 2.58),
            "figresolution": 300,
            "title_loc": "center",
            "title_pad": 10,
            "title_fontsize": 10,
            "label_fontsize": 8,
            "tick_fontsize": 6,
        },
    }
    p = profiles["small"]
    if isinstance(profile, AnnData):
        try:
            p = _get_from_uns(profile, "plotting_profile")
        except KeyError:
            pass
    else:
        if isinstance(profile, str):
            if profile in profiles:
                p = profiles[profile]
    return p
