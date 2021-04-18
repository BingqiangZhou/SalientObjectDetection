# conv = "groupconv"
conv = "dsconv"
# conv = "conv"

if conv == "conv":
    from u2net import U2NET, U2NETP
elif conv == "groupconv":
    from u2net_groupconv import U2NET, U2NETP
elif conv == "dsconv":
    from u2net_dsconv import U2NET, U2NETP