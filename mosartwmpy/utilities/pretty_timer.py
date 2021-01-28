def pretty_timer(seconds):
    # format elapsed times in a human friendly way
    if seconds < 1:
        return f'{round(seconds * 1.0e3, 0)} milliseconds'
    elif seconds < 60:
        return f'{round(seconds, 3)} seconds'
    elif seconds < 3600:
        return f'{int(round(seconds) // 60)} minutes and {int(round(seconds) % 60)} seconds'
    elif seconds < 86400:
        return f'{int(round(seconds) // 3600)} hours, {int((round(seconds) % 3600) // 60)} minutes, and {int(round(seconds) % 60)} seconds'
    else:
        return f'{int(round(seconds) // 86400)} days, {int((round(seconds) % 86400) // 3600)} hours, and {int((round(seconds) % 3600) // 60)} minutes'