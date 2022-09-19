import logging


def get_config_variable_name(self, config_location: str, variable: str):
    """Looks at the <config_location> list to find where the key "variable" has a value of <variable> and returns the value associated with the key "name" in the same list. 

    Args: 
        config_location (str): where in the config to search for matching key-value pairs. Config must have a list in this location
        variable (str): key "variable"'s value that we want to search for

    Returns: 
        str: key "name"'s value in the same list where the key "variable" is <variable>
    """
    try:
        name_value = next((o for o in self.config.get(config_location) if o.get('variable', '').casefold() == variable), None).get('name')
        return name_value
    except Exception as e:
        logging.error(f"{e}\nCan't find the config variable: {config_location} {variable}.")
        raise
