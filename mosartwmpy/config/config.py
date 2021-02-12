from benedict import benedict
from benedict.dicts import benedict as Benedict

def get_config(config_file_path: str) -> Benedict:
    """Configuration object for the model, using the Benedict type.
    
    Args:
        config_file_path (string): path to the user defined configuration yaml file
    
    Returns:
        Benedict: A Benedict instance containing the merged configuration
    """
    
    config = benedict('./config_defaults.yaml', format='yaml')
    if config_file_path and config_file_path != '':
        config.merge(benedict(config_file_path, format='yaml'), overwrite=True)
    
    return config