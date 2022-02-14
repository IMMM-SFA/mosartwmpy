from datetime import timedelta
from typing import Tuple
import logging

def get_experiment_name(self) -> Tuple[str, bool]:
    """Determine the experiment name from the config

    Args:
        self: config

    Returns:
        str: experiment name
        bool: if this is a new period

    """

    # Check the write frequency to see if writing to new file or appending to existing file.
    period = self.config.get('simulation.output_file_frequency')
    is_new_period = False

    # Use yesterday's date as the file name, to match with what is actually being averaged.
    true_date = self.current_time if not (self.current_time.hour == 0 and self.current_time.minute == 0 and self.current_time.second == 0) else (self.current_time - timedelta(days=1))

    # Create experiment name.
    name = f'{self.config.get("simulation.output_path")}/{self.name}/{self.name}_{true_date.year}'
    if period == 'daily':
        name += f'_{true_date.strftime("%m")}_{true_date.strftime("%d")}'
        if self.current_time.hour == 0 and self.current_time.second == 0:
            is_new_period = True
    elif period == 'monthly':
        name += f'_{true_date.strftime("%m")}'
        if self.current_time.day == 2 and self.current_time.hour == 0 and self.current_time.second == 0:
            is_new_period = True
    elif period == 'yearly':
        if self.current_time.month == 1 and self.current_time.day == 2 and self.current_time.hour == 0 and self.current_time.second == 0:
            is_new_period = True
    else:
        logging.warning(f'Configuration value for `simulation.output_file_frequency: {period}` is not recognized.')
        return
    return name, is_new_period
