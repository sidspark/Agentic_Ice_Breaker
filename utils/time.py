from datetime import timedelta


def calculate_elapsed_time(start_time, end_time):
    """
    Calculate the elapsed time in hours, minutes, and seconds.

    Parameters:
    - start_time (datetime): The start time.
    - end_time (datetime): The end time.

    Returns:
    - A tuple containing hours, minutes, and seconds.
    """
    elapsed_time = end_time - start_time

    # Check if elapsed_time is a timedelta object
    if isinstance(elapsed_time, timedelta):
        hrs = elapsed_time.seconds // 3600
        mins = (elapsed_time.seconds // 60) % 60
        secs = elapsed_time.seconds % 60
    else:
        # If elapsed_time is not a timedelta, handle it as a float (assuming it's in seconds)
        hrs = int(elapsed_time // 3600)
        mins = int((elapsed_time % 3600) // 60)
        secs = round(elapsed_time % 60, 2)

    return hrs, mins, secs