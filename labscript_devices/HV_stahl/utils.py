from labscript_utils import dedent

def split_conn_AO(connection):
    """Return analog output number of a connection string such as 'ao1' as an
    integer, or raise ValueError if format is invalid"""
    try:
        return int(connection.split('ao', 1)[1])
    except (ValueError, IndexError):
        msg = """Analog output connection string %s does not match format 'ao<N>' for
            integer N"""
        raise ValueError(dedent(msg) % str(connection))

def _ao_to_channel_name(ao_name: str) -> str:
        """ Convert 'ao0' to 'CH1' """
        try:
            channel_index = int(ao_name.replace('ao', '')) + 1
            return f'CH{channel_index}'
        except ValueError:
            raise ValueError(f"Impossible to convert from '{ao_name}'")

def _get_channel_num(channel: str) -> int:
        """Gets channel number from strings like 'AOX' or 'channel X'.
        Args:
            channel (str): The name of the channel, e.g. 'AO0', 'AO12', or 'channel 3'.

        Returns:
            int: e.g. 1..8 """
        ch_lower = channel.lower()
        if ch_lower.startswith("ao"):
            channel_num = int(channel[2:]) + 1  # 'ao0' -> '1'
        elif ch_lower.startswith("channel"):
            _, channel_num_str = channel.split()  # 'channel 1' -> '1'
            channel_num = int(channel_num_str)
        else:
            msg = """Unexpected channel name format: """
            raise ValueError(dedent(msg) % str(channel))

        return channel_num