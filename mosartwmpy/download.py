from benedict import benedict
from mosartwmpy.utilities.download_data import download_data
from pathlib import Path
import pkg_resources

available_data = benedict.from_yaml(pkg_resources.resource_filename('mosartwmpy', 'data_manifest.yaml'))

data_list = []
data = []

for i, name in enumerate(available_data.keys()):
    data_list.append(name)
    data.append(f"""
        {i + 1}) {name} - {available_data.get(f'{name}.description')}""")

# clear the terminal
print(chr(27) + "[2J")

print(f"""
    🎶 Welcome to the mosartwmpy download utility! 🎵

       Please select the data you wish to download by typing the number:
""")

for d in data:
    print(f"""
    {d}""")

print(f"""
        
        0) exit
        
""")
try:
    user_input = int(input("""
    Please select a number and press enter: """))
except:
    pass

if not user_input or user_input == 0 or user_input > len(data):
    print("""
        
        Exiting...
        
    """)
    exit()


try:
    destination = Path(input("""
    Where would you like to download and unpack the data? (default: './'): """) or './')
except:
    print("""

            Failed to parse download directory. Exiting...

        """)
    exit()

if not destination.exists() or not destination.is_dir():
    try:
        destination.mkdir()
    except:
        print("""

                Failed to create download directory. Exiting...

            """)
        exit()


print("")
print("")
download_data(data_list[user_input - 1], destination=destination)
