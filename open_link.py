import webbrowser
import subprocess

# Set this appropriately depending on your situation
chrome_path = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
first_url = ''

# This should open a NEW WINDOW with the URL specified above
command = f"cmd /C \"{chrome_path}\" {first_url} --new-window"

# Alternative way that achieves the same result
# command = f"cmd /C start chrome {first_url} --new-window"

subprocess.Popen(command)

new_tab_urls = [
    'https://www.python.org/',
    'https://hackernoon.com/30-jokes-only-programmers-will-get-a901e1cea549'
]

for url in new_tab_urls:
    webbrowser.open_new_tab(url)